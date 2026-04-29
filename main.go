package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/joho/godotenv"
	_ "golang.org/x/image/webp"
)

// --- CONFIGURATION VARS ---
var ImmichHostIP string
var ImmichAPIKey string
var OllamaHost string
var OllamaModel string
var BenchmarkMode bool
var VerboseMode bool
var WatchMode bool
var WatchInterval time.Duration
var CleanJunk string // "", "dry-run", or "true"

// Derived URLs
var ImmichBaseURL string
var PostgresURL string

// Structs
type ChatRequest struct {
	Model    string                 `json:"model"`
	Messages []Message              `json:"messages"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options"`
}

type Message struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type ChatResponse struct {
	Message struct {
		Content string `json:"content"`
	}
	Done bool `json:"done"`
}

func main() {
	// 1. Load .env
	if err := godotenv.Load(); err != nil {
		// It's okay if .env doesn't exist, we'll check env vars or defaults
	}

	// 2. Define Defaults from ENV
	envImmichHost := getEnv("IMMICH_HOST", "127.0.0.1")
	envImmichKey := getEnv("IMMICH_API_KEY", "")
	envOllamaHost := getEnv("OLLAMA_HOST", "http://localhost:11434")
	envOllamaModel := getEnv("OLLAMA_MODEL", "minicpm-v:latest")
	envDBUser := getEnv("DB_USER", "postgres")
	envDBPass := getEnv("DB_PASS", "postgres")
	envDBName := getEnv("DB_NAME", "immich")
	envDBPort := getEnv("DB_PORT", "5432")
	envDBHost := getEnv("DB_HOST", envImmichHost)
	envImmichPort := getEnv("IMMICH_PORT", "2283")
	envWatchInterval := getEnv("WATCH_INTERVAL", "1m")
	CleanJunk = strings.ToLower(strings.TrimSpace(getEnv("CLEAN_JUNK", "")))

	// 3. Define Flags (override ENV)
	flag.StringVar(&ImmichHostIP, "host", envImmichHost, "Immich Host IP")
	flag.StringVar(&ImmichAPIKey, "key", envImmichKey, "Immich API Key")
	flag.StringVar(&OllamaHost, "ollama", envOllamaHost, "Ollama Server URL")
	flag.StringVar(&OllamaModel, "model", envOllamaModel, "Ollama model to use")
	
	var intervalStr string
	flag.StringVar(&intervalStr, "interval", envWatchInterval, "Watch interval (e.g. 1m, 1h)")
	flag.BoolVar(&WatchMode, "watch", false, "Run in watcher mode (poll for new images)")
	
	flag.BoolVar(&BenchmarkMode, "benchmark", false, "Run benchmark mode")
	flag.BoolVar(&VerboseMode, "verbose", false, "Print full description to terminal")
	flag.Parse()

	var err error
	WatchInterval, err = time.ParseDuration(intervalStr)
	if err != nil {
		log.Fatalf("Invalid interval format: %v", err)
	}

	// 4. Construct Derived URLs
	ImmichBaseURL = fmt.Sprintf("http://%s:%s", ImmichHostIP, envImmichPort)
	// Use envDBHost for Postgres, but if user overrides -host flag, should we respect that for DB too if DB_HOST wasn't explicitly set?
	// Simplest logic: If DB_HOST is set in env, use it. If not, use the final ImmichHostIP (which might be from flag).
	
	// Re-evaluate DB Host logic after flags
	finalDBHost := envDBHost
	if os.Getenv("DB_HOST") == "" {
		// If explicit DB_HOST wasn't provided in env, assume it follows the Immich Host (even if changed via flag)
		finalDBHost = ImmichHostIP
	}

	PostgresURL = fmt.Sprintf("postgres://%s:%s@%s:%s/%s", url.PathEscape(envDBUser), url.PathEscape(envDBPass), finalDBHost, envDBPort, envDBName)

	if BenchmarkMode {
		runBenchmark()
	} else {
		runNormal()
	}
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func runBenchmark() {
	fmt.Println("--- BENCHMARK MODE ---")
	models := []string{"qwen3-vl:latest", "moondream:latest", "minicpm-v:latest"}
	
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, PostgresURL)
	if err != nil {
		log.Fatal(fmt.Errorf("DB connect error: %v (URL: %s)", err, PostgresURL))
	}
	defer conn.Close(ctx)

	// Get 5 images
	query := `
		SELECT a.id
		FROM asset a
		WHERE a.type = 'IMAGE'
		ORDER BY a."createdAt" DESC
		LIMIT 5
	`
	rows, err := conn.Query(ctx, query)
	if err != nil {
		log.Fatal(err)
	}
	
	var assetIDs []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			log.Fatal(err)
		}
		assetIDs = append(assetIDs, id)
	}
	rows.Close()
	client := &http.Client{Timeout: 0}

	for i, assetID := range assetIDs {
		fmt.Printf("\n[%d/5] Image ID: %s\n", i+1, assetID)
		
		imgBytes, err := downloadThumbnail(assetID)
		if err != nil {
			fmt.Printf("Error downloading: %v\n", err)
			continue
		}
		imgBytes, err = ensureJPEG(imgBytes)
		if err != nil {
			fmt.Printf("Error converting: %v\n", err)
			continue
		}
		b64Image := base64.StdEncoding.EncodeToString(imgBytes)

		for _, model := range models {
			fmt.Printf("  Testing %s ... ", model)
			start := time.Now()
			
			// Call generate with specific model
			desc, err := generateDescription(client, b64Image, model)
			duration := time.Since(start)

			if err != nil {
				fmt.Printf("FAILED (%v)\n", err)
			} else {
				fmt.Printf("DONE in %.2fs\n", duration.Seconds())
				fmt.Printf("    -> Description: %s\n", desc)
			}
		}
	}
	fmt.Println("\n--- BENCHMARK COMPLETE ---")
}

func runNormal() {
	fmt.Printf("Using model: %s\n", OllamaModel)
	ctx := context.Background()

	fmt.Println("1. Connecting to DB...")
	conn, err := pgx.Connect(ctx, PostgresURL)
	if err != nil {
		log.Fatal(fmt.Errorf("DB connect error: %v (URL: %s)", err, PostgresURL))
	}
	defer conn.Close(ctx)

	if CleanJunk == "dry-run" || CleanJunk == "true" {
		runCleanJunk(ctx, conn, CleanJunk == "true")
	}

	ollamaHTTPClient := &http.Client{Timeout: 0}
	totalProcessed := 0

	for {
		fmt.Println("2. Scanning for images (batch of 100)...")
		query := `
			SELECT a.id, a."originalPath"
			FROM asset a
			JOIN asset_exif ae ON a.id = ae."assetId"
			WHERE (ae.description IS NULL OR ae.description = '')
			AND a."originalPath" NOT LIKE '%/encoded-video/%'
				AND a."originalPath" NOT LIKE '%BURST%'
			ORDER BY COALESCE(ae."dateTimeOriginal", a."fileCreatedAt", a."createdAt") DESC
			LIMIT 100
		`
		rows, err := conn.Query(ctx, query)
		if err != nil {
			log.Fatal(err)
		}

		type assetRow struct {
			ID   string
			Path string
		}
		var assets []assetRow
		for rows.Next() {
			var a assetRow
			if err := rows.Scan(&a.ID, &a.Path); err != nil {
				log.Fatal(err)
			}
			assets = append(assets, a)
		}
		rows.Close()

		if len(assets) == 0 {
			if WatchMode {
				if totalProcessed > 0 {
					fmt.Printf("All caught up! Processed %d images.\n", totalProcessed)
					totalProcessed = 0
				}
				fmt.Printf("Sleeping for %v... (Ctrl+C to stop)\n", WatchInterval)
				time.Sleep(WatchInterval)
				continue
			}

			if totalProcessed == 0 {
				fmt.Println("No images found to process.")
			} else {
				fmt.Printf("All done! Processed %d images in total.\n", totalProcessed)
			}
			break
		}

		count := 0
		batchSuccess := 0
		for _, a := range assets {
			assetID := a.ID
			count++
			totalProcessed++
			shortPath := filepath.Base(filepath.Dir(a.Path)) + "/" + filepath.Base(a.Path)
			fmt.Printf("[%d|Total:%d] Processing %s %s ", count, totalProcessed, shortPath, assetID)

			imgBytes, err := downloadThumbnail(assetID)
			if err != nil {
				if strings.Contains(err.Error(), "status 404") {
					fileName, filePath := getAssetMetadata(assetID)
					fmt.Printf("\n   [SKIP] Thumbnail not ready | file=%s | path=%s\n", fileName, filePath)
				} else {
					fmt.Printf("\n   [SKIP] Download error: %v\n", err)
				}
				continue
			}

			imgBytes, err = ensureJPEG(imgBytes)
			if err != nil {
				fmt.Printf("\n   [SKIP] Image conversion error: %v\n", err)
				continue
			}

			b64Image := base64.StdEncoding.EncodeToString(imgBytes)

			fmt.Print("... Sending to GPU ... ")
			// Use global OllamaModel
			desc, err := generateDescription(ollamaHTTPClient, b64Image, OllamaModel)
			if err != nil {
				fmt.Printf("\n   [FAIL] Ollama error: %v\n", err)
				continue
			}

			_, err = conn.Exec(ctx, `UPDATE asset_exif SET description = $1 WHERE "assetId" = $2`, desc, assetID)
			if err != nil {
				fmt.Printf("\n   [ERR] DB Save error: %v\n", err)
				continue
			}
			if VerboseMode {
				fmt.Printf("Done! (%d chars)\nDescription: %s\n", len(desc), desc)
			} else {
				fmt.Printf("Done! (%d chars)\n", len(desc))
			}
			batchSuccess++
		}

		// If we found images but processed none (e.g. all 404), sleep to avoid hammering
		if len(assets) > 0 && batchSuccess == 0 {
			fmt.Println("Batch failed (waiting for thumbnails). Sleeping 2m...")
			time.Sleep(2 * time.Minute)
		}
	}
}

// runCleanJunk wipes asset_exif.description rows that look like imported JSON metadata
// (e.g. Pixel motion-photo blobs) so the watcher can re-describe them with Ollama.
// Match shape: starts with [ or { AND contains a distinctive JSON key from the known blobs.
// If execute is false, only counts and logs a sample (dry-run).
func runCleanJunk(ctx context.Context, conn *pgx.Conn, execute bool) {
	mode := "dry-run"
	if execute {
		mode = "true"
	}
	fmt.Printf("[CLEAN_JUNK=%s] Scanning for junk JSON descriptions...\n", mode)

	const matchClause = `
		(description LIKE '[%' OR description LIKE '{%')
		AND (
			description LIKE '%"videoResolution"%'
			OR description LIKE '%"videoDuration"%'
			OR description LIKE '%"isRecord"%'
			OR description LIKE '%"isCropped"%'
		)
	`

	previewQuery := `
		SELECT a."originalPath"
		FROM asset_exif ae
		JOIN asset a ON a.id = ae."assetId"
		WHERE ` + matchClause + `
		ORDER BY a."originalPath"
		LIMIT 10
	`
	rows, err := conn.Query(ctx, previewQuery)
	if err != nil {
		fmt.Printf("[CLEAN_JUNK=%s] Preview query error: %v\n", mode, err)
		return
	}
	var samples []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			fmt.Printf("[CLEAN_JUNK=%s] Scan error: %v\n", mode, err)
			rows.Close()
			return
		}
		samples = append(samples, p)
	}
	rows.Close()

	var total int
	countQuery := `SELECT COUNT(*) FROM asset_exif WHERE ` + matchClause
	if err := conn.QueryRow(ctx, countQuery).Scan(&total); err != nil {
		fmt.Printf("[CLEAN_JUNK=%s] Count query error: %v\n", mode, err)
		return
	}

	if total == 0 {
		fmt.Printf("[CLEAN_JUNK=%s] No matching junk descriptions found. Nothing to do.\n", mode)
		return
	}

	if !execute {
		fmt.Printf("[CLEAN_JUNK=dry-run] Would wipe %d junk JSON descriptions. Sample (up to 10):\n", total)
		for _, p := range samples {
			fmt.Printf("  %s\n", p)
		}
		fmt.Println("[CLEAN_JUNK=dry-run] Done — no changes made. Set CLEAN_JUNK=true to execute. Proceeding to watch mode.")
		return
	}

	tag, err := conn.Exec(ctx, `UPDATE asset_exif SET description = '' WHERE `+matchClause)
	if err != nil {
		fmt.Printf("[CLEAN_JUNK=true] UPDATE error: %v\n", err)
		return
	}
	fmt.Printf("[CLEAN_JUNK=true] Wiped %d junk JSON descriptions. Sample (up to 10):\n", tag.RowsAffected())
	for _, p := range samples {
		fmt.Printf("  %s\n", p)
	}
	fmt.Println("[CLEAN_JUNK=true] Done. Cleared rows will be re-described by the watcher. Proceeding to watch mode.")
}

var assetMetadataCache = make(map[string][2]string)

func getAssetMetadata(id string) (string, string) {
	if cached, ok := assetMetadataCache[id]; ok {
		return cached[0], cached[1]
	}

	u := fmt.Sprintf("%s/api/assets/%s", ImmichBaseURL, id)
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		assetMetadataCache[id] = [2]string{"UNKNOWN", "UNKNOWN"}
		return "UNKNOWN", "UNKNOWN"
	}
	req.Header.Set("x-api-key", ImmichAPIKey)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		assetMetadataCache[id] = [2]string{"UNKNOWN", "UNKNOWN"}
		return "UNKNOWN", "UNKNOWN"
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		assetMetadataCache[id] = [2]string{"UNKNOWN", "UNKNOWN"}
		return "UNKNOWN", "UNKNOWN"
	}

	var result struct {
		OriginalFileName string `json:"originalFileName"`
		OriginalPath     string `json:"originalPath"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		assetMetadataCache[id] = [2]string{"UNKNOWN", "UNKNOWN"}
		return "UNKNOWN", "UNKNOWN"
	}

	assetMetadataCache[id] = [2]string{result.OriginalFileName, result.OriginalPath}
	return result.OriginalFileName, result.OriginalPath
}

func downloadThumbnail(id string) ([]byte, error) {
	u := fmt.Sprintf("%s/api/assets/%s/thumbnail?format=JPEG", ImmichBaseURL, id)
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", ImmichAPIKey)
	req.Header.Set("Accept", "application/octet-stream")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

func generateDescription(client *http.Client, base64Image string, modelName string) (string, error) {
	payload := ChatRequest{
		Model:  modelName,
		Stream: false,
		Messages: []Message{
			{
				Role:    "user",
				Content: "Describe this image concisely. Then list 15 relevant keywords for search (objects, activities, setting, time, colors).",
				Images:  []string{base64Image},
			},
		},
		Options: map[string]interface{}{
			"num_predict": 500,
			"temperature": 0.1,
		},
	}

	jsonData, _ := json.Marshal(payload)

	resp, err := client.Post(OllamaHost+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama status %d: %s", resp.StatusCode, string(body))
	}

	var response ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", err
	}

	return response.Message.Content, nil
}

func ensureJPEG(data []byte) ([]byte, error) {
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}
	if format != "jpeg" {
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, img, nil); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	}
	return data, nil
}