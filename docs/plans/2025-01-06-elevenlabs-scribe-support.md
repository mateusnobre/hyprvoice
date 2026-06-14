# Eleven Labs Scribe Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Eleven Labs Scribe API as a transcription provider option in Hyprvoice, supporting both scribe_v1 (99 languages, best accuracy) and scribe_v2 (90 languages, real-time optimized) models.

**Architecture:** Create a new `ElevenLabsAdapter` implementing the `TranscriptionAdapter` interface, following the existing pattern used by OpenAI, Groq, and Mistral adapters. The adapter will make HTTP POST requests with multipart/form-data to Eleven Labs' speech-to-text endpoint. Configuration via environment variable `ELEVENLABS_API_KEY` with interactive CLI setup.

**Tech Stack:** Go 1.21+, HTTP client with multipart support, existing transcriber interface pattern

---

## Task 1: Create Eleven Labs Adapter

**Files:**
- Create: `internal/transcriber/adapter_elevenlabs.go`
- Test: `internal/transcriber/adapter_elevenlabs_test.go`

### Step 1: Write the failing test

Create: `internal/transcriber/adapter_elevenlabs_test.go`

```go
package transcriber

import (
	"context"
	"testing"
)

func TestNewElevenLabsAdapter(t *testing.T) {
	config := Config{
		Provider: "elevenlabs",
		APIKey:   "test-api-key",
		Language: "en",
		Model:    "scribe_v1",
	}

	adapter := NewElevenLabsAdapter(config)

	if adapter == nil {
		t.Fatalf("NewElevenLabsAdapter() returned nil")
	}

	if adapter.config.APIKey != "test-api-key" {
		t.Errorf("APIKey not set correctly, got: %s", adapter.config.APIKey)
	}

	if adapter.config.Model != "scribe_v1" {
		t.Errorf("Model not set correctly, got: %s", adapter.config.Model)
	}
}

func TestElevenLabsAdapter_Transcribe_EmptyAudio(t *testing.T) {
	config := Config{
		Provider: "elevenlabs",
		APIKey:   "test-key",
		Model:    "scribe_v1",
	}

	adapter := NewElevenLabsAdapter(config)
	ctx := context.Background()

	result, err := adapter.Transcribe(ctx, []byte{})

	if err != nil {
		t.Errorf("Transcribe() with empty audio should not error, got: %v", err)
	}

	if result != "" {
		t.Errorf("Transcribe() with empty audio should return empty string, got: %s", result)
	}
}

func TestElevenLabsAdapter_Transcribe_ValidAudio(t *testing.T) {
	// This test will require mocking the HTTP client
	// For now, we test the structure exists
	config := Config{
		Provider: "elevenlabs",
		APIKey:   "test-key",
		Language: "en",
		Model:    "scribe_v1",
	}

	adapter := NewElevenLabsAdapter(config)

	if adapter == nil {
		t.Fatal("NewElevenLabsAdapter() returned nil")
	}

	// Test that adapter has a client
	if adapter.client == nil {
		t.Error("adapter.client is nil")
	}
}
```

Run: `go test ./internal/transcriber/adapter_elevenlabs_test.go -v`

Expected: FAIL with "undefined: NewElevenLabsAdapter"

### Step 2: Write minimal adapter implementation

Create: `internal/transcriber/adapter_elevenlabs.go`

```go
package transcriber

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"time"
)

// ElevenLabsAdapter implements TranscriptionAdapter for ElevenLabs Scribe API
type ElevenLabsAdapter struct {
	client *http.Client
	config Config
}

// ElevenLabsResponse represents the API response
type ElevenLabsResponse struct {
	Text string `json:"text"`
}

// NewElevenLabsAdapter creates a new ElevenLabs adapter
func NewElevenLabsAdapter(config Config) *ElevenLabsAdapter {
	return &ElevenLabsAdapter{
		client: &http.Client{Timeout: 30 * time.Second},
		config: config,
	}
}

// Transcribe sends audio to ElevenLabs API for transcription
func (a *ElevenLabsAdapter) Transcribe(ctx context.Context, audioData []byte) (string, error) {
	if len(audioData) == 0 {
		return "", nil
	}

	// Convert raw PCM to WAV format
	wavData, err := convertToWAV(audioData)
	if err != nil {
		return "", fmt.Errorf("convert to WAV: %w", err)
	}

	// Create multipart form body
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	// Add audio file
	part, err := writer.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}
	if _, err := io.Copy(part, bytes.NewReader(wavData)); err != nil {
		return "", fmt.Errorf("copy audio data: %w", err)
	}

	// Add model_id
	if err := writer.WriteField("model_id", a.config.Model); err != nil {
		return "", fmt.Errorf("write model_id: %w", err)
	}

	// Add language_code if specified
	if a.config.Language != "" {
		if err := writer.WriteField("language_code", a.config.Language); err != nil {
			return "", fmt.Errorf("write language_code: %w", err)
		}
	}

	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("close writer: %w", err)
	}

	// Create HTTP request
	url := "https://api.elevenlabs.io/v1/speech-to-text"
	req, err := http.NewRequestWithContext(ctx, "POST", url, &body)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("xi-api-key", a.config.APIKey)

	start := time.Now()
	resp, err := a.client.Do(req)
	duration := time.Since(start)

	if err != nil {
		log.Printf("elevenlabs-adapter: API call failed after %v: %v", duration, err)
		return "", fmt.Errorf("elevenlabs request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("elevenlabs-adapter: API returned status %d: %s", resp.StatusCode, string(bodyBytes))
		return "", fmt.Errorf("elevenlabs API status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result ElevenLabsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	log.Printf("elevenlabs-adapter: transcribed %d bytes in %v: %q", len(audioData), duration, result.Text)
	return result.Text, nil
}
```

### Step 3: Run tests to verify they pass

Run: `go test ./internal/transcriber/ -v -run TestNewElevenLabsAdapter`

Expected: PASS

Run: `go test ./internal/transcriber/ -v -run TestElevenLabsAdapter`

Expected: PASS

### Step 4: Commit

```bash
git add internal/transcriber/adapter_elevenlabs.go internal/transcriber/adapter_elevenlabs_test.go
git commit -m "feat: add Eleven Labs Scribe adapter"
```

---

## Task 2: Register Eleven Labs Provider

**Files:**
- Modify: `internal/transcriber/transcriber.go:35-62`
- Modify: `internal/transcriber/transcriber_test.go:98-115`

### Step 1: Add test cases for Eleven Labs provider

Modify: `internal/transcriber/transcriber_test.go`

Add after line 97 (after mistral-transcription config without api key test):

```go
		{
			name: "valid elevenlabs config with scribe_v1",
			config: Config{
				Provider: "elevenlabs",
				APIKey:   "test-key",
				Language: "en",
				Model:    "scribe_v1",
			},
			wantErr: false,
		},
		{
			name: "valid elevenlabs config with scribe_v2",
			config: Config{
				Provider: "elevenlabs",
				APIKey:   "test-key",
				Language: "pt",
				Model:    "scribe_v2",
			},
			wantErr: false,
		},
		{
			name: "elevenlabs config without api key",
			config: Config{
				Provider: "elevenlabs",
				APIKey:   "",
				Language: "en",
				Model:    "scribe_v1",
			},
			wantErr: true,
		},
```

Run: `go test ./internal/transcriber/ -v -run TestNewTranscriber`

Expected: FAIL - test case for elevenlabs fails with "unsupported provider"

### Step 2: Add Eleven Labs case to NewTranscriber

Modify: `internal/transcriber/transcriber.go`

Add after line 58 (after mistral case):

```go
	case "elevenlabs":
		if config.APIKey == "" {
			return nil, fmt.Errorf("ElevenLabs API key required")
		}
		adapter = NewElevenLabsAdapter(config)
```

Also update the error message on line 61 to include elevenlabs:

Change:
```go
		return nil, fmt.Errorf("unsupported provider: %s", config.Provider)
```

### Step 3: Run tests to verify they pass

Run: `go test ./internal/transcriber/ -v -run TestNewTranscriber`

Expected: PASS (all tests including new elevenlabs tests)

### Step 4: Commit

```bash
git add internal/transcriber/transcriber.go internal/transcriber/transcriber_test.go
git commit -m "feat: register Eleven Labs as transcription provider"
```

---

## Task 3: Add Config Validation for Eleven Labs

**Files:**
- Modify: `internal/config/config.go:122-134`
- Modify: `internal/config/config.go:172-248`

### Step 1: Add ELEVENLABS_API_KEY to ToTranscriberConfig

Modify: `internal/config/config.go`

In `ToTranscriberConfig()`, add elevenlabs case after line 129 (after mistral case):

```go
	case "elevenlabs":
		config.APIKey = os.Getenv("ELEVENLABS_API_KEY")
```

### Step 2: Add Eleven Labs validation in Validate()

Modify: `internal/config/config.go`

Add after line 244 (after mistral-transcription validation):

```go
	case "elevenlabs":
		apiKey := c.Transcription.APIKey
		if apiKey == "" {
			apiKey = os.Getenv("ELEVENLABS_API_KEY")
		}
		if apiKey == "" {
			return fmt.Errorf("ElevenLabs API key required: not found in config (transcription.api_key) or environment variable (ELEVENLABS_API_KEY)")
		}

		// Validate language code if provided (empty string means auto-detect)
		if c.Transcription.Language != "" && !isValidLanguageCode(c.Transcription.Language) {
			return fmt.Errorf("invalid transcription.language: %s (use empty string for auto-detect or ISO-639-1 codes like 'en', 'pt', 'es')", c.Transcription.Language)
		}

		// Validate Eleven Labs model
		validModels := map[string]bool{"scribe_v1": true, "scribe_v2": true}
		if c.Transcription.Model != "" && !validModels[c.Transcription.Model] {
			return fmt.Errorf("invalid model for elevenlabs: %s (must be scribe_v1 or scribe_v2)", c.Transcription.Model)
		}
```

Update the error message on line 247 to include elevenlabs:

Change from:
```go
		return fmt.Errorf("unsupported transcription.provider: %s (must be openai, groq-transcription, groq-translation, or mistral-transcription)", c.Transcription.Provider)
```

To:
```go
		return fmt.Errorf("unsupported transcription.provider: %s (must be openai, groq-transcription, groq-translation, mistral-transcription, or elevenlabs)", c.Transcription.Provider)
```

### Step 3: Test config validation

Run: `go test ./internal/config/ -v -run TestConfig`

Expected: PASS

### Step 4: Commit

```bash
git add internal/config/config.go
git commit -m "feat: add Eleven Labs config validation"
```

---

## Task 4: Update CLI Provider Selection

**Files:**
- Modify: `cmd/hyprvoice/main.go:163-194`
- Modify: `cmd/hyprvoice/main.go:196-278`

### Step 1: Add Eleven Labs to provider options

Modify: `cmd/hyprvoice/main.go`

Update the provider selection prompt (around line 164):

Change:
```go
		fmt.Println("Select transcription provider:")
		fmt.Println("  1. openai                 - OpenAI Whisper API (cloud-based)")
		fmt.Println("  2. groq-transcription     - Groq Whisper API (fast transcription)")
		fmt.Println("  3. groq-translation       - Groq Whisper API (translate to English)")
		fmt.Println("  4. mistral-transcription  - Mistral Voxtral API (European languages)")
		fmt.Printf("Provider [1-4] (current: %s): ", cfg.Transcription.Provider)
```

To:
```go
		fmt.Println("Select transcription provider:")
		fmt.Println("  1. openai                 - OpenAI Whisper API (cloud-based)")
		fmt.Println("  2. groq-transcription     - Groq Whisper API (fast transcription)")
		fmt.Println("  3. groq-translation       - Groq Whisper API (translate to English)")
		fmt.Println("  4. mistral-transcription  - Mistral Voxtral API (European languages)")
		fmt.Println("  5. elevenlabs             - ElevenLabs Scribe API (99 languages, excellent accuracy)")
		fmt.Printf("Provider [1-5] (current: %s): ", cfg.Transcription.Provider)
```

Update the switch statement to add case "5" and "elevenlabs":

After line 185 (after case "4"), add:
```go
		case "5":
			cfg.Transcription.Provider = "elevenlabs"
		case "elevenlabs":
			cfg.Transcription.Provider = "elevenlabs"
```

Update error message on line 189:

Change from:
```go
			fmt.Println("❌ Error: invalid provider. Please enter 1-4 or provider name.")
```

To:
```go
			fmt.Println("❌ Error: invalid provider. Please enter 1-5 or provider name.")
```

### Step 2: Add Eleven Labs model selection

Modify: `cmd/hyprvoice/main.go`

Add after line 278 (after mistral-transcription case, before the API key section):

```go
	case "elevenlabs":
		for {
			fmt.Println("\nElevenLabs Scribe Model:")
			fmt.Println("  Language Support:")
			fmt.Println("    scribe_v1: 99 languages (96.7% accuracy for English, ≤5% WER for Portuguese)")
			fmt.Println("    scribe_v2: 90 languages (real-time optimized, lower latency)")
			fmt.Println()
			fmt.Println("  Available Models:")
			fmt.Println("    1. scribe_v1  - Best accuracy, full timestamps (recommended)")
			fmt.Println("    2. scribe_v2  - Real-time streaming, lower latency")
			fmt.Printf("Model [1-2] (current: %s): ", cfg.Transcription.Model)
			if !scanner.Scan() {
				break
			}
			input := strings.TrimSpace(scanner.Text())
			switch input {
			case "1":
				cfg.Transcription.Model = "scribe_v1"
			case "2":
				cfg.Transcription.Model = "scribe_v2"
			case "scribe_v1", "scribe_v2":
				cfg.Transcription.Model = input
			case "":
				if cfg.Transcription.Model == "" {
					cfg.Transcription.Model = "scribe_v1"
				}
			default:
				fmt.Println("❌ Error: invalid model. Please enter 1, 2 or model name.")
				continue
			}
			break
		}
```

### Step 3: Add Eleven Labs to envVarName switch

Modify: `cmd/hyprvoice/main.go`

Update the envVarName switch (around line 281-289):

Change from:
```go
	var envVarName string
	switch cfg.Transcription.Provider {
	case "openai":
		envVarName = "OPENAI_API_KEY"
	case "mistral-transcription":
		envVarName = "MISTRAL_API_KEY"
	default:
		envVarName = "GROQ_API_KEY"
	}
```

To:
```go
	var envVarName string
	switch cfg.Transcription.Provider {
	case "openai":
		envVarName = "OPENAI_API_KEY"
	case "mistral-transcription":
		envVarName = "MISTRAL_API_KEY"
	case "elevenlabs":
		envVarName = "ELEVENLABS_API_KEY"
	default:
		envVarName = "GROQ_API_KEY"
	}
```

### Step 4: Add language performance info for Eleven Labs

Modify: `cmd/hyprvoice/main.go`

Update the language prompt section (around line 298-308):

Change from:
```go
	// Language
	if cfg.Transcription.Provider == "groq-translation" {
		fmt.Printf("\nSource language hint (empty for auto-detect, current: %s): ", cfg.Transcription.Language)
		fmt.Println("\n  Note: Translation always outputs English. Language hints at source audio language.")
	} else {
		fmt.Printf("\nLanguage (empty for auto-detect, current: %s): ", cfg.Transcription.Language)
	}
```

To:
```go
	// Language
	if cfg.Transcription.Provider == "groq-translation" {
		fmt.Printf("\nSource language hint (empty for auto-detect, current: %s): ", cfg.Transcription.Language)
		fmt.Println("\n  Note: Translation always outputs English. Language hints at source audio language.")
	} else if cfg.Transcription.Provider == "elevenlabs" {
		fmt.Println("\nLanguage Performance:")
		fmt.Println("  Excellent (≤5% WER): English, Portuguese, +25 languages")
		fmt.Println("  High (5-10% WER): French, German, Spanish, Italian, etc.")
		fmt.Println("  Good (10-20% WER): Most supported languages")
		fmt.Println("  Leave empty for auto-detection (recommended)")
		fmt.Printf("Language (current: %s): ", cfg.Transcription.Language)
	} else {
		fmt.Printf("\nLanguage (empty for auto-detect, current: %s): ", cfg.Transcription.Language)
	}
```

### Step 5: Test CLI configure command

Run: `go run ./cmd/hyprvoice configure`

Expected to show Eleven Labs as option 5

### Step 6: Commit

```bash
git add cmd/hyprvoice/main.go
git commit -m "feat: add Eleven Labs to CLI configuration wizard"
```

---

## Task 5: Update Config File Template

**Files:**
- Modify: `cmd/hyprvoice/main.go:595-666`
- Modify: `internal/config/config.go:387-493`

### Step 1: Update provider comment in CLI saveConfig

Modify: `cmd/hyprvoice/main.go`

Update line 611 provider comment:

Change from:
```go
  provider = "%s"          # Transcription service: "openai", "groq-transcription", "groq-translation", or "mistral-transcription"
```

To:
```go
  provider = "%s"          # Transcription service: "openai", "groq-transcription", "groq-translation", "mistral-transcription", or "elevenlabs"
```

Update line 612 api_key comment:

Change from:
```go
  api_key = "%s"                 # API key (or set OPENAI_API_KEY/GROQ_API_KEY/MISTRAL_API_KEY environment variable)
```

To:
```go
  api_key = "%s"                 # API key (or set OPENAI_API_KEY/GROQ_API_KEY/MISTRAL_API_KEY/ELEVENLABS_API_KEY environment variable)
```

Update line 614 model comment:

Change from:
```go
  model = "%s"          # Model: OpenAI="whisper-1", Groq="whisper-large-v3", Mistral="voxtral-mini-latest"
```

To:
```go
  model = "%s"          # Model: OpenAI="whisper-1", Groq="whisper-large-v3", Mistral="voxtral-mini-latest", ElevenLabs="scribe_v1" or "scribe_v2"
```

Update provider explanations (after line 636):

Add after mistral-transcription explanation:
```go
# - "elevenlabs": ElevenLabs Scribe API (excellent accuracy, 99 languages, requires ELEVENLABS_API_KEY)
#     Models: scribe_v1 (99 languages, best accuracy) or scribe_v2 (90 languages, real-time)
```

### Step 2: Update SaveDefaultConfig in config.go

Modify: `internal/config/config.go`

Update line 415 provider comment:

Change from:
```go
  provider = "openai"          # Transcription service: "openai", "groq-transcription", "groq-translation", or "mistral-transcription"
```

To:
```go
  provider = "openai"          # Transcription service: "openai", "groq-transcription", "groq-translation", "mistral-transcription", or "elevenlabs"
```

Update line 416 api_key comment:

Change from:
```go
  api_key = ""                 # API key (or set OPENAI_API_KEY/GROQ_API_KEY/MISTRAL_API_KEY environment variable)
```

To:
```go
  api_key = ""                 # API key (or set OPENAI_API_KEY/GROQ_API_KEY/MISTRAL_API_KEY/ELEVENLABS_API_KEY environment variable)
```

Update line 418 model comment:

Change from:
```go
  model = "whisper-1"          # Model: OpenAI="whisper-1", Groq="whisper-large-v3", Mistral="voxtral-mini-latest"
```

To:
```go
  model = "whisper-1"          # Model: OpenAI="whisper-1", Groq="whisper-large-v3", Mistral="voxtral-mini-latest", ElevenLabs="scribe_v1"
```

Add elevenlabs explanation after line 481:

```go
# - "elevenlabs": ElevenLabs Scribe API (excellent accuracy, 99 languages, requires ELEVENLABS_API_KEY)
#     Models: scribe_v1 (99 languages, best accuracy) or scribe_v2 (90 languages, real-time)
```

### Step 3: Test config generation

Run: `rm ~/.config/hyprvoice/config.toml && go run ./cmd/hyprvoice configure`

Select option 5 (elevenlabs), verify config is generated correctly

Check: `cat ~/.config/hyprvoice/config.toml`

Expected: Config file contains provider="elevenlabs" with correct comments

### Step 4: Commit

```bash
git add cmd/hyprvoice/main.go internal/config/config.go
git commit -m "feat: update config template with Eleven Labs documentation"
```

---

## Task 6: Update README Documentation

**Files:**
- Modify: `README.md`

### Step 1: Add Eleven Labs to features section

Find the "Multiple transcription backends" section in README and add Eleven Labs

Add to list of backends:
```markdown
- **Eleven Labs Scribe**: State-of-the-art accuracy with 99 languages, excellent performance for English and Portuguese
```

### Step 2: Update installation instructions if needed

Add environment variable setup instruction:
```bash
export ELEVENLABS_API_KEY="your-api-key-here"
```

### Step 3: Commit

```bash
git add README.md
git commit -m "docs: add Eleven Labs to README"
```

---

## Task 7: End-to-End Testing

### Step 1: Build the project

Run: `go build ./cmd/hyprvoice`

Expected: Binary created successfully

### Step 2: Test configuration with Eleven Labs

Run: `./hyprvoice configure`

Select option 5 (elevenlabs), select model 1 (scribe_v1), set language to empty for auto-detect

Expected: Configuration saved successfully

### Step 3: Verify config file

Run: `cat ~/.config/hyprvoice/config.toml`

Expected:
```toml
[transcription]
  provider = "elevenlabs"
  api_key = ""
  language = ""
  model = "scribe_v1"
```

### Step 4: Test daemon startup with mock API

Note: This requires a valid ELEVENLABS_API_KEY to fully test

For manual testing with real API:
```bash
export ELEVENLABS_API_KEY="your-key"
./hyprvoice serve
```

In another terminal:
```bash
./hyprvoice toggle
# Speak something
./hyprvoice toggle
```

Expected: Audio is transcribed via Eleven Labs API

### Step 5: Run all tests

Run: `go test ./...`

Expected: All tests pass

### Step 6: Final commit if any adjustments needed

```bash
git add .
git commit -m "test: ensure all tests pass for Eleven Labs integration"
```

---

## Summary

This plan adds complete Eleven Labs Scribe API support to Hyprvoice:

1. **Adapter Implementation** (`adapter_elevenlabs.go`): HTTP client with multipart/form-data requests to Eleven Labs API
2. **Provider Registration**: Integrated into transcriber factory pattern
3. **Config Validation**: API key validation via `ELEVENLABS_API_KEY`, model validation (scribe_v1, scribe_v2)
4. **CLI Integration**: Interactive setup with provider option 5, model selection, language performance info
5. **Documentation**: Updated config templates and README

**Models Supported:**
- `scribe_v1`: 99 languages, best accuracy (≤5% WER for English/Portuguese)
- `scribe_v2`: 90 languages, real-time optimized

**Language Performance:**
- Excellent (≤5% WER): English, Portuguese, +25 languages
- High (5-10% WER): Major European and Asian languages
- Good (10-20% WER): Most supported languages

**Configuration:**
- Environment variable: `ELEVENLABS_API_KEY`
- Config file: `provider = "elevenlabs"`, `model = "scribe_v1"` (or `scribe_v2`)
