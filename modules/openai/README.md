# OpenAI API Module for NanoLang

Comprehensive OpenAI-compatible API client enabling NanoLang programs to interact with LLMs for chat completions, embeddings, and more. Works with OpenAI API and any OpenAI-compatible endpoints (local LLMs, Ollama, LM Studio, etc.).

## Features

- ✅ Chat completions (main use case)
- ✅ Configurable endpoints (OpenAI or local LLMs)
- ✅ Temperature and token limit control
- ✅ Embeddings generation
- ✅ Model listing and information
- ✅ Token counting estimation
- ✅ Clean, type-safe API

## Installation

The module uses libcurl for HTTP requests. Install dependencies:

**macOS:**
```bash
brew install curl
```

**Linux:**
```bash
sudo apt install libcurl4-openssl-dev  # Debian/Ubuntu
sudo yum install libcurl-devel         # RHEL/CentOS
```

## Authentication

### For OpenAI API

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='sk-your_key_here'
```

Get your key at: https://platform.openai.com/api-keys

### For Local LLMs (Ollama, LM Studio, etc.)

```bash
export OPENAI_API_KEY='dummy'  # Any value works for local
```

Then configure the endpoint in your code (see examples below).

## Quick Start

### OpenAI API

```nano
import "modules/openai/openai.nano"

fn main() -> int {
    let api_key: string = (openai_get_key_from_env)

    # Simple chat completion
    let response: string = (openai_chat_completion_simple
        "gpt-4"
        "You are a helpful assistant."
        "What is the capital of France?"
        api_key)

    (println response)
    return 0
}

shadow main { assert true }
```

### Local LLM (Ollama, LM Studio, etc.)

```nano
import "modules/openai/openai.nano"

fn main() -> int {
    # Configure for local LLM endpoint
    (openai_set_api_base "http://localhost:8080/v1")

    let api_key: string = "dummy"  # Not needed for local

    let response: string = (openai_chat_completion_simple
        "llama2"  # Or whatever model your local LLM uses
        "You are a helpful assistant."
        "Explain quantum computing in simple terms."
        api_key)

    (println response)
    return 0
}

shadow main { assert true }
```

## API Reference

### Configuration

#### `openai_set_api_base(base_url: string) -> void`

Set custom API endpoint. Default is `https://api.openai.com/v1`.

```nano
# For Ollama
(openai_set_api_base "http://localhost:11434/v1")

# For LM Studio
(openai_set_api_base "http://localhost:1234/v1")

# For custom deployment
(openai_set_api_base "https://your-server.com/v1")
```

#### `openai_get_api_base() -> string`

Get current API endpoint.

```nano
let current_endpoint: string = (openai_get_api_base)
(println current_endpoint)
```

#### `openai_get_key_from_env() -> string`

Get API key from `OPENAI_API_KEY` environment variable.

```nano
let api_key: string = (openai_get_key_from_env)
```

### Chat Completions

#### `openai_chat_completion_simple(model: string, system_prompt: string, user_message: string, api_key: string) -> string`

Simplest way to get a completion. Returns JSON response.

```nano
let response: string = (openai_chat_completion_simple
    "gpt-4"
    "You are a helpful coding assistant."
    "Write a function to sort an array."
    api_key)
```

#### `openai_chat_completion(model: string, messages_json: string, api_key: string) -> string`

Full control with custom message array. Useful for multi-turn conversations.

```nano
let messages: string = "[{\"role\":\"system\",\"content\":\"You are helpful.\"},{\"role\":\"user\",\"content\":\"Hello!\"},{\"role\":\"assistant\",\"content\":\"Hi there!\"},{\"role\":\"user\",\"content\":\"How are you?\"}]"

let response: string = (openai_chat_completion "gpt-4" messages api_key)
```

#### `openai_chat_completion_with_temperature(model: string, messages_json: string, temperature: float, max_tokens: int, api_key: string) -> string`

Advanced completion with temperature and token limits.

```nano
let messages: string = "[{\"role\":\"user\",\"content\":\"Be creative!\"}]"

let response: string = (openai_chat_completion_with_temperature
    "gpt-4"
    messages
    0.9      # Higher = more creative (0.0 - 2.0)
    500      # Max tokens in response
    api_key)
```

**Temperature guide:**
- `0.0 - 0.3`: Focused, deterministic, factual
- `0.4 - 0.7`: Balanced (default is usually 0.7)
- `0.8 - 1.5`: Creative, varied
- `1.6 - 2.0`: Very random (experimental)

### Embeddings

#### `openai_create_embedding(model: string, input: string, api_key: string) -> string`

Generate embeddings for semantic search, clustering, etc.

```nano
let embedding_json: string = (openai_create_embedding
    "text-embedding-ada-002"
    "The quick brown fox jumps over the lazy dog"
    api_key)

# Response contains vector array in JSON
```

### Model Management

#### `openai_list_models(api_key: string) -> string`

List available models.

```nano
let models: string = (openai_list_models api_key)
(println models)
```

#### `openai_get_model(model_id: string, api_key: string) -> string`

Get information about a specific model.

```nano
let model_info: string = (openai_get_model "gpt-4" api_key)
(println model_info)
```

### Utility Functions

#### `openai_count_tokens_estimate(text: string) -> int`

Rough token count estimation (1 token ≈ 4 characters). Useful for staying within limits.

```nano
let text: string = "This is a sample message for token counting."
let tokens: int = (openai_count_tokens_estimate text)
(println (str_concat "Estimated tokens: " (int_to_string tokens)))
```

## Response Format

All functions return JSON strings. Use the `json` module to parse responses:

```nano
from "modules/std/json/json.nano" import Json, parse, get_string, get_nested_string

let response: string = (openai_chat_completion_simple
    "gpt-4"
    "You are helpful."
    "Say hello!"
    api_key)

let root: Json = (parse response)
let content: string = (get_nested_string root "choices" 0 "message" "content")

(println content)  # Prints the actual response text
```

## Error Handling

Errors are returned as JSON with an `"error"` field:

```nano
let result: string = (openai_chat_completion_simple "invalid-model" "" "test" api_key)
if (str_contains result "\"error\"") {
    (println "API call failed")
    (println result)
}
```

## Rate Limiting & Costs

### OpenAI API

- **Rate limits:** Vary by tier (check your account)
- **Costs:** Per token (input + output)
  - GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens
  - GPT-3.5-turbo: ~$0.0015/1K input tokens, ~$0.002/1K output tokens

Monitor your usage at: https://platform.openai.com/usage

### Local LLMs

- **No rate limits** - depends on your hardware
- **No costs** - runs locally
- **Speed:** Depends on GPU/CPU

## Popular Local LLM Setups

### Ollama

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama2

# Run (starts on port 11434)
ollama serve
```

```nano
# In your NanoLang code
(openai_set_api_base "http://localhost:11434/v1")
let response: string = (openai_chat_completion_simple "llama2" "" "Hello!" "dummy")
```

### LM Studio

1. Download from https://lmstudio.ai/
2. Load a model (e.g., Mistral, Llama 2)
3. Start local server (default port 1234)

```nano
(openai_set_api_base "http://localhost:1234/v1")
let response: string = (openai_chat_completion_simple "local-model" "" "Hello!" "dummy")
```

## Best Practices

### 1. System Prompts

Always include a clear system prompt:

```nano
let system: string = "You are an expert programmer specializing in systems programming. Be concise and provide working code examples."
```

### 2. Token Management

```nano
let user_input: string = get_user_input()
let tokens: int = (openai_count_tokens_estimate user_input)

if (> tokens 4000) {
    (println "Input too long, please shorten")
} else {
    let response: string = (openai_chat_completion_simple "gpt-4" "" user_input api_key)
}
```

### 3. Error Handling

```nano
let response: string = (openai_chat_completion_simple "gpt-4" "" query api_key)

if (str_contains response "\"error\"") {
    (println "API Error - falling back to cached response")
    return cached_response
}
```

### 4. Multi-turn Conversations

Build message history:

```nano
# Start with system message
let messages: string = "[{\"role\":\"system\",\"content\":\"You are helpful.\"}]"

# Add user message
set messages (str_concat messages ",{\"role\":\"user\",\"content\":\"Hello!\"}")

# Get response, parse it
let response: string = (openai_chat_completion "gpt-4" messages api_key)

# Add assistant response to history
set messages (str_concat messages ",{\"role\":\"assistant\",\"content\":\"Hi there!\"}")

# Continue conversation...
```

## Examples

### Example 1: Code Generation

```nano
fn generate_code(description: string, api_key: string) -> string {
    let system: string = "You are an expert programmer. Generate clean, well-documented code."

    return (openai_chat_completion_simple
        "gpt-4"
        system
        (str_concat "Generate code for: " description)
        api_key)
}
```

### Example 2: Issue Analysis

```nano
fn analyze_bug_report(issue_text: string, api_key: string) -> string {
    let system: string = "You are a software engineer. Analyze bug reports and suggest fixes."

    return (openai_chat_completion_simple
        "gpt-4"
        system
        (str_concat "Analyze this bug:\n\n" issue_text)
        api_key)
}
```

### Example 3: Autonomous Agent

See `examples/ai_github_agent.nano` for a complete example of an autonomous agent that:
- Fetches GitHub issues
- Analyzes them with an LLM
- Generates responses
- Can automatically comment or create PRs

## API Documentation

- **OpenAI:** https://platform.openai.com/docs/api-reference
- **Ollama:** https://github.com/ollama/ollama/blob/main/docs/api.md
- **LM Studio:** https://lmstudio.ai/docs

## License

Part of the NanoLang project. See LICENSE file for details.
