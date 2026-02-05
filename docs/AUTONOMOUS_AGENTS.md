# Autonomous Agents in NanoLang

**Status:** Production Ready
**Version:** 1.0.0
**Added:** February 2026

---

## Overview

NanoLang now supports building **autonomous agents** that can interact with GitHub repositories and reason using Large Language Models (LLMs). This enables self-improving systems that can analyze issues, generate fixes, and interact with their own codebase.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NanoLang Program                          │
│                  (Autonomous Agent Logic)                    │
└─────────────────┬──────────────────┬────────────────────────┘
                  │                  │
         ┌────────▼────────┐  ┌─────▼──────────┐
         │ GitHub Module   │  │ OpenAI Module  │
         │ (API Client)    │  │ (LLM Client)   │
         └────────┬────────┘  └─────┬──────────┘
                  │                  │
         ┌────────▼────────┐  ┌─────▼──────────┐
         │  GitHub API     │  │  OpenAI API    │
         │  (REST v3)      │  │  (or Local LLM)│
         └─────────────────┘  └────────────────┘
```

## Core Modules

### 1. GitHub Module (`modules/github`)

Comprehensive GitHub REST API client for:
- Repository management
- Issue tracking and triage
- Pull request creation and review
- Comments and collaboration

**Example:**
```nano
import "modules/github/github.nano"

fn get_open_issues(owner: string, repo: string) -> string {
    let token: string = (github_get_token_from_env)
    return (github_list_issues owner repo "open" token)
}
```

See: [`modules/github/README.md`](../modules/github/README.md)

### 2. OpenAI Module (`modules/openai`)

OpenAI-compatible API client supporting:
- GPT models (OpenAI)
- Local LLMs (Ollama, LM Studio, etc.)
- Chat completions and reasoning
- Embeddings for semantic search

**Example:**
```nano
import "modules/openai/openai.nano"

fn analyze_code(code: string) -> string {
    let api_key: string = (openai_get_key_from_env)
    return (openai_chat_completion_simple
        "gpt-4"
        "You are a code reviewer."
        (str_concat "Review this code:\n\n" code)
        api_key)
}
```

See: [`modules/openai/README.md`](../modules/openai/README.md)

## Autonomous Agent Example

Full working example: [`examples/ai_github_agent.nano`](../examples/ai_github_agent.nano)

This example demonstrates:
1. ✅ Fetching GitHub issues automatically
2. ✅ Analyzing them with an LLM
3. ✅ Generating suggested responses
4. ✅ Error handling and rate limiting

**Run it:**
```bash
export GITHUB_TOKEN='your_token'
export OPENAI_API_KEY='your_key'
nanoc examples/ai_github_agent.nano -o ai_agent
./ai_agent
```

## Use Cases

### 1. Automated Issue Triage

```nano
fn triage_new_issues(owner: string, repo: string) -> int {
    let github_token: string = (github_get_token_from_env)
    let openai_key: string = (openai_get_key_from_env)

    # Get recent issues
    let issues_json: string = (github_list_issues owner repo "open" github_token)

    # Parse and analyze each issue
    # Use LLM to classify: bug, feature, question
    # Add appropriate labels
    # Comment with triage results

    return 0
}
```

### 2. Automated Bug Analysis

```nano
fn analyze_bug_report(issue_number: int) -> string {
    # Fetch issue from GitHub
    let issue: string = (github_get_issue "owner" "repo" issue_number token)

    # Extract stack trace and description
    # Analyze with LLM
    let analysis: string = (openai_chat_completion_simple
        "gpt-4"
        "You are a debugging expert. Analyze this bug report."
        issue
        api_key)

    # Post analysis as comment
    (github_create_issue_comment "owner" "repo" issue_number analysis token)

    return analysis
}
```

### 3. Automated Code Review

```nano
fn review_pull_request(pr_number: int) -> string {
    # Fetch PR diff from GitHub
    let pr_json: string = (github_get_pr "owner" "repo" pr_number token)

    # Extract code changes
    # Send to LLM for review
    let review: string = (openai_chat_completion_simple
        "gpt-4"
        "You are a code reviewer. Review this PR for bugs and improvements."
        pr_json
        api_key)

    # Post review as PR comment
    return review
}
```

### 4. Self-Healing System

```nano
fn auto_fix_issues() -> int {
    # Monitor issues with "bug" label
    # Analyze error patterns
    # Generate potential fixes
    # Create PR with fixes
    # Run tests automatically
    # Merge if tests pass

    return 0
}
```

## Local LLM Support

For privacy, cost savings, or offline operation, use local LLMs:

### Ollama Setup

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama2

# Run server
ollama serve
```

```nano
# Use in NanoLang
(openai_set_api_base "http://localhost:11434/v1")
let response: string = (openai_chat_completion_simple
    "llama2"
    ""
    "Analyze this bug report"
    "dummy-key")  # Not needed for local
```

### LM Studio Setup

1. Download from https://lmstudio.ai/
2. Load a model (Mistral, Llama 2, etc.)
3. Start local server

```nano
(openai_set_api_base "http://localhost:1234/v1")
```

## Security & Best Practices

### 1. Token Management

```nano
# Never hardcode tokens
# ❌ BAD:
let token: string = "ghp_xxxxxxxxxxxx"

# ✅ GOOD:
let token: string = (github_get_token_from_env)
```

### 2. Rate Limiting

```nano
fn check_rate_limit_before_batch() -> bool {
    let remaining: int = (github_check_rate_limit token)
    if (< remaining 100) {
        (println "Rate limit low, waiting...")
        return false
    }
    return true
}
```

### 3. Error Handling

```nano
let response: string = (github_get_issue owner repo 123 token)
if (str_contains response "\"error\"") {
    (println "API call failed, implement retry logic")
    return
}
```

### 4. Cost Control (OpenAI)

```nano
fn estimate_cost(text: string) -> int {
    let tokens: int = (openai_count_tokens_estimate text)
    if (> tokens 4000) {
        (println "Warning: This will use many tokens!")
    }
    return tokens
}
```

## Advanced Patterns

### 1. Multi-Step Reasoning

```nano
fn complex_analysis(issue_text: string) -> string {
    # Step 1: Classify issue type
    let classification: string = (openai_chat_completion_simple
        "gpt-4" "Classify this issue as bug, feature, or question."
        issue_text api_key)

    # Step 2: Deep analysis based on classification
    let analysis: string = (openai_chat_completion_simple
        "gpt-4"
        (str_concat "This is a " classification)
        issue_text
        api_key)

    return analysis
}
```

### 2. Iterative Refinement

```nano
fn generate_fix_with_feedback(bug_description: string) -> string {
    let mut fix: string = ""
    let mut iteration: int = 0

    while (< iteration 3) {
        # Generate fix
        set fix (openai_chat_completion_simple "gpt-4" "" bug_description api_key)

        # Validate fix (run tests, static analysis, etc.)
        let valid: bool = (validate_fix fix)

        if valid {
            return fix
        }

        # Refine prompt with feedback
        set bug_description (str_concat bug_description "\n\nPrevious attempt failed. Try again.")
        set iteration (+ iteration 1)
    }

    return fix
}
```

### 3. Context Accumulation

```nano
fn build_conversation_history(messages: array<string>) -> string {
    # Build JSON message array for multi-turn conversation
    let mut history: string = "["

    # Add messages iteratively
    # Return complete message history for context-aware responses

    return history
}
```

## Performance Considerations

### GitHub API
- **Rate Limit:** 5,000 requests/hour (authenticated)
- **Best Practice:** Check rate limit before batch operations
- **Caching:** Cache repository data when possible

### OpenAI API
- **Latency:** 1-10 seconds per request (varies by model)
- **Cost:** ~$0.03/1K tokens (GPT-4)
- **Optimization:** Use GPT-3.5-turbo for simple tasks

### Local LLMs
- **Latency:** Depends on hardware (GPU recommended)
- **Cost:** Free after initial setup
- **Privacy:** All data stays local

## Future Enhancements

Planned features:
- [ ] Streaming responses for real-time output
- [ ] Webhook integration for event-driven automation
- [ ] Vector database integration for semantic search
- [ ] Multi-repository coordination
- [ ] Automated deployment and rollback

## Resources

- **GitHub Module:** [`modules/github/README.md`](../modules/github/README.md)
- **OpenAI Module:** [`modules/openai/README.md`](../modules/openai/README.md)
- **Example Agent:** [`examples/ai_github_agent.nano`](../examples/ai_github_agent.nano)
- **GitHub API Docs:** https://docs.github.com/en/rest
- **OpenAI API Docs:** https://platform.openai.com/docs
- **Ollama:** https://ollama.ai/
- **LM Studio:** https://lmstudio.ai/

## Getting Started

1. **Install dependencies:**
   ```bash
   brew install curl  # macOS
   # or
   sudo apt install libcurl4-openssl-dev  # Linux
   ```

2. **Set up authentication:**
   ```bash
   export GITHUB_TOKEN='your_github_token'
   export OPENAI_API_KEY='your_openai_key'
   ```

3. **Run the example:**
   ```bash
   nanoc examples/ai_github_agent.nano -o agent
   ./agent
   ```

4. **Build your own agent:**
   - Start with the example code
   - Add your custom logic
   - Deploy and automate!

---

**NanoLang** - Now with autonomous agent capabilities! 🤖🚀
