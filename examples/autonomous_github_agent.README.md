# Autonomous GitHub Agent - Production Ready

A complete, production-ready autonomous agent that monitors GitHub issues and provides intelligent analysis using LLMs.

## Features

✅ **Full environment variable configuration** - Zero hardcoded values
✅ **Multiple authentication methods** - Personal tokens, GitHub Apps
✅ **Flexible LLM backends** - OpenAI, Azure, Ollama, LM Studio, any OpenAI-compatible API
✅ **Multiple analysis modes** - Analyze, triage, fix, monitor
✅ **Dry run mode** - Test without making changes
✅ **Rate limit protection** - Checks limits before operation
✅ **Configurable processing** - Control batch size and filters

## Quick Start

### 1. Set Up Environment Variables

```bash
# Copy the example configuration
cp autonomous_github_agent.env.example .env

# Edit .env with your values
nano .env

# Minimum required:
export GITHUB_OWNER="your-username"
export GITHUB_REPO="your-repo"
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxx"
```

### 2. Compile and Run

```bash
# Load environment
source .env

# Compile
nanoc autonomous_github_agent.nano -o autonomous_github_agent

# Run (dry run by default - safe!)
./autonomous_github_agent
```

### 3. Review Output

The agent will:
- ✅ Validate configuration
- ✅ Check rate limits
- ✅ Fetch open issues
- ✅ Analyze with LLM
- ✅ Display analysis (dry run)
- ⏸️ Post comments (if `AGENT_DRY_RUN="false"`)

## Configuration Reference

### GitHub Authentication

#### Personal Access Token (Recommended)

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**How to get:**
1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Copy token immediately (shown once only!)

#### GitHub App (Advanced)

```bash
export GITHUB_APP_ID="123456"
export GITHUB_APP_INSTALLATION_ID="12345678"
export GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"
```

**How to get:**
1. Create app at https://github.com/settings/apps
2. Install on repository
3. Download private key
4. Get app ID and installation ID

**Note:** GitHub App JWT signing coming soon. For now, generate installation token:

```bash
gh api -X POST /app/installations/{installation_id}/access_tokens
# Use the "token" field as GITHUB_TOKEN
```

### Repository Configuration

```bash
export GITHUB_OWNER="username"      # Required
export GITHUB_REPO="repository"     # Required
export GITHUB_ISSUE_LABEL="bug"     # Optional filter
```

### LLM Configuration

#### OpenAI (Default)

```bash
export OPENAI_API_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-proj-xxxx"
export OPENAI_MODEL="gpt-4"
```

**Cost:** ~$0.03/1K tokens (GPT-4), ~$0.002/1K tokens (GPT-3.5)

#### Ollama (Free, Local)

```bash
# Install: curl https://ollama.ai/install.sh | sh
# Download model: ollama pull llama2
# Start server: ollama serve

export OPENAI_API_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="llama2"
```

**Cost:** Free (runs on your GPU)

#### LM Studio (Free, Local)

```bash
# Download from https://lmstudio.ai/
# Load a model and start server

export OPENAI_API_URL="http://localhost:1234/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="local-model"
```

#### Azure OpenAI

```bash
export OPENAI_API_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
export OPENAI_API_KEY="your-azure-key"
export OPENAI_MODEL="gpt-4"
```

### Agent Modes

#### Analyze Mode (Default)

Comprehensive analysis with summary, type, severity, and next steps.

```bash
export AGENT_MODE="analyze"
```

**Output example:**
```
Summary: User reports application crash on startup with null pointer error.
Type: Bug
Severity: High
Next Steps:
- Add null checks in startup sequence
- Review initialization order
- Add error logging
```

#### Triage Mode

Classify issues and suggest labels.

```bash
export AGENT_MODE="triage"
```

**Output example:**
```
Suggested Labels: bug, critical, needs-investigation
Category: Runtime Error
Priority: High
```

#### Fix Mode

Analyze bugs and suggest specific fixes with code.

```bash
export AGENT_MODE="fix"
```

**Output example:**
```
Root Cause: Missing null check on line 42
Suggested Fix:
```c
if (ptr == NULL) {
    log_error("Invalid pointer");
    return ERROR_NULL_PTR;
}
```
```

## Usage Examples

### Example 1: Safe Dry Run

```bash
# Test configuration without making changes
export AGENT_DRY_RUN="true"
./autonomous_github_agent
```

### Example 2: Process Bug Issues Only

```bash
export GITHUB_ISSUE_LABEL="bug"
export AGENT_MAX_ISSUES="10"
./autonomous_github_agent
```

### Example 3: Production Mode (Posts Comments)

```bash
export AGENT_DRY_RUN="false"
export AGENT_MODE="analyze"
./autonomous_github_agent
```

### Example 4: Use Local LLM

```bash
# Start Ollama
ollama serve

# Configure
export OPENAI_API_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="llama2"
export AGENT_DRY_RUN="true"

# Run
./autonomous_github_agent
```

### Example 5: Scheduled Automation

```bash
# Add to crontab for hourly runs
0 * * * * cd /path/to/nanolang && source .env && ./autonomous_github_agent >> /var/log/github-agent.log 2>&1
```

## Output Example

```
╔══════════════════════════════════════════════════════╗
║  🤖 Autonomous GitHub Agent - Production Ready  🚀  ║
╚══════════════════════════════════════════════════════╝

📋 Agent Configuration:
========================
  GitHub: jordanhubbard
          nanolang
  Token:  ghp_abcdef...masked

  LLM URL:   https://api.openai.com/v1
  LLM Model: gpt-4

  Mode:       analyze
  Dry Run:    YES (no actions taken)
  Max Issues: 5

✅ Configuration valid

📊 Checking GitHub rate limits...
   4,987 requests remaining

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Starting agent...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Fetching issues...
✅ Issues fetched
   Found 3 issues (will process up to 5)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 Issue #42
   Memory leak in parser module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Valgrind reports memory leak when parsing large files...

🤔 Analyzing with LLM...

📊 Analysis:
Summary: Memory leak occurs in parser when processing files >10MB
Type: Bug
Severity: High
Next Steps:
- Add proper memory cleanup in parse_large_file()
- Implement chunked parsing for large files
- Add valgrind to CI pipeline

🔵 DRY RUN: Would post this comment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Agent finished
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Processed: 1 issues
```

## Deployment Options

### Option 1: Cron Job

```bash
# Edit crontab
crontab -e

# Add hourly run
0 * * * * cd /path/to/project && source .env && ./autonomous_github_agent
```

### Option 2: GitHub Actions

```yaml
name: Autonomous Agent
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:  # Manual trigger

jobs:
  run-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Agent
        env:
          GITHUB_TOKEN: ${{ secrets.AGENT_GITHUB_TOKEN }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GITHUB_OWNER: ${{ github.repository_owner }}
          GITHUB_REPO: ${{ github.event.repository.name }}
          AGENT_DRY_RUN: "false"
        run: |
          make
          ./bin/nanoc examples/autonomous_github_agent.nano -o agent
          ./agent
```

### Option 3: Docker Container

```dockerfile
FROM nanolang:latest

COPY autonomous_github_agent.nano /app/
RUN nanoc /app/autonomous_github_agent.nano -o /app/agent

ENV AGENT_DRY_RUN=false
CMD ["/app/agent"]
```

## Troubleshooting

### "GITHUB_TOKEN not set"

```bash
# Make sure to export (not just set)
export GITHUB_TOKEN="ghp_xxx"  # ✅ Correct
GITHUB_TOKEN="ghp_xxx"         # ❌ Wrong (not exported)
```

### "Could not check rate limit"

- Token may be invalid or expired
- Token may lack `repo` scope
- Check: `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit`

### "LLM Error: curl failed"

- Check OPENAI_API_URL is accessible
- For local LLMs, ensure server is running:
  ```bash
  # Ollama
  curl http://localhost:11434/v1/models

  # LM Studio
  curl http://localhost:1234/v1/models
  ```

### Rate Limit Warnings

- GitHub allows 5,000 req/hour
- Use `AGENT_MAX_ISSUES` to limit processing
- Add delays between runs

## Security Best Practices

1. **Never commit tokens**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use minimum scopes**
   - Personal token: `public_repo` for public repos only
   - GitHub App: Only required permissions

3. **Rotate credentials**
   - Rotate tokens every 90 days
   - Use short-lived installation tokens

4. **Use secrets management**
   - GitHub Actions: Use secrets
   - Production: Use vault/secrets manager

5. **Monitor usage**
   - Check rate limits regularly
   - Monitor for unauthorized access

## Advanced Features (Coming Soon)

- [ ] GitHub App JWT authentication
- [ ] Multi-repository processing
- [ ] Webhook integration (real-time)
- [ ] Custom analysis templates
- [ ] Automated PR creation
- [ ] Test execution before commenting
- [ ] Multi-LLM consensus (vote on analysis)

## Support

- **Module Docs:** See `modules/github/README.md` and `modules/openai/README.md`
- **Issues:** https://github.com/jordanhubbard/nanolang/issues
- **Examples:** See `examples/ai_github_agent.nano` for simpler version

---

**Built with NanoLang** - Now with autonomous capabilities! 🤖🚀
