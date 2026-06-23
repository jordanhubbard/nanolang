# GitHub API Module for NanoLang

Comprehensive GitHub REST API client enabling NanoLang programs to interact with GitHub repositories, issues, pull requests, and more.

## Features

- ✅ Repository information
- ✅ Issue management (list, get, create, update, comment)
- ✅ Pull request management (list, get, create)
- ✅ Authentication via GitHub token
- ✅ Rate limit checking
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

Set your GitHub personal access token as an environment variable:

```bash
export GITHUB_TOKEN='your_github_token_here'
```

Create a token at: https://github.com/settings/tokens

Required scopes:
- `repo` - Full control of private repositories (or `public_repo` for public only)
- `read:org` - Read org and team membership (optional)

## Quick Start

```nano
import "modules/github/github.nano"

fn main() -> int {
    let token: string = (github_get_token_from_env)

    # Get repository information
    let repo_info: string = (github_get_repo "jordanhubbard" "nanolang" token)
    (println repo_info)

    # List open issues
    let issues: string = (github_list_issues "jordanhubbard" "nanolang" "open" token)
    (println issues)

    return 0
}

shadow main { assert true }
```

## API Reference

### Configuration

#### `github_get_token_from_env() -> string`

Get GitHub token from `GITHUB_TOKEN` environment variable.

```nano
let token: string = (github_get_token_from_env)
```

### Repository Operations

#### `github_get_repo(owner: string, repo: string, token: string) -> string`

Get repository information. Returns JSON response.

```nano
let repo_info: string = (github_get_repo "octocat" "Hello-World" token)
```

### Issue Operations

#### `github_list_issues(owner: string, repo: string, state: string, token: string) -> string`

List issues. State can be: `"open"`, `"closed"`, or `"all"`.

```nano
let open_issues: string = (github_list_issues "owner" "repo" "open" token)
```

#### `github_get_issue(owner: string, repo: string, issue_number: int, token: string) -> string`

Get specific issue by number.

```nano
let issue: string = (github_get_issue "owner" "repo" 42 token)
```

#### `github_create_issue(owner: string, repo: string, title: string, body: string, token: string) -> string`

Create a new issue.

```nano
let result: string = (github_create_issue
    "owner"
    "repo"
    "Bug: Application crashes on startup"
    "Steps to reproduce:\n1. Launch app\n2. Click login\n3. Crash occurs"
    token)
```

#### `github_update_issue(owner: string, repo: string, issue_number: int, title: string, body: string, state: string, token: string) -> string`

Update an existing issue. Pass empty string for fields you don't want to change.

```nano
# Close an issue
let result: string = (github_update_issue "owner" "repo" 42 "" "" "closed" token)

# Update title and body
let result: string = (github_update_issue
    "owner" "repo" 42
    "Updated Title"
    "Updated description"
    ""
    token)
```

#### `github_create_issue_comment(owner: string, repo: string, issue_number: int, body: string, token: string) -> string`

Add a comment to an issue.

```nano
let result: string = (github_create_issue_comment
    "owner"
    "repo"
    42
    "Thank you for reporting this issue!"
    token)
```

### Pull Request Operations

#### `github_list_prs(owner: string, repo: string, state: string, token: string) -> string`

List pull requests. State can be: `"open"`, `"closed"`, or `"all"`.

```nano
let open_prs: string = (github_list_prs "owner" "repo" "open" token)
```

#### `github_get_pr(owner: string, repo: string, pr_number: int, token: string) -> string`

Get specific pull request by number.

```nano
let pr: string = (github_get_pr "owner" "repo" 123 token)
```

#### `github_create_pr(owner: string, repo: string, title: string, body: string, head: string, base: string, token: string) -> string`

Create a pull request.

```nano
let result: string = (github_create_pr
    "owner"
    "repo"
    "Fix bug in authentication"
    "This PR fixes the login issue by...\n\nCloses #42"
    "feature-branch"
    "main"
    token)
```

### Utility Functions

#### `github_check_rate_limit(token: string) -> int`

Check remaining API rate limit. Returns number of requests remaining, or -1 on error.

```nano
let remaining: int = (github_check_rate_limit token)
if (< remaining 10) {
    (println "Warning: Low rate limit remaining!")
}
```

## Response Format

All functions return JSON strings. Use the `json` module to parse responses:

```nano
from "modules/std/json/json.nano" import Json, parse, get_string, get_int

let issue_json: string = (github_get_issue "owner" "repo" 1 token)
let root: Json = (parse issue_json)
let title: string = (get_string root "title")
let number: int = (get_int root "number")

(println title)
```

## Error Handling

Errors are returned as JSON with an `"error"` field:

```nano
let result: string = (github_get_issue "owner" "repo" 999 token)
if (str_contains result "\"error\"") {
    (println "Failed to fetch issue")
    (println result)
}
```

## Rate Limiting

GitHub API has rate limits:
- **Authenticated:** 5,000 requests per hour
- **Unauthenticated:** 60 requests per hour

Check rate limit before making many requests:

```nano
let remaining: int = (github_check_rate_limit token)
(println (str_concat "Requests remaining: " (int_to_string remaining)))
```

## Examples

See `examples/ai_github_agent.nano` for a complete example demonstrating autonomous GitHub issue management with LLM integration.

## API Documentation

Full GitHub REST API documentation: https://docs.github.com/en/rest

## License

Part of the NanoLang project. See LICENSE file for details.
