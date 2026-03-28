# 13.2 log — Structured Logging

**Multi-level, category-aware logging for debugging and production monitoring.**

The `stdlib/log.nano` module provides a lightweight structured logging system. Every message carries a severity level and an optional category string, making it easy to filter output by component or subsystem. The output format is human-readable by default and consistent enough to be parsed by log-aggregation tools.

The module is intentionally simple: there is no complex configuration file, no global state you have to initialize, and no teardown required. Import the functions you need and start logging.

---

## Quick Start

```nano
from "stdlib/log.nano" import log_info, log_warn, log_error

fn process_order(order_id: int) -> bool {
    (log_info "orders" (+ "Processing order #" (int_to_string order_id)))

    if (< order_id 1) {
        (log_error "orders" "Invalid order ID — must be positive")
        return false
    }

    (log_info "orders" "Order validated successfully")
    return true
}

shadow process_order {
    assert (process_order 42)
    assert (not (process_order 0))
}
```

Output:
```
[INFO] orders: Processing order #42
[INFO] orders: Order validated successfully
```

---

## Import

```nano
# Category-aware functions (recommended for libraries and components):
from "stdlib/log.nano" import log_trace, log_debug, log_info, log_warn, log_error, log_fatal

# Category-less convenience functions:
from "stdlib/log.nano" import trace, debug, info, warn, error, fatal

# Level introspection:
from "stdlib/log.nano" import log_get_level
```

---

## Log Levels

The log module defines six severity levels, ordered from least to most critical:

| Level | Integer | Constant | Typical Use |
|-------|---------|----------|-------------|
| TRACE | 0 | `LogLevel.TRACE` | Extremely verbose — call paths, inner loop state |
| DEBUG | 1 | `LogLevel.DEBUG` | Development detail — variable values, branch taken |
| INFO  | 2 | `LogLevel.INFO`  | Normal operations — requests served, jobs completed |
| WARN  | 3 | `LogLevel.WARN`  | Recoverable anomalies — retried operations, fallback paths |
| ERROR | 4 | `LogLevel.ERROR` | Failures that the system handles — bad input, transient errors |
| FATAL | 5 | `LogLevel.FATAL` | Unrecoverable failures — about to terminate |

The **default threshold is INFO** (level 2). Messages at TRACE and DEBUG are suppressed unless the threshold is lowered. Messages at INFO and above are always emitted in the default configuration.

---

## Output Format

Every log message follows this format:

```
[LEVEL] category: message
```

When category is the empty string `""`, the `category: ` prefix is omitted:

```
[INFO] message text here
```

**Examples:**

```
[TRACE] parser: Entering expression parser at offset 1042
[DEBUG] cache: Cache miss for key user_profile_42
[INFO] server: Listening on port 8080
[WARN] auth: Token expiring in 5 minutes for user 7
[ERROR] database: Query timeout after 30s — retrying
[FATAL] runtime: Out of memory — cannot continue
```

---

## API Reference

### `log_trace`

```nano
fn log_trace(category: string, message: string) -> void
```

Emits a TRACE-level message. Only visible when the log level is explicitly lowered to TRACE. Use for extremely detailed information: entering/leaving small functions, per-element loop state, intermediate computed values during debugging sessions.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `category` | `string` | Subsystem or component name, e.g. `"parser"` |
| `message` | `string` | The message text |

**Example:**

```nano
from "stdlib/log.nano" import log_trace

fn scan_token(input: string, pos: int) -> int {
    (log_trace "lexer" (+ "Scanning at pos " (int_to_string pos)))
    # ... tokenization logic ...
    return pos
}
```

---

### `log_debug`

```nano
fn log_debug(category: string, message: string) -> void
```

Emits a DEBUG-level message. Suppressed at the default INFO threshold. Use during development to log variable state, control flow decisions, and diagnostic information that would be too noisy in production.

**Example:**

```nano
from "stdlib/log.nano" import log_debug

fn load_config(path: string) -> bool {
    (log_debug "config" (+ "Loading config from: " path))
    # ... load logic ...
    (log_debug "config" "Config loaded successfully")
    return true
}
```

---

### `log_info`

```nano
fn log_info(category: string, message: string) -> void
```

Emits an INFO-level message. This is the default threshold — INFO messages are always visible unless the threshold is raised. Use for normal, expected events that are useful to see in production: service start/stop, jobs completed, requests received.

**Example:**

```nano
from "stdlib/log.nano" import log_info

fn start_server(port: int) -> void {
    (log_info "server" (+ "Starting HTTP server on port " (int_to_string port)))
}

fn job_complete(job_id: int, item_count: int) -> void {
    (log_info "jobs" (+ "Job #" (+ (int_to_string job_id)
                    (+ " processed " (+ (int_to_string item_count) " items")))))
}
```

---

### `log_warn`

```nano
fn log_warn(category: string, message: string) -> void
```

Emits a WARN-level message. Use when something unexpected happened but the system has recovered or can continue. Examples: a retry succeeded, a configuration value is missing and a default was applied, resource usage is approaching a limit.

**Example:**

```nano
from "stdlib/log.nano" import log_warn

fn connect_with_retry(host: string, max_retries: int) -> bool {
    let mut attempt: int = 0
    while (< attempt max_retries) {
        # ... attempt connection ...
        if (== attempt 0) {
            return true
        }
        (log_warn "network" (+ "Connection failed, retrying (attempt "
                             (+ (int_to_string attempt) ")")))
        set attempt (+ attempt 1)
    }
    return false
}
```

---

### `log_error`

```nano
fn log_error(category: string, message: string) -> void
```

Emits an ERROR-level message. Use when a failure has occurred that the system is handling, but that represents a real problem: a user request could not be served, a database query failed, an expected file was missing. The program continues, but something went wrong.

**Example:**

```nano
from "stdlib/log.nano" import log_error

fn read_user_record(user_id: int) -> bool {
    if (< user_id 1) {
        (log_error "database" (+ "Invalid user ID: " (int_to_string user_id)))
        return false
    }
    # ... database fetch ...
    return true
}
```

---

### `log_fatal`

```nano
fn log_fatal(category: string, message: string) -> void
```

Emits a FATAL-level message. Use immediately before aborting, exiting, or propagating an unrecoverable error. A FATAL log entry signals that the system cannot continue in a valid state.

**Example:**

```nano
from "stdlib/log.nano" import log_fatal

fn require_config_file(path: string) -> void {
    # If this fails, nothing else can work
    (log_fatal "startup" (+ "Required config file not found: " path))
    # caller is expected to exit after this
}
```

---

### Category-Less Convenience Functions

For scripts and small programs where a category is unnecessary, the module exports single-parameter variants that omit the category prefix from output:

```nano
fn trace(message: string) -> void
fn debug(message: string) -> void
fn info(message: string) -> void
fn warn(message: string) -> void
fn error(message: string) -> void
fn fatal(message: string) -> void
```

**Import:**

```nano
from "stdlib/log.nano" import info, warn, error
```

**Example:**

```nano
from "stdlib/log.nano" import info, error

fn main() -> int {
    (info "Application starting")

    let ok: bool = true   # ... some initialization ...
    if (not ok) {
        (error "Initialization failed")
        return 1
    }

    (info "Ready")
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

Output:
```
[INFO] Application starting
[INFO] Ready
```

---

### `log_get_level`

```nano
fn log_get_level() -> int
```

Returns the current minimum log level as an integer. Messages below this level are suppressed. The default is `2` (INFO), exposed as the constant `default_log_level`.

**Example:**

```nano
from "stdlib/log.nano" import log_get_level

fn is_debug_mode() -> bool {
    return (< (log_get_level) 2)   # true when level is TRACE or DEBUG
}
```

---

## Examples

### Example 1: Basic Application Lifecycle Logging

```nano
from "stdlib/log.nano" import log_info, log_error, log_warn

fn run_app() -> int {
    (log_info "app" "Starting up")

    let db_ok: bool = true  # connect to database
    if (not db_ok) {
        (log_error "app" "Failed to connect to database — aborting")
        return 1
    }

    (log_info "app" "Database connection established")
    (log_warn "app" "Running with default configuration — set APP_CONFIG to customize")
    (log_info "app" "Ready to serve requests")

    # ... serve ...

    (log_info "app" "Shutting down gracefully")
    return 0
}

shadow run_app {
    assert (== (run_app) 0)
}
```

### Example 2: Per-Component Category Discipline

Organizing log categories by component makes it easy to search logs for specific subsystem activity.

```nano
from "stdlib/log.nano" import log_info, log_debug, log_error, log_warn

fn authenticate_user(user_id: int, token: string) -> bool {
    (log_debug "auth" (+ "Checking token for user " (int_to_string user_id)))

    if (== (str_length token) 0) {
        (log_warn "auth" "Empty token presented — rejecting")
        return false
    }

    # ... validate token ...
    (log_info "auth" (+ "User " (+ (int_to_string user_id) " authenticated")))
    return true
}

fn store_session(user_id: int) -> bool {
    (log_debug "session" (+ "Creating session for user " (int_to_string user_id)))
    # ... write to store ...
    (log_info "session" "Session created")
    return true
}

fn handle_login(user_id: int, token: string) -> bool {
    if (not (authenticate_user user_id token)) {
        (log_error "login" "Authentication failed — login rejected")
        return false
    }
    return (store_session user_id)
}

shadow handle_login {
    assert (handle_login 1 "valid-token")
    assert (not (handle_login 1 ""))
}
```

### Example 3: Request/Response Logging Pattern

A common web-service pattern: log at INFO when a request arrives, DEBUG for processing detail, and ERROR on failure.

```nano
from "stdlib/log.nano" import log_info, log_debug, log_error

fn handle_request(path: string, method: string) -> int {
    (log_info "http" (+ method (+ " " path)))
    (log_debug "http" (+ "Routing: " path))

    # ... route and process ...
    let status: int = 200

    if (== status 200) {
        (log_debug "http" "Response: 200 OK")
    } else {
        (log_error "http" (+ "Response error: " (int_to_string status)))
    }

    return status
}

shadow handle_request {
    assert (== (handle_request "/api/users" "GET") 200)
}
```

### Example 4: Logging Values with `int_to_string`

The log module accepts only `string` messages. Use `int_to_string` and `float_to_string` to embed numeric values.

```nano
from "stdlib/log.nano" import log_info, log_warn

fn report_cache_stats(hits: int, misses: int) -> void {
    let total: int = (+ hits misses)
    (log_info "cache" (+ "Cache stats — hits: " (+ (int_to_string hits)
                        (+ ", misses: " (int_to_string misses)))))

    if (> total 0) {
        if (< hits (/ total 2)) {
            (log_warn "cache" "Hit rate below 50% — consider increasing cache size")
        }
    }
}

shadow report_cache_stats {
    (report_cache_stats 80 20)
    (report_cache_stats 10 90)
}
```

### Example 5: Tracing Function Entry and Exit

Use TRACE to instrument function boundaries during deep debugging sessions. Remember that TRACE messages are suppressed by default and will not appear in production without changing the log threshold.

```nano
from "stdlib/log.nano" import log_trace, log_debug, log_info

fn compute_price(base: int, discount_pct: int) -> int {
    (log_trace "pricing" (+ "compute_price called with base=" (+ (int_to_string base)
                           (+ " discount=" (int_to_string discount_pct)))))

    let discount: int = (/ (* base discount_pct) 100)
    let final_price: int = (- base discount)

    (log_debug "pricing" (+ "Discount amount: " (int_to_string discount)))
    (log_trace "pricing" (+ "compute_price returning " (int_to_string final_price)))

    return final_price
}

shadow compute_price {
    assert (== (compute_price 100 20) 80)
    assert (== (compute_price 50 10) 45)
}
```

---

## Best Practices

**Choose the right level.** The most common mistake is reaching for `log_error` for anything that goes wrong, including routine validation failures. Ask: does this represent a bug or infrastructure failure (`ERROR`), a recoverable anomaly (`WARN`), or normal operation (`INFO`)? A user submitting an invalid form is not an error — it is expected behavior that warrants `INFO` at most.

**Include context in the message.** A message like `"Failed"` is useless in production. Include the relevant IDs, values, and conditions that let you reproduce or understand the problem without re-reading the code.

```nano
# Too vague:
(log_error "db" "Query failed")

# Much more useful:
(log_error "db" (+ "Query failed for user_id=" (+ (int_to_string uid)
               (+ " table=" table))))
```

**Use consistent category names across a codebase.** Pick a fixed vocabulary (`"auth"`, `"db"`, `"http"`, `"cache"`, `"jobs"`) and use those exact strings everywhere. Inconsistent casing or spelling fragments your log output across spurious categories.

**Never log secrets.** Passwords, tokens, API keys, and personally-identifying information must never appear in log output, even at DEBUG or TRACE level. Log IDs and opaque identifiers instead.

```nano
# Security violation — never do this:
(log_debug "auth" (+ "Checking password: " password))

# Safe:
(log_debug "auth" (+ "Checking credentials for user_id=" (int_to_string uid)))
```

**Avoid logging in hot loops.** Calling `log_debug` on every iteration of a 100,000-element loop will dominate your runtime even if the messages are suppressed (the string concatenation still occurs). Log before and after the loop, not inside it.

```nano
from "stdlib/log.nano" import log_info, log_debug

fn process_items(items: array<string>) -> int {
    let count: int = (array_length items)
    (log_info "batch" (+ "Processing " (int_to_string count) " items"))

    let mut i: int = 0
    let mut ok: int = 0
    while (< i count) {
        # Do NOT call log_debug here per-item unless count is small
        set ok (+ ok 1)
        set i (+ i 1)
    }

    (log_info "batch" (+ "Done — processed " (int_to_string ok) " items"))
    return ok
}
```

**The six-level discipline — a quick reference:**

| Situation | Level |
|-----------|-------|
| Entering a small internal function | TRACE |
| Printing a variable mid-computation | DEBUG |
| Request received, service started | INFO |
| Retry, fallback, approaching a limit | WARN |
| Request failed, file not found | ERROR |
| Cannot continue — about to abort | FATAL |

---

**Previous:** [13.1 regex](regex.html)
**Next:** [13.3 StringBuilder](stringbuilder.html)
