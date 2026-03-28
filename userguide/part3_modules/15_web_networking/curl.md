# 15.1 curl - HTTP Client

**Make HTTP requests to web APIs and services.**

The `curl` module wraps libcurl, one of the most widely deployed HTTP client libraries. Use it to fetch data from REST APIs, send form and JSON payloads, download files, and build custom requests with full header and timeout control. The module provides both a simple high-level API for common cases and a lower-level handle-based API for advanced configuration.

## Installation

libcurl must be available on the system. On most Linux distributions it is provided by `libcurl4-openssl-dev` (or equivalent); on macOS it ships with Xcode Command Line Tools.

Import the symbols you need:

```nano
from "modules/curl/curl.nano" import nl_curl_simple_get, nl_curl_simple_post,
    nl_curl_download_file,
    nl_curl_global_init, nl_curl_global_cleanup,
    nl_curl_easy_init, nl_curl_easy_cleanup,
    nl_curl_easy_setopt_url, nl_curl_easy_setopt_follow_location,
    nl_curl_easy_setopt_timeout, nl_curl_easy_setopt_useragent,
    nl_curl_easy_perform, nl_curl_easy_getinfo_response_code,
    curl_global_init_safe, curl_global_cleanup_safe,
    curl_easy_init_safe, curl_easy_cleanup_safe
```

Most programs only need `nl_curl_simple_get` and `nl_curl_simple_post`.

## Quick Start

```nano
from "modules/curl/curl.nano" import nl_curl_simple_get

fn fetch_json(url: string) -> string {
    unsafe {
        return (nl_curl_simple_get url)
    }
}

shadow fetch_json { assert true }
```

---

## API Reference

### Simple High-Level Functions

These are the recommended starting point. They handle initialization and cleanup internally.

#### `extern fn nl_curl_simple_get(_url: string) -> string`

Perform an HTTP GET request and return the response body as a string. Returns an empty string on network error. Must be called inside an `unsafe` block.

| Parameter | Type | Description |
|-----------|------|-------------|
| `_url` | `string` | The URL to fetch (must include scheme, e.g. `https://`) |

**Returns:** `string` — response body, or `""` on failure.

```nano
unsafe {
    let body: string = (nl_curl_simple_get "https://api.example.com/status")
    if (== (str_length body) 0) {
        (println "Request failed or empty response")
    } else {
        (println body)
    }
}
```

#### `extern fn nl_curl_simple_post(_url: string, _data: string) -> string`

Perform an HTTP POST request with `_data` as the raw body. Returns the response body. Must be called inside an `unsafe` block.

| Parameter | Type | Description |
|-----------|------|-------------|
| `_url` | `string` | The endpoint URL |
| `_data` | `string` | Raw body to POST (e.g. JSON text, form data) |

**Returns:** `string` — response body, or `""` on failure.

```nano
unsafe {
    let payload: string = "{\"event\": \"login\", \"user\": \"alice\"}"
    let response: string = (nl_curl_simple_post "https://api.example.com/events" payload)
}
```

#### `extern fn nl_curl_download_file(_url: string, _output_path: string) -> int`

Download a URL to a local file. Returns `0` on success, non-zero on failure. Must be called inside an `unsafe` block.

| Parameter | Type | Description |
|-----------|------|-------------|
| `_url` | `string` | URL of the resource to download |
| `_output_path` | `string` | Local file path where the content will be written |

**Returns:** `int` — `0` on success, non-zero on error.

```nano
unsafe {
    let rc: int = (nl_curl_download_file "https://example.com/data.csv" "/tmp/data.csv")
    if (!= rc 0) {
        (println "Download failed")
    } else {
        (println "Download complete")
    }
}
```

---

### Global Initialization

libcurl requires a one-time global initialization before any requests. For most programs you can rely on `nl_curl_simple_get` and `nl_curl_simple_post`, which handle this internally. When using the handle-based API directly, call these explicitly.

#### `fn curl_global_init_safe() -> int`

Initialize the global libcurl state. Returns `0` on success. Call once at program startup.

#### `fn curl_global_cleanup_safe() -> void`

Release all global libcurl resources. Call once at program shutdown, after all handles have been cleaned up.

```nano
from "modules/curl/curl.nano" import curl_global_init_safe, curl_global_cleanup_safe

fn main() -> int {
    let rc: int = (curl_global_init_safe)
    if (!= rc 0) {
        (println "curl init failed")
        return 1
    } else {
        (print "")
    }
    # ... do work ...
    (curl_global_cleanup_safe)
    return 0
}

shadow main { assert true }
```

---

### Handle-Based API

For advanced use cases — custom user agents, timeouts, redirect control — create and configure a curl handle directly.

#### `fn curl_easy_init_safe() -> int`

Create a new curl easy handle. Returns a non-zero handle on success, `0` on failure.

#### `fn curl_easy_cleanup_safe(handle: int) -> void`

Destroy a curl easy handle and release its resources. Always call this when done with a handle.

#### `extern fn nl_curl_easy_setopt_url(_handle: int, _url: string) -> int`

Set the URL for the handle. Returns `0` on success.

#### `extern fn nl_curl_easy_setopt_follow_location(_handle: int, _follow: int) -> int`

Control redirect following. Pass `1` to follow redirects (default behavior of the simple functions), `0` to disable. Returns `0` on success.

#### `extern fn nl_curl_easy_setopt_timeout(_handle: int, _timeout_secs: int) -> int`

Set a maximum time in seconds for the entire transfer. Pass `0` to disable the timeout. Returns `0` on success.

#### `extern fn nl_curl_easy_setopt_useragent(_handle: int, _useragent: string) -> int`

Set the User-Agent header string. Returns `0` on success.

#### `extern fn nl_curl_easy_perform(_handle: int) -> int`

Execute the request that has been configured on the handle. Returns `0` (`CURLE_OK`) on success, a non-zero libcurl error code on failure.

#### `extern fn nl_curl_easy_getinfo_response_code(_handle: int) -> int`

After a successful `nl_curl_easy_perform`, retrieve the HTTP response status code (e.g. `200`, `404`, `500`).

```nano
from "modules/curl/curl.nano" import curl_easy_init_safe, curl_easy_cleanup_safe,
    nl_curl_easy_setopt_url, nl_curl_easy_setopt_follow_location,
    nl_curl_easy_setopt_timeout, nl_curl_easy_setopt_useragent,
    nl_curl_easy_perform, nl_curl_easy_getinfo_response_code

fn check_url_status(url: string) -> int {
    let handle: int = (curl_easy_init_safe)
    if (== handle 0) {
        return -1
    } else {
        (print "")
    }

    unsafe {
        (nl_curl_easy_setopt_url handle url)
        (nl_curl_easy_setopt_follow_location handle 1)
        (nl_curl_easy_setopt_timeout handle 10)
        (nl_curl_easy_setopt_useragent handle "NanoLang/1.0")

        let rc: int = (nl_curl_easy_perform handle)
        let status: int = 0
        if (== rc 0) {
            let status: int = (nl_curl_easy_getinfo_response_code handle)
            (curl_easy_cleanup_safe handle)
            return status
        } else {
            (curl_easy_cleanup_safe handle)
            return -1
        }
    }
}

shadow check_url_status { assert true }
```

---

## Examples

### Example 1: Fetching and Parsing a JSON API

```nano
from "modules/curl/curl.nano" import nl_curl_simple_get
from "modules/std/json/json.nano" import parse, free, get_string, get_int, Json

struct Repo {
    name: string,
    stars: int,
    language: string
}

fn fetch_repo(owner: string, repo: string) -> Repo {
    let url: string = (+ "https://api.github.com/repos/" (+ owner (+ "/" repo)))
    unsafe {
        let body: string = (nl_curl_simple_get url)
        if (== (str_length body) 0) {
            return Repo { name: "", stars: 0, language: "" }
        } else {
            let obj: Json = (parse body)
            let r: Repo = Repo {
                name:     (get_string obj "full_name"),
                stars:    (get_int obj "stargazers_count"),
                language: (get_string obj "language")
            }
            (free obj)
            return r
        }
    }
}

shadow fetch_repo { assert true }
```

### Example 2: POSTing JSON to a REST API

```nano
from "modules/curl/curl.nano" import nl_curl_simple_post
from "modules/std/json/json.nano" import new_object, new_string, new_int, object_set, stringify, free, parse, get_int, Json

fn create_item(name: string, quantity: int) -> int {
    # Build request body
    let body_obj: Json = (new_object)
    (object_set body_obj "name" (new_string name))
    (object_set body_obj "quantity" (new_int quantity))
    let body: string = (stringify body_obj)
    (free body_obj)

    unsafe {
        let response: string = (nl_curl_simple_post "https://api.example.com/items" body)
        if (== (str_length response) 0) {
            return 0
        } else {
            # Parse response to get the created item's ID
            let resp_obj: Json = (parse response)
            let new_id: int = (get_int resp_obj "id")
            (free resp_obj)
            return new_id
        }
    }
}

shadow create_item { assert true }
```

### Example 3: Downloading a File

```nano
from "modules/curl/curl.nano" import nl_curl_download_file

fn download_release(version: string, dest_dir: string) -> bool {
    let url: string = (+ "https://releases.example.com/v" (+ version "/release.tar.gz"))
    let path: string = (+ dest_dir "/release.tar.gz")
    unsafe {
        let rc: int = (nl_curl_download_file url path)
        return (== rc 0)
    }
}

shadow download_release { assert true }
```

### Example 4: Health Check with Status Code Inspection

```nano
from "modules/curl/curl.nano" import curl_easy_init_safe, curl_easy_cleanup_safe,
    nl_curl_easy_setopt_url, nl_curl_easy_setopt_timeout,
    nl_curl_easy_perform, nl_curl_easy_getinfo_response_code

fn is_service_healthy(base_url: string) -> bool {
    let url: string = (+ base_url "/health")
    let handle: int = (curl_easy_init_safe)
    if (== handle 0) {
        return false
    } else {
        (print "")
    }
    unsafe {
        (nl_curl_easy_setopt_url handle url)
        (nl_curl_easy_setopt_timeout handle 5)
        let rc: int = (nl_curl_easy_perform handle)
        if (!= rc 0) {
            (curl_easy_cleanup_safe handle)
            return false
        } else {
            let code: int = (nl_curl_easy_getinfo_response_code handle)
            (curl_easy_cleanup_safe handle)
            return (== code 200)
        }
    }
}

shadow is_service_healthy { assert true }
```

### Example 5: Simple API Client Module Pattern

```nano
from "modules/curl/curl.nano" import nl_curl_simple_get, nl_curl_simple_post
from "modules/std/json/json.nano" import parse, free, get_string, get_int, new_object, new_string, object_set, stringify, Json

fn api_get(base_url: string, path: string) -> Json {
    let url: string = (+ base_url path)
    unsafe {
        let body: string = (nl_curl_simple_get url)
        if (== (str_length body) 0) {
            return 0
        } else {
            return (parse body)
        }
    }
}

fn api_post(base_url: string, path: string, payload: string) -> Json {
    let url: string = (+ base_url path)
    unsafe {
        let body: string = (nl_curl_simple_post url payload)
        if (== (str_length body) 0) {
            return 0
        } else {
            return (parse body)
        }
    }
}

fn get_user_name(base_url: string, user_id: int) -> string {
    let path: string = (+ "/users/" (int_to_string user_id))
    let resp: Json = (api_get base_url path)
    if (== resp 0) {
        return ""
    } else {
        let name: string = (get_string resp "name")
        (free resp)
        return name
    }
}

shadow get_user_name { assert true }
```

---

## Common Pitfalls

### Pitfall 1: Forgetting the unsafe block

All `extern` curl functions require an `unsafe` block. The safe wrapper functions (`curl_global_init_safe`, etc.) do not, because the `unsafe` is inside their implementation.

```nano
# WRONG — compile error
let body: string = (nl_curl_simple_get "https://example.com")

# CORRECT
unsafe {
    let body: string = (nl_curl_simple_get "https://example.com")
}
```

### Pitfall 2: Not checking for empty response

Network failures, DNS errors, and timeouts all cause `nl_curl_simple_get` and `nl_curl_simple_post` to return `""`. Calling `parse` on an empty string will crash.

```nano
# WRONG
unsafe {
    let body: string = (nl_curl_simple_get url)
    let obj: Json = (parse body)   # Crash if body is ""
}

# CORRECT
unsafe {
    let body: string = (nl_curl_simple_get url)
    if (== (str_length body) 0) {
        (println "Request failed")
    } else {
        let obj: Json = (parse body)
        # ... use obj ...
        (free obj)
    }
}
```

### Pitfall 3: Leaking curl handles

When using the handle-based API, always call `curl_easy_cleanup_safe` on every code path — including error paths.

```nano
# WRONG — leaks handle on early return
let handle: int = (curl_easy_init_safe)
unsafe {
    let rc: int = (nl_curl_easy_perform handle)
    if (!= rc 0) {
        return -1   # handle leaked
    }
    (curl_easy_cleanup_safe handle)
}

# CORRECT
let handle: int = (curl_easy_init_safe)
unsafe {
    let rc: int = (nl_curl_easy_perform handle)
    (curl_easy_cleanup_safe handle)
    if (!= rc 0) {
        return -1
    } else {
        (print "")
    }
}
return 0
```

### Pitfall 4: Sending JSON without setting Content-Type

`nl_curl_simple_post` sends the body as-is but does not set a `Content-Type` header. Many APIs require `Content-Type: application/json`. For simple cases this often works anyway, but if the server rejects your POST, add headers via the handle-based API.

### Pitfall 5: Ignoring HTTP status codes

A successful libcurl transfer (return code `0`) means the HTTP conversation completed — but the server may have returned a `4xx` or `5xx` error. Use `nl_curl_easy_getinfo_response_code` with the handle API to inspect the HTTP status, or check the response body for error indicators.

---

## Best Practices

- Use `nl_curl_simple_get` and `nl_curl_simple_post` for straightforward requests. Reach for the handle API only when you need timeouts, redirect control, or custom user agents.
- Always check that the response string is non-empty before attempting to parse it as JSON.
- Set a timeout (`nl_curl_easy_setopt_timeout`) on long-running or production requests to avoid indefinite hangs.
- In shadow tests, use `assert true` rather than making real network calls — tests should be fast and offline-safe.
- Validate URLs before passing them to curl: ensure they begin with `http://` or `https://`.

---

**Previous:** [Chapter 15 Overview](index.html)
**Next:** [15.2 http_server - Building Web Services](http_server.html)
