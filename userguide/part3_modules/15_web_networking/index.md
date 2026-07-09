# Chapter 15: Web & Networking

**HTTP clients and servers for web communication.**

This chapter covers web and networking modules: making HTTP requests with curl and creating HTTP servers.

## 15.1 HTTP Client (curl)

The `modules/curl/curl.nano` module provides HTTP client functionality.

### Simple GET Request

```nano
from "modules/curl/curl.nano" import get

fn fetch_webpage() -> string {
    let response: string = (get "https://example.com")
    return response
}

shadow fetch_webpage {
    # Would test with actual HTTP request
    assert true
}
```

### GET with Headers

```nano
from "modules/curl/curl.nano" import get_with_headers

fn fetch_with_auth() -> string {
    let headers: array<string> = ["Authorization: Bearer token123", "Accept: application/json"]
    let response: string = (get_with_headers "https://api.example.com/data" headers)
    return response
}

shadow fetch_with_auth {
    assert true
}
```

### POST Request

```nano
from "modules/curl/curl.nano" import post

fn send_data() -> string {
    let data: string = "{\"name\": \"Alice\", \"age\": 30}"
    let response: string = (post "https://api.example.com/users" data)
    return response
}

shadow send_data {
    assert true
}
```

### POST with Headers

```nano
from "modules/curl/curl.nano" import post_with_headers

fn send_json() -> string {
    let data: string = "{\"message\": \"hello\"}"
    let headers: array<string> = ["Content-Type: application/json"]
    let response: string = (post_with_headers "https://api.example.com/messages" data headers)
    return response
}

shadow send_json {
    assert true
}
```

### Download File

```nano
from "modules/curl/curl.nano" import download

fn download_image(url: string, path: string) -> bool {
    let result: int = (download url path)
    return (== result 0)
}

shadow download_image {
    assert true
}
```

### Complete Example: API Client

```nano
from "modules/curl/curl.nano" import get, post_with_headers
from "modules/std/json/json.nano" import parse, get_string, Json

fn fetch_user(user_id: int) -> string {
    let url: string = (+ "https://api.example.com/users/" (int_to_string user_id))
    let response: string = (get url)

    let json: Json = (parse response)
    let name: string = (get_string json "name")

    return name
    # No free() needed - automatic GC!
}

fn create_user(name: string, email: string) -> bool {
    let data: string = (+ "{\"name\": \"" (+ name (+ "\", \"email\": \"" (+ email "\"}"))))
    let headers: array<string> = ["Content-Type: application/json"]
    let response: string = (post_with_headers "https://api.example.com/users" data headers)

    return (> (str_length response) 0)
}

shadow create_user {
    assert true
}
```

## 15.2 HTTP Server

The `modules/http_server/http_server.nano` module provides HTTP server functionality.

### Basic Server

```nano
from "modules/http_server/http_server.nano" import create_server, start_server, stop_server

fn run_server() -> int {
    let server: int = (create_server 8080)
    (start_server server)
    # Server runs...
    (stop_server server)
    return 0
}

shadow run_server {
    # Would test with actual server
    assert true
}
```

### Handling Routes

```nano
from "modules/http_server/http_server.nano" import create_server, register_route, start_server

fn hello_handler(request: string) -> string {
    return "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nHello, World!"
}

fn setup_routes() -> int {
    let server: int = (create_server 8080)
    (register_route server "GET" "/" hello_handler)
    (start_server server)
    return 0
}

shadow setup_routes {
    assert true
}
```

### JSON API Endpoint

```nano
from "modules/http_server/http_server.nano" import create_server, register_route
from "modules/std/json/json.nano" import new_object, new_string, new_int, object_set, stringify

fn api_handler(request: string) -> string {
    let json: Json = (new_object)
    (object_set json "status" (new_string "ok"))
    (object_set json "code" (new_int 200))

    let body: string = (stringify json)

    let response: string = (+ "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" body)
    return response
    # No free() needed - automatic GC!
}

shadow api_handler {
    let response: string = (api_handler "")
    assert (str_contains response "200")
}
```

### Static File Server

```nano
from "modules/http_server/http_server.nano" import create_server, register_route
from "modules/std/fs.nano" import file_read, file_exists

fn serve_file(path: string) -> string {
    if (file_exists path) {
        let content: string = (file_read path)
        return (+ "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" content)
    }
    return "HTTP/1.1 404 Not Found\r\n\r\n404 Not Found"
}

fn file_handler(request: string) -> string {
    return (serve_file "index.html")
}

shadow file_handler {
    let response: string = (file_handler "GET /")
    assert (str_contains response "404")
}
```

## 15.3 Best Practices

### ✅ DO

**1. Handle errors gracefully:**

```nano
from "modules/curl/curl.nano" import get

fn safe_fetch(url: string) -> string {
    let response: string = (get url)
    if (== (str_length response) 0) {
        (println "Request failed")
        return ""
    }
    return response
}

shadow safe_fetch {
    assert true
}
```

**2. Set appropriate headers:**

```nano
from "modules/curl/curl.nano" import post_with_headers

fn post_json_properly(url: string, data: string) -> string {
    let headers: array<string> = [
        "Content-Type: application/json",
        "Accept: application/json",
        "User-Agent: NanoLang/1.0"
    ]
    return (post_with_headers url data headers)
}

shadow post_json_properly {
    assert true
}
```

**3. Validate URLs:**

```nano
fn is_valid_url(url: string) -> bool {
    return (or 
        (str_contains url "http://")
        (str_contains url "https://")
    )
}

shadow is_valid_url {
    assert (is_valid_url "https://example.com")
    assert (not (is_valid_url "invalid"))
}
```

### ❌ DON'T

**1. Don't hardcode URLs:**

```nano
# ❌ Bad
let response: string = (get "http://localhost:8080/api")

# ✅ Good
fn get_api_url() -> string {
    let host: string = (getenv "API_HOST")
    if (== host "") {
        return "http://localhost:8080"
    }
    return host
}
```

**2. Don't ignore status codes:**

```nano
# ❌ Bad
let response: string = (get url)
# Use response without checking

# ✅ Good
let response: string = (get url)
if (str_contains response "404") {
    (println "Resource not found")
}
```

## Summary

In this chapter, you learned:
- ✅ HTTP GET/POST requests with curl
- ✅ Custom headers for authentication
- ✅ File downloads
- ✅ HTTP server creation
- ✅ Route handling
- ✅ JSON APIs
- ✅ Static file serving

### Quick Reference

| Operation | curl | http_server |
|-----------|------|-------------|
| **GET** | `get(url)` | - |
| **POST** | `post(url, data)` | - |
| **Headers** | `get_with_headers`, `post_with_headers` | - |
| **Download** | `download(url, path)` | - |
| **Create** | - | `create_server(port)` |
| **Route** | - | `register_route(server, method, path, handler)` |
| **Start** | - | `start_server(server)` |

---

**Previous:** [Chapter 14: Data Formats](../14_data_formats/index.html)  
**Next:** [Chapter 16: Graphics Fundamentals](../16_graphics_fundamentals/index.html)
