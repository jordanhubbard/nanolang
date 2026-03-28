# 15.2 http_server - Building Web Services

**Create HTTP servers and REST APIs in NanoLang.**

The `http_server` module provides a lightweight HTTP/1.1 server with route registration, static file serving, and structured request/response handling. Use it to expose REST APIs, serve HTML pages, or build webhooks. The server runs on a single port and dispatches incoming requests to registered handler functions based on HTTP method and path.

Three opaque types represent the core abstractions:

- `HttpServer` — the server instance, bound to a port
- `HttpRequest` — an incoming request (method, path, query string, body)
- `HttpResponse` — the response under construction (status, headers, body)

## Installation

Import the symbols you need:

```nano
from "modules/http_server/http_server.nano" import create, start, stop, free_server,
    set_static_dir,
    request_method, request_path, request_query, request_body,
    response_status, response_header,
    send_json, send_html, send_text,
    send_ok_json, send_error_json, send_not_found,
    HttpServer, HttpRequest, HttpResponse
```

## Quick Start

```nano
from "modules/http_server/http_server.nano" import create, start, stop, free_server,
    send_ok_json, HttpServer, HttpRequest, HttpResponse

fn handle_ping(req: HttpRequest, res: HttpResponse) -> void {
    (send_ok_json res "{\"status\": \"ok\"}")
    return
}

fn run() -> int {
    let server: HttpServer = (create 8080)
    # Register routes here when the routing API is available
    let rc: int = (start server)
    (stop server)
    (free_server server)
    return rc
}

shadow run { assert true }
```

---

## API Reference

### Server Lifecycle

#### `fn create(port: int) -> HttpServer`

Create an HTTP server that will listen on the given port. Does not start accepting connections until `start` is called.

| Parameter | Type | Description |
|-----------|------|-------------|
| `port` | `int` | TCP port number (1–65535; use 8080 for development) |

**Returns:** `HttpServer` — opaque server handle.

```nano
let server: HttpServer = (create 8080)
```

#### `fn start(server: HttpServer) -> int`

Start accepting connections. This call **blocks** until the server is stopped (by calling `stop` from a handler or signal handler). Returns `0` on clean shutdown.

```nano
let rc: int = (start server)
```

#### `fn stop(server: HttpServer) -> void`

Signal the server to stop accepting new connections and shut down. Typically called from within a handler function or a signal handler.

```nano
(stop server)
```

#### `fn free_server(server: HttpServer) -> void`

Release all memory and OS resources held by the server. Call after `stop` returns.

```nano
(stop server)
(free_server server)
```

#### `fn set_static_dir(server: HttpServer, dir: string) -> void`

Serve files from a local directory for paths that do not match any registered route. The server maps URL paths directly to file system paths within `dir`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `server` | `HttpServer` | The server to configure |
| `dir` | `string` | Local directory path to serve files from |

```nano
let server: HttpServer = (create 8080)
(set_static_dir server "./public")
```

---

### Route Registration (Low-Level)

#### `extern fn nl_http_server_add_route(_server: HttpServer, _method: string, _path: string, _handler_id: int) -> int`

Register a route handler. `_method` is the HTTP method string (`"GET"`, `"POST"`, `"PUT"`, `"DELETE"`, etc.). `_path` is the URL path (e.g. `"/api/users"`). `_handler_id` is a numeric identifier used to dispatch to the correct NanoLang function (handler management is done in C).

Returns `0` on success.

```nano
unsafe {
    (nl_http_server_add_route server "GET" "/api/status" 1)
    (nl_http_server_add_route server "POST" "/api/users" 2)
}
```

---

### Request Accessors

These functions extract information from an `HttpRequest`. Call them inside handler functions.

#### `fn request_method(req: HttpRequest) -> string`

Return the HTTP method of the request: `"GET"`, `"POST"`, `"PUT"`, `"DELETE"`, `"PATCH"`, etc.

#### `fn request_path(req: HttpRequest) -> string`

Return the URL path component, without the query string (e.g. `"/api/users/42"`).

#### `fn request_query(req: HttpRequest) -> string`

Return the raw query string, without the leading `?` (e.g. `"page=2&limit=20"`). Returns `""` if there is no query string.

#### `fn request_body(req: HttpRequest) -> string`

Return the raw request body as a string. For JSON APIs this will be the JSON text. Returns `""` for requests with no body (typically GET).

```nano
fn log_request(req: HttpRequest) -> void {
    let method: string = (request_method req)
    let path: string = (request_path req)
    let query: string = (request_query req)
    (println (+ method (+ " " (+ path (+ "?" query)))))
    return
}
```

---

### Response Builders

These functions build and send the HTTP response. Call them from handler functions on the `HttpResponse` object provided by the server.

#### `fn response_status(res: HttpResponse, code: int, message: string) -> void`

Set the HTTP status code and reason phrase. Must be called before sending the body. Common codes:

| Code | Message |
|------|---------|
| `200` | `"OK"` |
| `201` | `"Created"` |
| `204` | `"No Content"` |
| `400` | `"Bad Request"` |
| `401` | `"Unauthorized"` |
| `403` | `"Forbidden"` |
| `404` | `"Not Found"` |
| `500` | `"Internal Server Error"` |

```nano
(response_status res 201 "Created")
```

#### `fn response_header(res: HttpResponse, name: string, value: string) -> void`

Add a response header. Can be called multiple times for different headers.

```nano
(response_header res "Cache-Control" "no-cache")
(response_header res "X-Request-Id" "abc-123")
```

#### `fn send_json(res: HttpResponse, json: string) -> void`

Send `json` as the response body with `Content-Type: application/json`. Call `response_status` before this if you want a non-200 status.

#### `fn send_html(res: HttpResponse, html: string) -> void`

Send `html` as the response body with `Content-Type: text/html`.

#### `fn send_text(res: HttpResponse, text: string) -> void`

Send `text` as the response body with `Content-Type: text/plain`.

---

### Convenience Response Functions

These combine setting the status and sending the body in a single call.

#### `fn send_ok_json(res: HttpResponse, json: string) -> void`

Send a `200 OK` response with a JSON body. Equivalent to calling `response_status(res, 200, "OK")` then `send_json(res, json)`.

```nano
(send_ok_json res "{\"message\": \"success\"}")
```

#### `fn send_error_json(res: HttpResponse, code: int, message: string) -> void`

Send an error response with the given status `code` and a JSON body of `{"error": "<message>"}`.

```nano
(send_error_json res 400 "Invalid request body")
(send_error_json res 500 "Internal error")
```

#### `fn send_not_found(res: HttpResponse) -> void`

Send a `404 Not Found` response with an HTML body of `<h1>404 Not Found</h1>`.

```nano
(send_not_found res)
```

---

## Examples

### Example 1: Minimal Hello World Server

```nano
from "modules/http_server/http_server.nano" import create, start, stop, free_server,
    send_ok_json, HttpServer, HttpRequest, HttpResponse

fn handle_root(req: HttpRequest, res: HttpResponse) -> void {
    (send_ok_json res "{\"message\": \"Hello, World!\"}")
    return
}

fn main() -> int {
    let server: HttpServer = (create 8080)
    (println "Listening on http://localhost:8080")
    let rc: int = (start server)
    (free_server server)
    return rc
}

shadow main { assert true }
```

### Example 2: JSON REST API with Multiple Routes

```nano
from "modules/http_server/http_server.nano" import create, start, free_server,
    request_path, request_body,
    send_ok_json, send_error_json, send_not_found,
    HttpServer, HttpRequest, HttpResponse
from "modules/std/json/json.nano" import parse, free, get_string, get_int, Json

fn handle_get_user(req: HttpRequest, res: HttpResponse) -> void {
    let path: string = (request_path req)
    # In a real app, parse the user ID from the path
    let data: string = "{\"id\": 1, \"name\": \"Alice\", \"email\": \"alice@example.com\"}"
    (send_ok_json res data)
    return
}

fn handle_create_user(req: HttpRequest, res: HttpResponse) -> void {
    let body: string = (request_body req)
    if (== (str_length body) 0) {
        (send_error_json res 400 "Request body is required")
        return
    } else {
        (print "")
    }

    let obj: Json = (parse body)
    let name: string = (get_string obj "name")
    (free obj)

    if (== (str_length name) 0) {
        (send_error_json res 400 "name field is required")
        return
    } else {
        (print "")
    }

    let response: string = (+ "{\"id\": 42, \"name\": \"" (+ name "\"}"))
    (send_ok_json res response)
    return
}

shadow handle_create_user {
    assert true
}
```

### Example 3: Static File Server with API Fallback

```nano
from "modules/http_server/http_server.nano" import create, start, free_server,
    set_static_dir, request_path, send_ok_json, send_not_found,
    HttpServer, HttpRequest, HttpResponse

fn handle_api_health(req: HttpRequest, res: HttpResponse) -> void {
    (send_ok_json res "{\"status\": \"healthy\", \"version\": \"1.0.0\"}")
    return
}

fn handle_unknown(req: HttpRequest, res: HttpResponse) -> void {
    let path: string = (request_path req)
    if (str_starts_with path "/api/") {
        (send_not_found res)
    } else {
        # Static file handler will catch non-API routes
        (send_not_found res)
    }
    return
}

fn run_file_server(static_dir: string, port: int) -> int {
    let server: HttpServer = (create port)
    (set_static_dir server static_dir)
    (println (+ "Serving " (+ static_dir (+ " on port " (int_to_string port)))))
    let rc: int = (start server)
    (free_server server)
    return rc
}

shadow run_file_server { assert true }
```

### Example 4: Reading Query Parameters

```nano
from "modules/http_server/http_server.nano" import request_query, send_ok_json, send_error_json,
    HttpRequest, HttpResponse

fn handle_search(req: HttpRequest, res: HttpResponse) -> void {
    let query: string = (request_query req)
    # query might be "q=hello&limit=10"
    if (== (str_length query) 0) {
        (send_error_json res 400 "q parameter required")
        return
    } else {
        (print "")
    }

    # Build a mock response including the received query
    let resp: string = (+ "{\"query\": \"" (+ query "\", \"results\": []}"))
    (send_ok_json res resp)
    return
}

shadow handle_search { assert true }
```

### Example 5: Custom Headers and CORS

```nano
from "modules/http_server/http_server.nano" import response_status, response_header,
    send_json, HttpRequest, HttpResponse

fn handle_with_cors(req: HttpRequest, res: HttpResponse) -> void {
    # Set CORS headers
    (response_header res "Access-Control-Allow-Origin" "*")
    (response_header res "Access-Control-Allow-Methods" "GET, POST, OPTIONS")
    (response_header res "Access-Control-Allow-Headers" "Content-Type, Authorization")
    (response_header res "Cache-Control" "no-store")

    (response_status res 200 "OK")
    (send_json res "{\"data\": \"cross-origin response\"}")
    return
}

shadow handle_with_cors { assert true }
```

### Example 6: Structuring a Real Application

```nano
from "modules/http_server/http_server.nano" import create, start, free_server, set_static_dir,
    request_body, request_path,
    send_ok_json, send_error_json, send_not_found,
    HttpServer, HttpRequest, HttpResponse
from "modules/std/json/json.nano" import parse, free, get_string, new_object, new_string, new_int, object_set, stringify, Json
from "modules/sqlite/sqlite.nano" import open, close, exec_ok, prepare, step, finalize,
    bind_text, column_int, column_text, has_row, last_insert_rowid

# Application state (database handle)
let mut g_db: int = 0

fn init_db() -> void {
    set g_db (open "app.db")
    (exec_ok g_db "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, body TEXT)")
    return
}

fn api_create_note(req: HttpRequest, res: HttpResponse) -> void {
    let body: string = (request_body req)
    if (== (str_length body) 0) {
        (send_error_json res 400 "Body required")
        return
    } else {
        (print "")
    }

    let obj: Json = (parse body)
    let title: string = (get_string obj "title")
    let note_body: string = (get_string obj "body")
    (free obj)

    let stmt: int = (prepare g_db "INSERT INTO notes (title, body) VALUES (?, ?)")
    (bind_text stmt 1 title)
    (bind_text stmt 2 note_body)
    (step stmt)
    (finalize stmt)
    let new_id: int = (last_insert_rowid g_db)

    let resp_obj: Json = (new_object)
    (object_set resp_obj "id" (new_int new_id))
    (object_set resp_obj "title" (new_string title))
    let resp: string = (stringify resp_obj)
    (free resp_obj)

    (send_ok_json res resp)
    return
}

shadow api_create_note { assert true }
```

---

## Common Pitfalls

### Pitfall 1: Calling start before registering routes

`start` begins accepting connections immediately. Register all routes before calling `start`.

```nano
# WRONG — routes registered after the server starts blocking
let server: HttpServer = (create 8080)
let rc: int = (start server)   # Blocks here
# Routes below never execute

# CORRECT
let server: HttpServer = (create 8080)
(set_static_dir server "./public")
# Register routes
let rc: int = (start server)
```

### Pitfall 2: Forgetting to call free_server

`stop` signals shutdown but does not free resources. Always call `free_server` after `start` returns.

```nano
let rc: int = (start server)
(free_server server)   # Must call this
return rc
```

### Pitfall 3: Not setting status before sending body

`send_json`, `send_html`, and `send_text` do not set a status code on their own. The server defaults to `200`, but if you need a different code call `response_status` first.

```nano
# WRONG — sends 200 even though item was created
fn handle_create(req: HttpRequest, res: HttpResponse) -> void {
    (send_json res "{\"id\": 1}")
    return
}

# CORRECT
fn handle_create(req: HttpRequest, res: HttpResponse) -> void {
    (response_status res 201 "Created")
    (send_json res "{\"id\": 1}")
    return
}
```

### Pitfall 4: Parsing an empty request body

GET requests have no body. Always guard against empty body before parsing as JSON.

```nano
fn handler(req: HttpRequest, res: HttpResponse) -> void {
    let body: string = (request_body req)
    if (== (str_length body) 0) {
        (send_error_json res 400 "Body required")
        return
    } else {
        let obj: Json = (parse body)
        # ...
        (free obj)
    }
    return
}
```

### Pitfall 5: Leaking Json values in handlers

Handler functions that parse request bodies must free the `Json` objects they create, including on all error paths.

---

## Best Practices

- Use `send_ok_json`, `send_error_json`, and `send_not_found` for the most common responses — they handle status codes and content types automatically.
- Define one handler function per route for clarity.
- Keep handlers stateless where possible. Pass shared state (database handles, caches) through module-level mutable variables.
- Always guard body parsing with a length check to avoid crashes on empty bodies.
- Use `set_static_dir` for serving frontend assets instead of reading files manually in handlers.
- Add CORS headers in handlers if the API will be called from browser JavaScript.

---

**Previous:** [15.1 curl - HTTP Client](curl.html)
**Next:** [15.3 uv - Async I/O](uv.html)
