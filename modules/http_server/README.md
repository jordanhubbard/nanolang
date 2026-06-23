# HTTP Server Module

High-performance HTTP/1.1 server for NanoLang built on libuv.

## Features

- **Async I/O**: Built on libuv event loop for high concurrency
- **Static File Serving**: Serve files from a directory with automatic MIME types
- **Routing**: Register handlers for different HTTP methods and paths
- **REST API Support**: Easy JSON response building
- **Security**: Built-in directory traversal protection
- **HTTP/1.1**: Full HTTP/1.1 support with proper headers

## Installation

The module requires libuv to be installed:

```bash
# macOS
brew install libuv

# Ubuntu/Debian
sudo apt-get install libuv1-dev

# Fedora/RHEL
sudo dnf install libuv-devel
```

## Quick Start

### Static File Server

```nano
import "modules/http_server/http_server.nano" as Server

fn main() -> int {
    let server: Server.HttpServer = (Server.create 8080)
    (Server.set_static_dir server "./public")
    (println "Static file server running on http://localhost:8080")
    (Server.start server)
    return 0
}
```

### Simple REST API

```nano
import "modules/http_server/http_server.nano" as Server

fn main() -> int {
    let server: Server.HttpServer = (Server.create 8080)
    
    # Note: Route handlers are currently managed in C
    # Full NanoLang handler support coming soon
    
    (println "API server running on http://localhost:8080")
    (Server.start server)
    return 0
}
```

## API Reference

### Server Management

- `create(port: int) -> HttpServer` - Create server on port
- `set_static_dir(server, dir: string)` - Enable static file serving
- `start(server) -> int` - Start server (blocks)
- `stop(server)` - Stop server
- `free_server(server)` - Free resources

### Request Accessors

- `request_method(req) -> string` - Get HTTP method
- `request_path(req) -> string` - Get request path
- `request_query(req) -> string` - Get query string
- `request_body(req) -> string` - Get request body

### Response Builders

- `response_status(res, code: int, message: string)` - Set status
- `response_header(res, name: string, value: string)` - Add header
- `send_json(res, json: string)` - Send JSON response
- `send_html(res, html: string)` - Send HTML response
- `send_text(res, text: string)` - Send text response

### Convenience Functions

- `send_ok_json(res, json: string)` - Send 200 OK with JSON
- `send_error_json(res, code: int, message: string)` - Send error JSON
- `send_not_found(res)` - Send 404 response

## Supported MIME Types

The server automatically detects MIME types for:

- HTML: `.html`, `.htm`
- CSS: `.css`
- JavaScript: `.js`
- JSON: `.json`
- Images: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.ico`
- Documents: `.txt`, `.xml`, `.pdf`

## Security

- **Directory Traversal Protection**: Automatically blocks `..` in paths
- **Connection Limits**: Configurable connection backlog
- **Request Size Limits**: 64KB maximum request size

## Performance

- **Async I/O**: Non-blocking operations via libuv
- **Connection Pooling**: Efficient client connection management
- **Zero-Copy**: Minimal memory allocations

## Roadmap

- [ ] Full NanoLang route handler support (currently C-side only)
- [ ] Middleware system
- [ ] WebSocket support
- [ ] HTTPS/TLS support
- [ ] Request body parsing (JSON, form data)
- [ ] Cookie support
- [ ] Session management
- [ ] Compression (gzip, deflate)
- [ ] Rate limiting
- [ ] CORS support

## Examples

See `examples/http_server_demo.nano` and `examples/rest_api_demo.nano` for complete examples.

## License

MIT

