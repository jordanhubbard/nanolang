# HTTP Server Callback Design

## Goal

Enable NanoLang functions as HTTP route handlers, allowing web applications written entirely in NanoLang.

## Current State

HTTP server module exists (`modules/http_server/`) with C-based request handling.
Currently requires C callbacks - NanoLang can't register route handlers.

## Proposed Architecture

### 1. Callback Registration API

```nano
from "modules/http_server/http_server.nano" import Server, Request, Response

fn handle_hello(req: Request) -> Response {
    return Response {
        status: 200,
        body: "Hello, World!",
        content_type: "text/plain"
    }
}

fn main() -> int {
    let server: Server = (server_create 8080)
    
    # Register NanoLang callback
    (server_route server "GET" "/hello" handle_hello)
    
    (server_start server)
    return 0
}
```

### 2. Implementation Strategy

**Option A: Function Pointers (Simpler)**
- Store NanoLang function pointers in route table
- Direct C → NanoLang function calls
- Limitations: No closures, simple signatures only

**Option B: FFI Trampolines (Full-featured)**
- Generate C wrapper for each NanoLang handler
- Support complex types, closures
- More implementation work

**Recommendation**: Start with Option A for MVP.

### 3. Type Mapping

**Request struct**:
```nano
struct Request {
    method: string,
    path: string,
    body: string,
    headers: array<Header>
}

struct Header {
    name: string,
    value: string
}
```

**Response struct**:
```nano
struct Response {
    status: int,
    body: string,
    content_type: string
}
```

### 4. C Implementation

In `modules/http_server/http_server.c`:

```c
typedef nl_Response (*NanoLangRouteHandler)(nl_Request);

typedef struct Route {
    const char* method;
    const char* path;
    NanoLangRouteHandler handler;
} Route;

void server_route(Server* server, const char* method, const char* path, 
                  NanoLangRouteHandler handler) {
    // Add route to server's route table
    add_route(&server->routes, method, path, handler);
}
```

### 5. Route Matching

Use existing libuv event loop + http-parser integration.
On request:
1. Parse HTTP request → nl_Request struct
2. Match route in table
3. Call NanoLang handler
4. Convert nl_Response → HTTP response
5. Send via libuv

### 6. Example Application

**REST API**:
```nano
fn api_users_list(req: Request) -> Response {
    let users: string = "[{\"id\":1,\"name\":\"Alice\"}]"
    return Response { status: 200, body: users, content_type: "application/json" }
}

fn api_users_create(req: Request) -> Response {
    # Parse req.body, create user
    return Response { status: 201, body: "{\"id\":2}", content_type: "application/json" }
}

fn main() -> int {
    let server: Server = (server_create 3000)
    (server_route server "GET" "/api/users" api_users_list)
    (server_route server "POST" "/api/users" api_users_create)
    (server_start server)
    return 0
}
```

## Implementation Plan

1. **Phase 1**: Type definitions (Request/Response structs) - 1 day
2. **Phase 2**: C callback infrastructure - 2 days
3. **Phase 3**: NanoLang bindings - 1 day
4. **Phase 4**: Example REST API - 1 day
5. **Phase 5**: Documentation + tests - 1 day

**Total**: ~6 days

## Benefits

- Web applications entirely in NanoLang
- REST APIs without C code
- Microservices in NanoLang
- Showcase language capabilities

## Status

**Design**: ✅ Complete
**Implementation**: ⏸️  Pending (6 days estimated)
**Priority**: P2 (valuable but not critical)

This design provides clear path forward for HTTP callbacks when prioritized.
