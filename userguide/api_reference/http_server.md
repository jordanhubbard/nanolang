# http_server API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_http_server_create(_port: int) -> struct<HttpServer>`

**Parameters:**

| Name | Type |
|------|------|
| `_port` | `int` |

**Returns:** `struct`


#### `extern fn nl_http_server_set_static(_server: struct<HttpServer>, _dir: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_server` | `struct<HttpServer>` |
| `_dir` | `string` |

**Returns:** `void`


#### `extern fn nl_http_server_start(_server: struct<HttpServer>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_server` | `struct<HttpServer>` |

**Returns:** `int`


#### `extern fn nl_http_server_stop(_server: struct<HttpServer>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_server` | `struct<HttpServer>` |

**Returns:** `void`


#### `extern fn nl_http_server_free(_server: struct<HttpServer>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_server` | `struct<HttpServer>` |

**Returns:** `void`


#### `extern fn nl_http_server_add_route(_server: struct<HttpServer>, _method: string, _path: string, _handler_id: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_server` | `struct<HttpServer>` |
| `_method` | `string` |
| `_path` | `string` |
| `_handler_id` | `int` |

**Returns:** `int`


#### `extern fn nl_http_request_method(_request: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_request` | `struct<HttpRequest>` |

**Returns:** `string`


#### `extern fn nl_http_request_path(_request: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_request` | `struct<HttpRequest>` |

**Returns:** `string`


#### `extern fn nl_http_request_query(_request: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_request` | `struct<HttpRequest>` |

**Returns:** `string`


#### `extern fn nl_http_request_body(_request: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_request` | `struct<HttpRequest>` |

**Returns:** `string`


#### `extern fn nl_http_response_status(_response: struct<HttpResponse>, _code: int, _message: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_response` | `struct<HttpResponse>` |
| `_code` | `int` |
| `_message` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_header(_response: struct<HttpResponse>, _name: string, _value: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_response` | `struct<HttpResponse>` |
| `_name` | `string` |
| `_value` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_send_json(_response: struct<HttpResponse>, _json: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_response` | `struct<HttpResponse>` |
| `_json` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_send_html(_response: struct<HttpResponse>, _html: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_response` | `struct<HttpResponse>` |
| `_html` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_send_text(_response: struct<HttpResponse>, _text: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_response` | `struct<HttpResponse>` |
| `_text` | `string` |

**Returns:** `void`


#### `fn create(port: int) -> struct<HttpServer>`

**Parameters:**

| Name | Type |
|------|------|
| `port` | `int` |

**Returns:** `struct`


#### `fn set_static_dir(server: struct<HttpServer>, dir: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `server` | `struct<HttpServer>` |
| `dir` | `string` |

**Returns:** `void`


#### `fn start(server: struct<HttpServer>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `server` | `struct<HttpServer>` |

**Returns:** `int`


#### `fn stop(server: struct<HttpServer>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `server` | `struct<HttpServer>` |

**Returns:** `void`


#### `fn free_server(server: struct<HttpServer>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `server` | `struct<HttpServer>` |

**Returns:** `void`


#### `fn request_method(req: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `req` | `struct<HttpRequest>` |

**Returns:** `string`


#### `fn request_path(req: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `req` | `struct<HttpRequest>` |

**Returns:** `string`


#### `fn request_query(req: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `req` | `struct<HttpRequest>` |

**Returns:** `string`


#### `fn request_body(req: struct<HttpRequest>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `req` | `struct<HttpRequest>` |

**Returns:** `string`


#### `fn response_status(res: struct<HttpResponse>, code: int, message: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |
| `code` | `int` |
| `message` | `string` |

**Returns:** `void`


#### `fn response_header(res: struct<HttpResponse>, name: string, value: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |
| `name` | `string` |
| `value` | `string` |

**Returns:** `void`


#### `fn send_json(res: struct<HttpResponse>, json: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |
| `json` | `string` |

**Returns:** `void`


#### `fn send_html(res: struct<HttpResponse>, html: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |
| `html` | `string` |

**Returns:** `void`


#### `fn send_text(res: struct<HttpResponse>, text: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |
| `text` | `string` |

**Returns:** `void`


#### `fn send_ok_json(res: struct<HttpResponse>, json: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |
| `json` | `string` |

**Returns:** `void`


#### `fn send_error_json(res: struct<HttpResponse>, code: int, message: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |
| `code` | `int` |
| `message` | `string` |

**Returns:** `void`


#### `fn send_not_found(res: struct<HttpResponse>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `res` | `struct<HttpResponse>` |

**Returns:** `void`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

- `opaque type HttpServer`
- `opaque type HttpRequest`
- `opaque type HttpResponse`

### Constants

*No constants*
