# http_server API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_http_server_create(_port: int) -> HttpServer`

**Parameters:**
| Name | Type |
|------|------|
| `_port` | `int` |

**Returns:** `HttpServer`


#### `extern fn nl_http_server_set_static(_server: HttpServer, _dir: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_server` | `HttpServer` |
| `_dir` | `string` |

**Returns:** `void`


#### `extern fn nl_http_server_start(_server: HttpServer) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_server` | `HttpServer` |

**Returns:** `int`


#### `extern fn nl_http_server_stop(_server: HttpServer) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_server` | `HttpServer` |

**Returns:** `void`


#### `extern fn nl_http_server_free(_server: HttpServer) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_server` | `HttpServer` |

**Returns:** `void`


#### `extern fn nl_http_server_add_route(_server: HttpServer, _method: string, _path: string, _handler_id: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_server` | `HttpServer` |
| `_method` | `string` |
| `_path` | `string` |
| `_handler_id` | `int` |

**Returns:** `int`


#### `extern fn nl_http_request_method(_request: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_request` | `HttpRequest` |

**Returns:** `string`


#### `extern fn nl_http_request_path(_request: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_request` | `HttpRequest` |

**Returns:** `string`


#### `extern fn nl_http_request_query(_request: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_request` | `HttpRequest` |

**Returns:** `string`


#### `extern fn nl_http_request_body(_request: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_request` | `HttpRequest` |

**Returns:** `string`


#### `extern fn nl_http_response_status(_response: HttpResponse, _code: int, _message: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_response` | `HttpResponse` |
| `_code` | `int` |
| `_message` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_header(_response: HttpResponse, _name: string, _value: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_response` | `HttpResponse` |
| `_name` | `string` |
| `_value` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_send_json(_response: HttpResponse, _json: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_response` | `HttpResponse` |
| `_json` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_send_html(_response: HttpResponse, _html: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_response` | `HttpResponse` |
| `_html` | `string` |

**Returns:** `void`


#### `extern fn nl_http_response_send_text(_response: HttpResponse, _text: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_response` | `HttpResponse` |
| `_text` | `string` |

**Returns:** `void`


#### `fn create(port: int) -> HttpServer`

**Parameters:**
| Name | Type |
|------|------|
| `port` | `int` |

**Returns:** `HttpServer`


#### `fn set_static_dir(server: HttpServer, dir: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `server` | `HttpServer` |
| `dir` | `string` |

**Returns:** `void`


#### `fn start(server: HttpServer) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `server` | `HttpServer` |

**Returns:** `int`


#### `fn stop(server: HttpServer) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `server` | `HttpServer` |

**Returns:** `void`


#### `fn free_server(server: HttpServer) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `server` | `HttpServer` |

**Returns:** `void`


#### `fn request_method(req: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `req` | `HttpRequest` |

**Returns:** `string`


#### `fn request_path(req: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `req` | `HttpRequest` |

**Returns:** `string`


#### `fn request_query(req: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `req` | `HttpRequest` |

**Returns:** `string`


#### `fn request_body(req: HttpRequest) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `req` | `HttpRequest` |

**Returns:** `string`


#### `fn response_status(res: HttpResponse, code: int, message: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |
| `code` | `int` |
| `message` | `string` |

**Returns:** `void`


#### `fn response_header(res: HttpResponse, name: string, value: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |
| `name` | `string` |
| `value` | `string` |

**Returns:** `void`


#### `fn send_json(res: HttpResponse, json: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |
| `json` | `string` |

**Returns:** `void`


#### `fn send_html(res: HttpResponse, html: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |
| `html` | `string` |

**Returns:** `void`


#### `fn send_text(res: HttpResponse, text: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |
| `text` | `string` |

**Returns:** `void`


#### `fn send_ok_json(res: HttpResponse, json: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |
| `json` | `string` |

**Returns:** `void`


#### `fn send_error_json(res: HttpResponse, code: int, message: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |
| `code` | `int` |
| `message` | `string` |

**Returns:** `void`


#### `fn send_not_found(res: HttpResponse) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `res` | `HttpResponse` |

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

