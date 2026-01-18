# uv API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_uv_version_string() -> string`

**Returns:** `string`


#### `extern fn nl_uv_version() -> int`

**Returns:** `int`


#### `extern fn nl_uv_default_loop() -> int`

**Returns:** `int`


#### `extern fn nl_uv_loop_new() -> int`

**Returns:** `int`


#### `extern fn nl_uv_loop_close(_loop: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |

**Returns:** `int`


#### `extern fn nl_uv_run(_loop: int, _mode: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |
| `_mode` | `int` |

**Returns:** `int`


#### `extern fn nl_uv_stop(_loop: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |

**Returns:** `void`


#### `extern fn nl_uv_loop_alive(_loop: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |

**Returns:** `int`


#### `extern fn nl_uv_loop_get_active_handles(_loop: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |

**Returns:** `int`


#### `extern fn nl_uv_now(_loop: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |

**Returns:** `int`


#### `extern fn nl_uv_update_time(_loop: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |

**Returns:** `void`


#### `extern fn nl_uv_hrtime() -> int`

**Returns:** `int`


#### `extern fn nl_uv_sleep(_msec: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_msec` | `int` |

**Returns:** `void`


#### `extern fn nl_uv_backend_timeout(_loop: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_loop` | `int` |

**Returns:** `int`


#### `extern fn nl_uv_strerror(_err: int) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_err` | `int` |

**Returns:** `string`


#### `extern fn nl_uv_err_name(_err: int) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_err` | `int` |

**Returns:** `string`


#### `extern fn nl_uv_translate_sys_error(_sys_errno: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_sys_errno` | `int` |

**Returns:** `int`


#### `extern fn nl_uv_get_total_memory() -> int`

**Returns:** `int`


#### `extern fn nl_uv_get_free_memory() -> int`

**Returns:** `int`


#### `extern fn nl_uv_cpu_count() -> int`

**Returns:** `int`


#### `extern fn nl_uv_loadavg_1min() -> int`

**Returns:** `int`


#### `extern fn nl_uv_os_getpid() -> int`

**Returns:** `int`


#### `extern fn nl_uv_os_getppid() -> int`

**Returns:** `int`


#### `extern fn nl_uv_cwd() -> string`

**Returns:** `string`


#### `extern fn nl_uv_os_gethostname() -> string`

**Returns:** `string`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

*No opaque types*

### Constants

*No constants*
