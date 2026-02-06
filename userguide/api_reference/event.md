# event API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_event_base_new() -> int`

**Returns:** `int`


#### `extern fn nl_event_base_free(_base: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |

**Returns:** `void`


#### `extern fn nl_event_base_dispatch(_base: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |

**Returns:** `int`


#### `extern fn nl_event_base_loop(_base: int, _flags: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |
| `_flags` | `int` |

**Returns:** `int`


#### `extern fn nl_event_base_loopexit(_base: int, _timeout_secs: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |
| `_timeout_secs` | `int` |

**Returns:** `int`


#### `extern fn nl_event_base_loopbreak(_base: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |

**Returns:** `int`


#### `extern fn nl_event_base_get_method(_base: int) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |

**Returns:** `string`


#### `extern fn nl_event_base_get_num_events(_base: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |

**Returns:** `int`


#### `extern fn nl_event_get_version() -> string`

**Returns:** `string`


#### `extern fn nl_event_get_version_number() -> int`

**Returns:** `int`


#### `extern fn nl_evtimer_new(_base: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |

**Returns:** `int`


#### `extern fn nl_event_free(_event: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_event` | `int` |

**Returns:** `void`


#### `extern fn nl_evtimer_add_timeout(_event: int, _timeout_secs: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_event` | `int` |
| `_timeout_secs` | `int` |

**Returns:** `int`


#### `extern fn nl_event_del(_event: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_event` | `int` |

**Returns:** `int`


#### `extern fn nl_event_enable_debug_mode() -> void`

**Returns:** `void`


#### `extern fn nl_event_sleep(_base: int, _seconds: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `int` |
| `_seconds` | `int` |

**Returns:** `int`


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

