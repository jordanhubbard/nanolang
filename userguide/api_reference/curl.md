# curl API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_curl_global_init() -> int`

**Returns:** `int`


#### `extern fn nl_curl_global_cleanup() -> void`

**Returns:** `void`


#### `extern fn nl_curl_easy_init() -> int`

**Returns:** `int`


#### `extern fn nl_curl_easy_cleanup(_handle: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `void`


#### `fn curl_global_init_safe() -> int`

**Returns:** `int`


#### `fn curl_global_cleanup_safe() -> void`

**Returns:** `void`


#### `fn curl_easy_init_safe() -> int`

**Returns:** `int`


#### `fn curl_easy_cleanup_safe(handle: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `handle` | `int` |

**Returns:** `void`


#### `extern fn nl_curl_simple_get(_url: string) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_url` | `string` |

**Returns:** `string`


#### `extern fn nl_curl_simple_post(_url: string, _data: string) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_url` | `string` |
| `_data` | `string` |

**Returns:** `string`


#### `extern fn nl_curl_download_file(_url: string, _output_path: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_url` | `string` |
| `_output_path` | `string` |

**Returns:** `int`


#### `extern fn nl_curl_easy_setopt_url(_handle: int, _url: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_handle` | `int` |
| `_url` | `string` |

**Returns:** `int`


#### `extern fn nl_curl_easy_setopt_follow_location(_handle: int, _follow: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_handle` | `int` |
| `_follow` | `int` |

**Returns:** `int`


#### `extern fn nl_curl_easy_setopt_timeout(_handle: int, _timeout_secs: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_handle` | `int` |
| `_timeout_secs` | `int` |

**Returns:** `int`


#### `extern fn nl_curl_easy_setopt_useragent(_handle: int, _useragent: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_handle` | `int` |
| `_useragent` | `string` |

**Returns:** `int`


#### `extern fn nl_curl_easy_perform(_handle: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `int`


#### `extern fn nl_curl_easy_getinfo_response_code(_handle: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_handle` | `int` |

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

