# filesystem API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_fs_list_files(_path: string, _extension: string) -> array`

**Parameters:**

| Name | Type |
|------|------|
| `_path` | `string` |
| `_extension` | `string` |

**Returns:** `array`


#### `extern fn nl_fs_list_files_ci(_path: string, _extension: string) -> array`

**Parameters:**

| Name | Type |
|------|------|
| `_path` | `string` |
| `_extension` | `string` |

**Returns:** `array`


#### `extern fn nl_fs_list_dirs(_path: string) -> array`

**Parameters:**

| Name | Type |
|------|------|
| `_path` | `string` |

**Returns:** `array`


#### `extern fn nl_fs_parent_dir(_path: string) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_path` | `string` |

**Returns:** `string`


#### `extern fn nl_fs_is_directory(_path: string) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_path` | `string` |

**Returns:** `int`


#### `extern fn nl_fs_file_exists(_path: string) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_path` | `string` |

**Returns:** `int`


#### `extern fn nl_fs_file_size(_path: string) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_path` | `string` |

**Returns:** `int`


#### `extern fn nl_fs_join_path(_dir: string, _filename: string) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_dir` | `string` |
| `_filename` | `string` |

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
