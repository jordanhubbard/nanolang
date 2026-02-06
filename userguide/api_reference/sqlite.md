# sqlite API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_sqlite3_version() -> string`

**Returns:** `string`


#### `extern fn nl_sqlite3_version_number() -> int`

**Returns:** `int`


#### `extern fn nl_sqlite3_open(_filename: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_filename` | `string` |

**Returns:** `int`


#### `extern fn nl_sqlite3_close(_db: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_errmsg(_db: int) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |

**Returns:** `string`


#### `extern fn nl_sqlite3_exec(_db: int, _sql: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |
| `_sql` | `string` |

**Returns:** `int`


#### `extern fn nl_sqlite3_prepare(_db: int, _sql: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |
| `_sql` | `string` |

**Returns:** `int`


#### `extern fn nl_sqlite3_finalize(_stmt: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_step(_stmt: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_reset(_stmt: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_bind_int(_stmt: int, _index: int, _value: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |
| `_value` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_bind_double(_stmt: int, _index: int, _value: float) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |
| `_value` | `float` |

**Returns:** `int`


#### `extern fn nl_sqlite3_bind_text(_stmt: int, _index: int, _value: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |
| `_value` | `string` |

**Returns:** `int`


#### `extern fn nl_sqlite3_bind_null(_stmt: int, _index: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_column_count(_stmt: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_column_name(_stmt: int, _index: int) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |

**Returns:** `string`


#### `extern fn nl_sqlite3_column_int(_stmt: int, _index: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_column_double(_stmt: int, _index: int) -> float`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |

**Returns:** `float`


#### `extern fn nl_sqlite3_column_text(_stmt: int, _index: int) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |

**Returns:** `string`


#### `extern fn nl_sqlite3_column_type(_stmt: int, _index: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_stmt` | `int` |
| `_index` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_last_insert_rowid(_db: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_changes(_db: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_begin_transaction(_db: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_commit(_db: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |

**Returns:** `int`


#### `extern fn nl_sqlite3_rollback(_db: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_db` | `int` |

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

