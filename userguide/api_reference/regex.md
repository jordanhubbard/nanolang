# regex API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_regex_compile(pattern: string) -> Regex`

**Parameters:**
| Name | Type |
|------|------|
| `pattern` | `string` |

**Returns:** `Regex`


#### `extern fn nl_regex_match(regex: Regex, text: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `int`


#### `extern fn nl_regex_find(regex: Regex, text: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `int`


#### `extern fn nl_regex_find_all(regex: Regex, text: string) -> array<int>`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `array<int>`


#### `extern fn nl_regex_groups(regex: Regex, text: string) -> array<string>`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `array<string>`


#### `extern fn nl_regex_replace(regex: Regex, text: string, replacement: string) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |
| `replacement` | `string` |

**Returns:** `string`


#### `extern fn nl_regex_replace_all(regex: Regex, text: string, replacement: string) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |
| `replacement` | `string` |

**Returns:** `string`


#### `extern fn nl_regex_split(regex: Regex, text: string) -> array<string>`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `array<string>`


#### `extern fn nl_regex_free(regex: Regex) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |

**Returns:** `void`


#### `fn compile(pattern: string) -> Regex`

**Parameters:**
| Name | Type |
|------|------|
| `pattern` | `string` |

**Returns:** `Regex`


#### `fn matches(regex: Regex, text: string) -> bool`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `bool`


#### `fn find(regex: Regex, text: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `int`


#### `fn find_all(regex: Regex, text: string) -> array<int>`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `array<int>`


#### `fn groups(regex: Regex, text: string) -> array<string>`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `array<string>`


#### `fn replace(regex: Regex, text: string, replacement: string) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |
| `replacement` | `string` |

**Returns:** `string`


#### `fn replace_all(regex: Regex, text: string, replacement: string) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |
| `replacement` | `string` |

**Returns:** `string`


#### `fn split(regex: Regex, text: string) -> array<string>`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `array<string>`


#### `fn quick_match(pattern: string, text: string) -> bool`

**Parameters:**
| Name | Type |
|------|------|
| `pattern` | `string` |
| `text` | `string` |

**Returns:** `bool`


#### `fn quick_find(pattern: string, text: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `pattern` | `string` |
| `text` | `string` |

**Returns:** `int`


#### `fn quick_replace(pattern: string, text: string, replacement: string) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `pattern` | `string` |
| `text` | `string` |
| `replacement` | `string` |

**Returns:** `string`


#### `fn quick_split(pattern: string, text: string) -> array<string>`

**Parameters:**
| Name | Type |
|------|------|
| `pattern` | `string` |
| `text` | `string` |

**Returns:** `array<string>`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

- `opaque type Regex`

### Constants

*No constants*

