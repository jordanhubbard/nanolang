# sdl_ttf API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn TTF_Init() -> int`

**Returns:** `int`


#### `extern fn TTF_Quit() -> void`

**Returns:** `void`


#### `extern fn TTF_WasInit() -> int`

**Returns:** `int`


#### `extern fn TTF_OpenFont(_file: string, _ptsize: int) -> struct<TTF_Font>`

**Parameters:**

| Name | Type |
|------|------|
| `_file` | `string` |
| `_ptsize` | `int` |

**Returns:** `struct`


#### `extern fn TTF_CloseFont(_font: struct<TTF_Font>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |

**Returns:** `void`


#### `extern fn TTF_RenderText_Solid(_font: struct<TTF_Font>, _text: string, _r: int, _g: int, _b: int, _a: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |
| `_text` | `string` |
| `_r` | `int` |
| `_g` | `int` |
| `_b` | `int` |
| `_a` | `int` |

**Returns:** `int`


#### `extern fn TTF_RenderText_Blended(_font: struct<TTF_Font>, _text: string, _r: int, _g: int, _b: int, _a: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |
| `_text` | `string` |
| `_r` | `int` |
| `_g` | `int` |
| `_b` | `int` |
| `_a` | `int` |

**Returns:** `int`


#### `extern fn TTF_RenderText_Shaded(_font: struct<TTF_Font>, _text: string, _fg_r: int, _fg_g: int, _fg_b: int, _fg_a: int, _bg_r: int, _bg_g: int, _bg_b: int, _bg_a: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |
| `_text` | `string` |
| `_fg_r` | `int` |
| `_fg_g` | `int` |
| `_fg_b` | `int` |
| `_fg_a` | `int` |
| `_bg_r` | `int` |
| `_bg_g` | `int` |
| `_bg_b` | `int` |
| `_bg_a` | `int` |

**Returns:** `int`


#### `extern fn TTF_FontHeight(_font: struct<TTF_Font>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |

**Returns:** `int`


#### `extern fn TTF_FontAscent(_font: struct<TTF_Font>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |

**Returns:** `int`


#### `extern fn TTF_FontDescent(_font: struct<TTF_Font>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |

**Returns:** `int`


#### `extern fn TTF_FontLineSkip(_font: struct<TTF_Font>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |

**Returns:** `int`


#### `extern fn TTF_SizeText(_font: struct<TTF_Font>, _text: string, _w_out: int, _h_out: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |
| `_text` | `string` |
| `_w_out` | `int` |
| `_h_out` | `int` |

**Returns:** `int`


#### `extern fn TTF_GetFontStyle(_font: struct<TTF_Font>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |

**Returns:** `int`


#### `extern fn TTF_SetFontStyle(_font: struct<TTF_Font>, _style: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_font` | `struct<TTF_Font>` |
| `_style` | `int` |

**Returns:** `void`


#### `extern fn TTF_GetError() -> string`

**Returns:** `string`


#### `extern fn TTF_ClearError() -> void`

**Returns:** `void`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

- `opaque type TTF_Font`

### Constants

*No constants*
