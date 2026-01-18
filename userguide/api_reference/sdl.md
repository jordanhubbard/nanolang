# sdl API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn SDL_Init(_flags: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_flags` | `int` |

**Returns:** `int`


#### `extern fn SDL_Quit() -> void`

**Returns:** `void`


#### `extern fn SDL_GetError() -> string`

**Returns:** `string`


#### `extern fn SDL_GetTicks() -> int`

**Returns:** `int`


#### `extern fn SDL_Delay(_ms: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_ms` | `int` |

**Returns:** `void`


#### `extern fn SDL_CreateWindow(_title: string, _x: int, _y: int, _w: int, _h: int, _flags: int) -> struct<SDL_Window>`

**Parameters:**

| Name | Type |
|------|------|
| `_title` | `string` |
| `_x` | `int` |
| `_y` | `int` |
| `_w` | `int` |
| `_h` | `int` |
| `_flags` | `int` |

**Returns:** `struct`


#### `extern fn SDL_DestroyWindow(_window: struct<SDL_Window>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |

**Returns:** `void`


#### `extern fn SDL_SetWindowTitle(_window: struct<SDL_Window>, _title: string) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |
| `_title` | `string` |

**Returns:** `void`


#### `extern fn SDL_SetWindowSize(_window: struct<SDL_Window>, _w: int, _h: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `void`


#### `extern fn SDL_GetWindowSize(_window: struct<SDL_Window>, _w_ptr: int, _h_ptr: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |
| `_w_ptr` | `int` |
| `_h_ptr` | `int` |

**Returns:** `void`


#### `extern fn SDL_GL_SetAttribute(_attr: int, _value: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_attr` | `int` |
| `_value` | `int` |

**Returns:** `int`


#### `extern fn SDL_GL_CreateContext(_window: struct<SDL_Window>) -> struct<SDL_GLContext>`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |

**Returns:** `struct`


#### `extern fn SDL_GL_MakeCurrent(_window: struct<SDL_Window>, _context: struct<SDL_GLContext>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |
| `_context` | `struct<SDL_GLContext>` |

**Returns:** `int`


#### `extern fn SDL_GL_SetSwapInterval(_interval: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_interval` | `int` |

**Returns:** `int`


#### `extern fn SDL_GL_SwapWindow(_window: struct<SDL_Window>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |

**Returns:** `void`


#### `extern fn SDL_GL_DeleteContext(_context: struct<SDL_GLContext>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_context` | `struct<SDL_GLContext>` |

**Returns:** `void`


#### `extern fn SDL_CreateRenderer(_window: struct<SDL_Window>, _index: int, _flags: int) -> struct<SDL_Renderer>`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<SDL_Window>` |
| `_index` | `int` |
| `_flags` | `int` |

**Returns:** `struct`


#### `extern fn SDL_DestroyRenderer(_renderer: struct<SDL_Renderer>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |

**Returns:** `void`


#### `extern fn SDL_RenderClear(_renderer: struct<SDL_Renderer>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |

**Returns:** `int`


#### `extern fn SDL_RenderPresent(_renderer: struct<SDL_Renderer>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |

**Returns:** `void`


#### `extern fn SDL_RenderSetLogicalSize(_renderer: struct<SDL_Renderer>, _w: int, _h: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderSetScale(_renderer: struct<SDL_Renderer>, _scale_x: float, _scale_y: float) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_scale_x` | `float` |
| `_scale_y` | `float` |

**Returns:** `int`


#### `extern fn SDL_SetRenderDrawColor(_renderer: struct<SDL_Renderer>, _r: int, _g: int, _b: int, _a: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_r` | `int` |
| `_g` | `int` |
| `_b` | `int` |
| `_a` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderFillRect(_renderer: struct<SDL_Renderer>, _rect_ptr: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_rect_ptr` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderDrawPoint(_renderer: struct<SDL_Renderer>, _x: int, _y: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_x` | `int` |
| `_y` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderDrawLine(_renderer: struct<SDL_Renderer>, _x1: int, _y1: int, _x2: int, _y2: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_x1` | `int` |
| `_y1` | `int` |
| `_x2` | `int` |
| `_y2` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderDrawRect(_renderer: struct<SDL_Renderer>, _rect_ptr: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_rect_ptr` | `int` |

**Returns:** `int`


#### `extern fn SDL_SetRenderDrawBlendMode(_renderer: struct<SDL_Renderer>, _mode: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_mode` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderCopy(_renderer: struct<SDL_Renderer>, _texture: struct<SDL_Texture>, _srcrect: int, _dstrect: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_texture` | `struct<SDL_Texture>` |
| `_srcrect` | `int` |
| `_dstrect` | `int` |

**Returns:** `int`


#### `extern fn SDL_CreateTexture(_renderer: struct<SDL_Renderer>, _format: int, _access: int, _w: int, _h: int) -> struct<SDL_Texture>`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_format` | `int` |
| `_access` | `int` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `struct`


#### `extern fn SDL_UpdateTexture(_texture: struct<SDL_Texture>, _rect: int, _pixels: int, _pitch: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_texture` | `struct<SDL_Texture>` |
| `_rect` | `int` |
| `_pixels` | `int` |
| `_pitch` | `int` |

**Returns:** `int`


#### `extern fn SDL_SetTextureBlendMode(_texture: struct<SDL_Texture>, _mode: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_texture` | `struct<SDL_Texture>` |
| `_mode` | `int` |

**Returns:** `int`


#### `extern fn SDL_SetTextureAlphaMod(_texture: struct<SDL_Texture>, _alpha: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_texture` | `struct<SDL_Texture>` |
| `_alpha` | `int` |

**Returns:** `int`


#### `extern fn SDL_QueryTexture(_texture: struct<SDL_Texture>, _format: int, _access: int, _w: int, _h: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_texture` | `struct<SDL_Texture>` |
| `_format` | `int` |
| `_access` | `int` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `int`


#### `extern fn SDL_CreateTextureFromSurface(_renderer: struct<SDL_Renderer>, _surface: struct<SDL_Surface>) -> struct<SDL_Texture>`

**Parameters:**

| Name | Type |
|------|------|
| `_renderer` | `struct<SDL_Renderer>` |
| `_surface` | `struct<SDL_Surface>` |

**Returns:** `struct`


#### `extern fn SDL_DestroyTexture(_texture: struct<SDL_Texture>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_texture` | `struct<SDL_Texture>` |

**Returns:** `void`


#### `extern fn SDL_FreeSurface(_surface: struct<SDL_Surface>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_surface` | `struct<SDL_Surface>` |

**Returns:** `void`


#### `extern fn SDL_PollEvent(_event_ptr: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_event_ptr` | `int` |

**Returns:** `int`


#### `extern fn SDL_EventState(_type: int, _state: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_type` | `int` |
| `_state` | `int` |

**Returns:** `int`


#### `extern fn SDL_SetHint(_name: string, _value: string) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_name` | `string` |
| `_value` | `string` |

**Returns:** `int`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

- `opaque type SDL_Window`
- `opaque type SDL_Renderer`
- `opaque type SDL_Texture`
- `opaque type SDL_Surface`
- `opaque type SDL_GLContext`

### Constants

*No constants*
