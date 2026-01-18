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


#### `extern fn SDL_CreateWindow(_title: string, _x: int, _y: int, _w: int, _h: int, _flags: int) -> SDL_Window`

**Parameters:**
| Name | Type |
|------|------|
| `_title` | `string` |
| `_x` | `int` |
| `_y` | `int` |
| `_w` | `int` |
| `_h` | `int` |
| `_flags` | `int` |

**Returns:** `SDL_Window`


#### `extern fn SDL_DestroyWindow(_window: SDL_Window) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |

**Returns:** `void`


#### `extern fn SDL_SetWindowTitle(_window: SDL_Window, _title: string) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |
| `_title` | `string` |

**Returns:** `void`


#### `extern fn SDL_SetWindowSize(_window: SDL_Window, _w: int, _h: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `void`


#### `extern fn SDL_GetWindowSize(_window: SDL_Window, _w_ptr: int, _h_ptr: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |
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


#### `extern fn SDL_GL_CreateContext(_window: SDL_Window) -> SDL_GLContext`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |

**Returns:** `SDL_GLContext`


#### `extern fn SDL_GL_MakeCurrent(_window: SDL_Window, _context: SDL_GLContext) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |
| `_context` | `SDL_GLContext` |

**Returns:** `int`


#### `extern fn SDL_GL_SetSwapInterval(_interval: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_interval` | `int` |

**Returns:** `int`


#### `extern fn SDL_GL_SwapWindow(_window: SDL_Window) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |

**Returns:** `void`


#### `extern fn SDL_GL_DeleteContext(_context: SDL_GLContext) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_context` | `SDL_GLContext` |

**Returns:** `void`


#### `extern fn SDL_CreateRenderer(_window: SDL_Window, _index: int, _flags: int) -> SDL_Renderer`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `SDL_Window` |
| `_index` | `int` |
| `_flags` | `int` |

**Returns:** `SDL_Renderer`


#### `extern fn SDL_DestroyRenderer(_renderer: SDL_Renderer) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |

**Returns:** `void`


#### `extern fn SDL_RenderClear(_renderer: SDL_Renderer) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |

**Returns:** `int`


#### `extern fn SDL_RenderPresent(_renderer: SDL_Renderer) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |

**Returns:** `void`


#### `extern fn SDL_RenderSetLogicalSize(_renderer: SDL_Renderer, _w: int, _h: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderSetScale(_renderer: SDL_Renderer, _scale_x: float, _scale_y: float) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_scale_x` | `float` |
| `_scale_y` | `float` |

**Returns:** `int`


#### `extern fn SDL_SetRenderDrawColor(_renderer: SDL_Renderer, _r: int, _g: int, _b: int, _a: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_r` | `int` |
| `_g` | `int` |
| `_b` | `int` |
| `_a` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderFillRect(_renderer: SDL_Renderer, _rect_ptr: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_rect_ptr` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderDrawPoint(_renderer: SDL_Renderer, _x: int, _y: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_x` | `int` |
| `_y` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderDrawLine(_renderer: SDL_Renderer, _x1: int, _y1: int, _x2: int, _y2: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_x1` | `int` |
| `_y1` | `int` |
| `_x2` | `int` |
| `_y2` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderDrawRect(_renderer: SDL_Renderer, _rect_ptr: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_rect_ptr` | `int` |

**Returns:** `int`


#### `extern fn SDL_SetRenderDrawBlendMode(_renderer: SDL_Renderer, _mode: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_mode` | `int` |

**Returns:** `int`


#### `extern fn SDL_RenderCopy(_renderer: SDL_Renderer, _texture: SDL_Texture, _srcrect: int, _dstrect: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_texture` | `SDL_Texture` |
| `_srcrect` | `int` |
| `_dstrect` | `int` |

**Returns:** `int`


#### `extern fn SDL_CreateTexture(_renderer: SDL_Renderer, _format: int, _access: int, _w: int, _h: int) -> SDL_Texture`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_format` | `int` |
| `_access` | `int` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `SDL_Texture`


#### `extern fn SDL_UpdateTexture(_texture: SDL_Texture, _rect: int, _pixels: int, _pitch: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `SDL_Texture` |
| `_rect` | `int` |
| `_pixels` | `int` |
| `_pitch` | `int` |

**Returns:** `int`


#### `extern fn SDL_SetTextureBlendMode(_texture: SDL_Texture, _mode: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `SDL_Texture` |
| `_mode` | `int` |

**Returns:** `int`


#### `extern fn SDL_SetTextureAlphaMod(_texture: SDL_Texture, _alpha: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `SDL_Texture` |
| `_alpha` | `int` |

**Returns:** `int`


#### `extern fn SDL_QueryTexture(_texture: SDL_Texture, _format: int, _access: int, _w: int, _h: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `SDL_Texture` |
| `_format` | `int` |
| `_access` | `int` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `int`


#### `extern fn SDL_CreateTextureFromSurface(_renderer: SDL_Renderer, _surface: SDL_Surface) -> SDL_Texture`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_surface` | `SDL_Surface` |

**Returns:** `SDL_Texture`


#### `extern fn SDL_DestroyTexture(_texture: SDL_Texture) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `SDL_Texture` |

**Returns:** `void`


#### `extern fn SDL_FreeSurface(_surface: SDL_Surface) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_surface` | `SDL_Surface` |

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

| Name | Type | Value |
|------|------|-------|
| `SDL_INIT_VIDEO` | `int` | `32` |
| `SDL_INIT_AUDIO` | `int` | `16` |
| `SDL_INIT_TIMER` | `int` | `1` |
| `SDL_INIT_EVERYTHING` | `int` | `62977` |
| `SDL_WINDOWPOS_UNDEFINED` | `int` | `536805376` |
| `SDL_WINDOWPOS_CENTERED` | `int` | `805240832` |
| `SDL_WINDOW_SHOWN` | `int` | `4` |
| `SDL_WINDOW_FULLSCREEN` | `int` | `1` |
| `SDL_WINDOW_FULLSCREEN_DESKTOP` | `int` | `4097` |
| `SDL_WINDOW_RESIZABLE` | `int` | `32` |
| `SDL_WINDOW_OPENGL` | `int` | `2` |
| `SDL_RENDERER_SOFTWARE` | `int` | `1` |
| `SDL_RENDERER_ACCELERATED` | `int` | `2` |
| `SDL_RENDERER_PRESENTVSYNC` | `int` | `4` |
| `SDL_BLENDMODE_NONE` | `int` | `0` |
| `SDL_BLENDMODE_BLEND` | `int` | `1` |
| `SDL_BLENDMODE_ADD` | `int` | `2` |
| `SDL_BLENDMODE_MOD` | `int` | `4` |
| `SDL_GL_RED_SIZE` | `int` | `0` |
| `SDL_GL_GREEN_SIZE` | `int` | `1` |
| `SDL_GL_BLUE_SIZE` | `int` | `2` |
| `SDL_GL_ALPHA_SIZE` | `int` | `3` |
| `SDL_GL_BUFFER_SIZE` | `int` | `4` |
| `SDL_GL_DOUBLEBUFFER` | `int` | `5` |
| `SDL_GL_DEPTH_SIZE` | `int` | `6` |
| `SDL_GL_STENCIL_SIZE` | `int` | `7` |
| `SDL_GL_CONTEXT_MAJOR_VERSION` | `int` | `17` |
| `SDL_GL_CONTEXT_MINOR_VERSION` | `int` | `18` |
| `SDL_GL_CONTEXT_PROFILE_MASK` | `int` | `21` |
| `SDL_GL_CONTEXT_PROFILE_CORE` | `int` | `1` |
| `SDL_GL_CONTEXT_PROFILE_COMPATIBILITY` | `int` | `2` |

