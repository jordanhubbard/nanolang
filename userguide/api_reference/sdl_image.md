# sdl_image API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn IMG_Init(_flags: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_flags` | `int` |

**Returns:** `int`


#### `extern fn IMG_Quit() -> void`

**Returns:** `void`


#### `extern fn IMG_Linked_Version() -> int`

**Returns:** `int`


#### `extern fn IMG_Load(_file: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_file` | `string` |

**Returns:** `int`


#### `extern fn IMG_LoadTexture(_renderer: SDL_Renderer, _file: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_file` | `string` |

**Returns:** `int`


#### `extern fn IMG_Load_RW(_src: int, _freesrc: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |
| `_freesrc` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadPNG_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadJPG_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadBMP_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadGIF_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadTGA_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadPCX_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadTIF_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadWEBP_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadXPM_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadXV_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadTexture_RW(_renderer: SDL_Renderer, _src: int, _freesrc: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_src` | `int` |
| `_freesrc` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadTextureTyped_RW(_renderer: SDL_Renderer, _src: int, _freesrc: int, _type: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_src` | `int` |
| `_freesrc` | `int` |
| `_type` | `string` |

**Returns:** `int`


#### `extern fn IMG_isPNG(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isJPG(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isBMP(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isGIF(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isTIF(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isPCX(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isPNM(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isSVG(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isTGA(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isWEBP(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isXPM(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isXV(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isICO(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_isCUR(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_SavePNG(_surface: int, _file: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_surface` | `int` |
| `_file` | `string` |

**Returns:** `int`


#### `extern fn IMG_SavePNG_RW(_surface: int, _dst: int, _freedst: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_surface` | `int` |
| `_dst` | `int` |
| `_freedst` | `int` |

**Returns:** `int`


#### `extern fn IMG_SaveJPG(_surface: int, _file: string, _quality: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_surface` | `int` |
| `_file` | `string` |
| `_quality` | `int` |

**Returns:** `int`


#### `extern fn IMG_SaveJPG_RW(_surface: int, _dst: int, _freedst: int, _quality: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_surface` | `int` |
| `_dst` | `int` |
| `_freedst` | `int` |
| `_quality` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadAnimation(_file: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_file` | `string` |

**Returns:** `int`


#### `extern fn IMG_LoadAnimation_RW(_src: int, _freesrc: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |
| `_freesrc` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadAnimationTyped_RW(_src: int, _freesrc: int, _type: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |
| `_freesrc` | `int` |
| `_type` | `string` |

**Returns:** `int`


#### `extern fn IMG_LoadGIFAnimation_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_LoadWEBPAnimation_RW(_src: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_src` | `int` |

**Returns:** `int`


#### `extern fn IMG_FreeAnimation(_anim: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_anim` | `int` |

**Returns:** `void`


#### `extern fn IMG_GetError() -> string`

**Returns:** `string`


#### `extern fn IMG_SetError(_fmt: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_fmt` | `string` |

**Returns:** `int`


#### `extern fn nl_img_load_png_texture(_renderer: SDL_Renderer, _file: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_file` | `string` |

**Returns:** `int`


#### `extern fn nl_img_load_texture(_renderer: SDL_Renderer, _file: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_file` | `string` |

**Returns:** `int`


#### `extern fn nl_img_render_texture(_renderer: SDL_Renderer, _texture: int, _x: int, _y: int, _w: int, _h: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_texture` | `int` |
| `_x` | `int` |
| `_y` | `int` |
| `_w` | `int` |
| `_h` | `int` |

**Returns:** `int`


#### `extern fn nl_img_render_texture_ex(_renderer: SDL_Renderer, _texture: int, _x: int, _y: int, _w: int, _h: int, _angle: float, _flip: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_texture` | `int` |
| `_x` | `int` |
| `_y` | `int` |
| `_w` | `int` |
| `_h` | `int` |
| `_angle` | `float` |
| `_flip` | `int` |

**Returns:** `int`


#### `extern fn nl_img_render_texture_sprite(_renderer: SDL_Renderer, _texture: int, _src_x: int, _src_y: int, _src_w: int, _src_h: int, _dst_x: int, _dst_y: int, _dst_w: int, _dst_h: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_texture` | `int` |
| `_src_x` | `int` |
| `_src_y` | `int` |
| `_src_w` | `int` |
| `_src_h` | `int` |
| `_dst_x` | `int` |
| `_dst_y` | `int` |
| `_dst_w` | `int` |
| `_dst_h` | `int` |

**Returns:** `int`


#### `extern fn nl_img_get_texture_size(_texture: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `int` |

**Returns:** `int`


#### `extern fn nl_img_get_texture_width(_texture: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `int` |

**Returns:** `int`


#### `extern fn nl_img_get_texture_height(_texture: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `int` |

**Returns:** `int`


#### `extern fn nl_img_set_texture_alpha(_texture: int, _alpha: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `int` |
| `_alpha` | `int` |

**Returns:** `int`


#### `extern fn nl_img_set_texture_color(_texture: int, _r: int, _g: int, _b: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `int` |
| `_r` | `int` |
| `_g` | `int` |
| `_b` | `int` |

**Returns:** `int`


#### `extern fn nl_img_set_texture_blend_mode(_texture: int, _blend: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `int` |
| `_blend` | `int` |

**Returns:** `int`


#### `extern fn nl_img_create_texture_from_pixels(_renderer: SDL_Renderer, _width: int, _height: int, _pixels: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_width` | `int` |
| `_height` | `int` |
| `_pixels` | `int` |

**Returns:** `int`


#### `extern fn nl_img_load_icon_batch(_renderer: SDL_Renderer, _files: array<string>, _count: int) -> array<int>`

**Parameters:**
| Name | Type |
|------|------|
| `_renderer` | `SDL_Renderer` |
| `_files` | `array<string>` |
| `_count` | `int` |

**Returns:** `array<int>`


#### `extern fn nl_img_destroy_texture(_texture: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_texture` | `int` |

**Returns:** `void`


#### `extern fn nl_img_destroy_texture_batch(_textures: array<int>, _count: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_textures` | `array<int>` |
| `_count` | `int` |

**Returns:** `void`


#### `extern fn nl_img_can_load(_file: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_file` | `string` |

**Returns:** `int`


#### `extern fn nl_img_get_supported_formats() -> array<string>`

**Returns:** `array<string>`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

*No opaque types*

### Constants

| Name | Type | Value |
|------|------|-------|
| `IMG_INIT_JPG` | `int` | `1` |
| `IMG_INIT_PNG` | `int` | `2` |
| `IMG_INIT_TIF` | `int` | `4` |
| `IMG_INIT_WEBP` | `int` | `8` |
| `IMG_INIT_JXL` | `int` | `16` |
| `IMG_INIT_AVIF` | `int` | `32` |
| `IMG_INIT_ALL` | `int` | `63` |
| `SDL_BLENDMODE_NONE` | `int` | `0` |
| `SDL_BLENDMODE_BLEND` | `int` | `1` |
| `SDL_BLENDMODE_ADD` | `int` | `2` |
| `SDL_BLENDMODE_MOD` | `int` | `4` |
| `SDL_FLIP_NONE` | `int` | `0` |
| `SDL_FLIP_HORIZONTAL` | `int` | `1` |
| `SDL_FLIP_VERTICAL` | `int` | `2` |
| `SDL_FLIP_BOTH` | `int` | `3` |

