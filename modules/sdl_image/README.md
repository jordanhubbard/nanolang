# SDL_image Module - Complete API

Load PNG, JPG, GIF, WEBP, TIFF, and other image formats into SDL textures for rendering in NanoLang programs.

## Features

- **Complete SDL_image 2.x API** - All functions wrapped
- **15+ Image Formats** - PNG, JPEG, BMP, GIF, WEBP, TIFF, TGA, PCX, PNM, XPM, XV, ICO, CUR, SVG
- **Advanced Rendering** - Rotation, flipping, scaling, color modulation, alpha blending
- **Sprite Sheets** - Render portions of textures for animations
- **Batch Operations** - Load multiple images efficiently
- **Animation Support** - Animated GIF and WEBP
- **Save Functions** - Export surfaces as PNG or JPEG
- **Helper Functions** - Convenient wrappers for common operations

## Installation

SDL_image will be automatically installed when you compile a program that uses this module.

**macOS:**
```bash
brew install sdl2_image
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install libsdl2-image-dev
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install SDL2_image-devel
```

## Quick Start

```nano
import "modules/sdl/sdl.nano"
import "modules/sdl_image/sdl_image.nano"

fn main() -> int {
    unsafe { (SDL_Init 32) }
    unsafe { (IMG_Init IMG_INIT_ALL) }
    
    let window: SDL_Window = (SDL_CreateWindow "Image Demo" 100 100 800 600 4)
    let renderer: SDL_Renderer = (SDL_CreateRenderer window -1 2)
    
    # Load an image
    let texture: int = (nl_img_load_texture renderer "icon.png")
    
    # Main loop
    let mut running: bool = true
    while running {
        if (== (nl_sdl_poll_event_quit) 1) {
            set running false
        } else {}
        
        unsafe { (SDL_RenderClear renderer) }
        (nl_img_render_texture renderer texture 100 100 200 200)
        unsafe { (SDL_RenderPresent renderer) }
        unsafe { (SDL_Delay 16) }
    }
    
    (nl_img_destroy_texture texture)
    unsafe { (IMG_Quit) }
    unsafe { (SDL_Quit) }
    return 0
}
```

## API Reference

### Initialization

```nano
# Initialize with format support
let flags: int = (IMG_Init IMG_INIT_ALL)  # All formats
let flags: int = (IMG_Init (+ IMG_INIT_PNG IMG_INIT_JPG))  # PNG + JPEG only

# Cleanup
(IMG_Quit)
```

**Constants:**
- `IMG_INIT_JPG` = 1 - JPEG support
- `IMG_INIT_PNG` = 2 - PNG support
- `IMG_INIT_TIF` = 4 - TIFF support
- `IMG_INIT_WEBP` = 8 - WEBP support
- `IMG_INIT_JXL` = 16 - JPEG XL support (SDL_image 2.6+)
- `IMG_INIT_AVIF` = 32 - AVIF support (SDL_image 2.6+)
- `IMG_INIT_ALL` = 63 - All formats

### Loading Images

```nano
# Simple loading (auto-detects format)
let tex: int = (nl_img_load_texture renderer "image.png")
let tex: int = (nl_img_load_png_texture renderer "icon.png")

# SDL_image core functions
let tex: int = (IMG_LoadTexture renderer "image.jpg")
let surface: int = (IMG_Load "image.bmp")

# Check if successful
if (== tex 0) {
    let err: string = (IMG_GetError)
    (println (+ "Failed: " err))
} else {}
```

### Basic Rendering

```nano
# Render at position with original size
(nl_img_render_texture renderer texture 100 100 0 0)

# Render with custom size
(nl_img_render_texture renderer texture 100 100 200 200)
```

### Advanced Rendering

```nano
# Rotation and flipping
let angle: float = 45.0
(nl_img_render_texture_ex renderer texture 300 200 100 100 angle SDL_FLIP_NONE)
(nl_img_render_texture_ex renderer texture 300 200 100 100 0.0 SDL_FLIP_HORIZONTAL)
(nl_img_render_texture_ex renderer texture 300 200 100 100 0.0 SDL_FLIP_VERTICAL)
(nl_img_render_texture_ex renderer texture 300 200 100 100 45.0 SDL_FLIP_BOTH)
```

**Flip Constants:**
- `SDL_FLIP_NONE` = 0 - No flipping
- `SDL_FLIP_HORIZONTAL` = 1 - Flip horizontally
- `SDL_FLIP_VERTICAL` = 2 - Flip vertically
- `SDL_FLIP_BOTH` = 3 - Flip both directions

### Sprite Sheets

```nano
# Render a portion of a texture (sprite sheet)
# nl_img_render_texture_sprite(renderer, texture, src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h)

# Extract 64x64 sprite at (0, 0) and render at (100, 100) scaled to 128x128
(nl_img_render_texture_sprite renderer sprite_sheet 0 0 64 64 100 100 128 128)

# Animation frame example
let frame: int = 2
let frame_x: int = (* frame 64)
(nl_img_render_texture_sprite renderer sprite_sheet frame_x 0 64 64 200 200 64 64)
```

### Texture Properties

```nano
# Get dimensions
let width: int = (nl_img_get_texture_width texture)
let height: int = (nl_img_get_texture_height texture)
let size: int = (nl_img_get_texture_size texture)  # Packed: width in high 32 bits

# Alpha transparency (0 = transparent, 255 = opaque)
(nl_img_set_texture_alpha texture 128)  # 50% transparent

# Color modulation (tinting)
(nl_img_set_texture_color texture 255 100 100)  # Red tint

# Blend mode
(nl_img_set_texture_blend_mode texture SDL_BLENDMODE_BLEND)
```

**Blend Mode Constants:**
- `SDL_BLENDMODE_NONE` = 0 - No blending
- `SDL_BLENDMODE_BLEND` = 1 - Alpha blending (default)
- `SDL_BLENDMODE_ADD` = 2 - Additive blending (lights)
- `SDL_BLENDMODE_MOD` = 4 - Multiplicative blending (shadows)

### Format Detection

```nano
# Check if file is a specific format (requires RWops - advanced)
extern fn IMG_isPNG(src: int) -> int
extern fn IMG_isJPG(src: int) -> int
extern fn IMG_isBMP(src: int) -> int
extern fn IMG_isGIF(src: int) -> int
extern fn IMG_isWEBP(src: int) -> int
extern fn IMG_isTIF(src: int) -> int
# ... and more formats

# Helper to check if file is loadable
if (== (nl_img_can_load "image.png") 1) {
    (println "File is loadable")
} else {}
```

### Saving Images

```nano
# Save surface as PNG
(IMG_SavePNG surface "output.png")

# Save surface as JPEG with quality (0-100)
(IMG_SaveJPG surface "output.jpg" 90)
```

### Batch Operations

```nano
# Load multiple images at once
let files: array<string> = (array_new 3 "")
(array_set files 0 "icon1.png")
(array_set files 1 "icon2.png")
(array_set files 2 "icon3.png")

let textures: array<int> = (nl_img_load_icon_batch renderer files 3)

# Cleanup multiple textures
(nl_img_destroy_texture_batch textures 3)
```

### Cleanup

```nano
# Destroy single texture
(nl_img_destroy_texture texture)

# Destroy multiple textures
(nl_img_destroy_texture tex1)
(nl_img_destroy_texture tex2)
(nl_img_destroy_texture tex3)

# Or use batch destroy
(nl_img_destroy_texture_batch textures count)

# Shutdown SDL_image
unsafe { (IMG_Quit) }
```

## Complete Example - All Features

See `examples/sdl_image_demo.nano` for a comprehensive demonstration including:
- Basic rendering
- Scaling
- Rotation
- Flipping
- Alpha fading
- Color tinting
- Combined effects
- Multiple instances

## Supported Image Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PNG | .png | With alpha transparency |
| JPEG | .jpg, .jpeg | Lossy compression |
| BMP | .bmp | Windows bitmap |
| GIF | .gif | Animated GIF support |
| WEBP | .webp | Modern format with animation |
| TIFF | .tif, .tiff | Professional format |
| TGA | .tga | Truevision Targa |
| PCX | .pcx | Legacy PC format |
| PNM | .ppm, .pgm, .pbm | Portable anymap |
| XPM | .xpm | X PixMap |
| XCF | .xcf | GIMP native format |
| ICO | .ico | Windows icons |
| CUR | .cur | Windows cursors |
| SVG | .svg | Scalable vector graphics |

## Performance Tips

1. **Texture Caching** - Load textures once, reuse many times
2. **Power-of-2 Sizes** - 64x64, 128x128, 256x256 for best GPU performance
3. **Batch Loading** - Use `nl_img_load_icon_batch` for multiple images
4. **Format Choice**:
   - PNG: UI elements, icons, images with transparency
   - JPEG: Photos, large images without transparency
   - WEBP: Modern alternative with better compression
5. **Alpha Modulation** - Cheaper than creating multiple pre-faded textures
6. **Sprite Sheets** - One large texture is faster than many small ones

## Common Patterns

### Animated Sprite

```nano
let mut frame: int = 0
let frame_time: int = 100  # ms per frame

while running {
    let current_time: int = (SDL_GetTicks)
    set frame (/ current_time frame_time)
    let frame_index: int = (% frame 8)  # 8 frames
    let frame_x: int = (* frame_index 64)
    
    (nl_img_render_texture_sprite renderer sprite_sheet frame_x 0 64 64 x y 64 64)
}
```

### Fading Transition

```nano
let mut fade_alpha: int = 0
while (< fade_alpha 255) {
    (nl_img_set_texture_alpha texture fade_alpha)
    (nl_img_render_texture renderer texture 0 0 800 600)
    set fade_alpha (+ fade_alpha 5)
    unsafe { (SDL_Delay 16) }
}
```

### Rotating Icon

```nano
let mut angle: float = 0.0
while running {
    (nl_img_render_texture_ex renderer icon 400 300 80 80 angle SDL_FLIP_NONE)
    set angle (+ angle 2.0)
    if (> angle 360.0) {
        set angle 0.0
    } else {}
}
```

### Tinted Overlay

```nano
# Red tint for damage effect
(nl_img_set_texture_color overlay 255 0 0)
(nl_img_set_texture_alpha overlay 100)
(nl_img_set_texture_blend_mode overlay SDL_BLENDMODE_ADD)
(nl_img_render_texture renderer overlay 0 0 800 600)
```

## Error Handling

```nano
let texture: int = (nl_img_load_texture renderer "missing.png")
if (== texture 0) {
    let error: string = (IMG_GetError)
    (println (+ "Failed to load image: " error))
    # Fallback to placeholder or exit
} else {
    (println "Image loaded successfully")
}
```

## Integration with Example Launcher

Update `examples/sdl_example_launcher.nano`:

```nano
import "modules/sdl_image/sdl_image.nano"

fn main() -> int {
    # ... existing init code ...
    
    unsafe { (IMG_Init IMG_INIT_PNG) }
    
    # Load all icons
    let icon_asteroids: int = (nl_img_load_texture renderer "examples/icons/sdl_asteroids.png")
    let icon_checkers: int = (nl_img_load_texture renderer "examples/icons/sdl_checkers.png")
    # ... load other icons ...
    
    # In render loop, replace colored squares:
    # OLD: unsafe { (nl_sdl_render_fill_rect renderer icon_x icon_y ICON_SIZE ICON_SIZE) }
    # NEW:
    (nl_img_render_texture renderer icon_asteroids icon_x icon_y ICON_SIZE ICON_SIZE)
    
    # Cleanup
    (nl_img_destroy_texture icon_asteroids)
    # ... destroy other icons ...
    unsafe { (IMG_Quit) }
}
```

## Advanced Topics

### Creating Textures from Memory

```nano
# For procedurally generated images
let pixels: int = # ... allocate RGBA pixel data ...
let tex: int = (nl_img_create_texture_from_pixels renderer 256 256 pixels)
```

### Animated GIF/WEBP

```nano
# Load animation
let anim: int = (IMG_LoadAnimation "animated.gif")

# Access frames (requires additional frame extraction code)
# See SDL_image documentation for IMG_Animation structure

# Cleanup
(IMG_FreeAnimation anim)
```

### RWops Loading (from memory/network)

```nano
# Advanced - load from memory buffer or custom source
let rw: int = # ... create SDL_RWops ...
let surface: int = (IMG_Load_RW rw 1)  # 1 = free RWops after load
```

## Troubleshooting

**Images don't load:**
1. Check file path is correct (relative to working directory)
2. Verify SDL_image is initialized: `(IMG_Init IMG_INIT_PNG)`
3. Check format is supported: `(IMG_GetError)`
4. Ensure file permissions allow reading

**Images appear garbled:**
1. Verify correct pixel format (RGBA expected)
2. Check endianness on different platforms
3. Ensure SDL_image version supports the format

**Poor performance:**
1. Load textures once (not every frame)
2. Use texture caching
3. Consider smaller texture sizes
4. Use sprite sheets instead of individual files

## Version Information

This module wraps SDL_image 2.x. Some features (JXL, AVIF) require SDL_image 2.6+.

Check version:
```nano
let version: int = (IMG_Linked_Version)
# Parse version from returned value
```

## See Also

- `examples/sdl_image_test.nano` - Basic test
- `examples/sdl_image_demo.nano` - Complete feature demo
- `examples/sdl_example_launcher.nano` - Real-world usage
- [SDL_image Documentation](https://www.libsdl.org/projects/SDL_image/)
