# 16.2 SDL_image - Image Loading

**Load PNG, JPEG, BMP, GIF, WEBP, and many other image formats into SDL textures.**

SDL_image extends SDL2 with support for a wide variety of image file formats. In NanoLang, it is exposed through `modules/sdl_image/sdl_image.nano` and provides both low-level SDL_image functions and higher-level NanoLang helper functions prefixed with `nl_img_`.

## Overview

The SDL_image module lets you:

- Load image files directly into GPU textures (`IMG_LoadTexture`)
- Load images into CPU surfaces for further processing (`IMG_Load`)
- Query and manipulate texture dimensions and appearance
- Save surfaces as PNG or JPEG
- Batch-load sprite sheets and icon sets
- Detect image formats programmatically

## Initialization

SDL_image has its own initialization step, separate from `SDL_Init`. Call `IMG_Init` with a bitmask of the formats you want to support. Always call `IMG_Quit` at shutdown.

```nano
from "modules/sdl_image/sdl_image.nano" import
    IMG_Init, IMG_Quit, IMG_GetError,
    IMG_INIT_PNG, IMG_INIT_JPG, IMG_INIT_WEBP

fn init_image() -> void {
    let flags: int = (+ IMG_INIT_PNG IMG_INIT_JPG)
    let result: int = (IMG_Init flags)
    if (== result 0) {
        (print (IMG_GetError))
    }
}

shadow init_image { assert true }
```

**Format flags:**

| Constant | Value | Format |
|----------|-------|--------|
| `IMG_INIT_JPG` | 1 | JPEG |
| `IMG_INIT_PNG` | 2 | PNG |
| `IMG_INIT_TIF` | 4 | TIFF |
| `IMG_INIT_WEBP` | 8 | WebP |
| `IMG_INIT_JXL` | 16 | JPEG XL |
| `IMG_INIT_AVIF` | 32 | AVIF |
| `IMG_INIT_ALL` | 63 | All supported formats |

## Loading Textures

The simplest path from file to screen is `IMG_LoadTexture`, which loads the image directly into a GPU texture handle (returned as `int`).

```nano
from "modules/sdl_image/sdl_image.nano" import IMG_LoadTexture

fn load_sprite(renderer: SDL_Renderer) -> int {
    let texture: int = (IMG_LoadTexture renderer "assets/player.png")
    if (== texture 0) {
        (print "Failed to load sprite")
    }
    return texture
}

shadow load_sprite { assert true }
```

The NanoLang helper `nl_img_load_texture` is equivalent and adds additional error reporting:

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_load_texture

let tex: int = (nl_img_load_texture renderer "assets/background.png")
```

For PNG files specifically, use `nl_img_load_png_texture` which validates the format before loading:

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_load_png_texture

let tex: int = (nl_img_load_png_texture renderer "assets/ui/button.png")
```

## Loading Surfaces

If you need to manipulate pixel data on the CPU side before uploading to the GPU, load into a surface first. `IMG_Load` returns a raw surface pointer as `int`.

```nano
from "modules/sdl_image/sdl_image.nano" import IMG_Load
from "modules/sdl/sdl.nano" import SDL_CreateTextureFromSurface, SDL_FreeSurface

fn surface_to_texture(renderer: SDL_Renderer, path: string) -> SDL_Texture {
    let surface: int = (IMG_Load path)
    let texture: SDL_Texture = (SDL_CreateTextureFromSurface renderer surface)
    (SDL_FreeSurface surface)
    return texture
}

shadow surface_to_texture { assert true }
```

## Rendering Textures

The `nl_img_render_texture` helper renders a texture at a given position and size in one call:

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_render_texture

fn draw_image(renderer: SDL_Renderer, tex: int, x: int, y: int) -> void {
    let w: int = (nl_img_get_texture_width tex)
    let h: int = (nl_img_get_texture_height tex)
    (nl_img_render_texture renderer tex x y w h)
}

shadow draw_image { assert true }
```

### Rotation and Flipping

`nl_img_render_texture_ex` adds rotation (in degrees) and a flip flag:

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_render_texture_ex,
    SDL_FLIP_NONE, SDL_FLIP_HORIZONTAL, SDL_FLIP_VERTICAL

# Draw flipped horizontally at 45 degrees rotation
(nl_img_render_texture_ex renderer tex 100 200 64 64 45.0 SDL_FLIP_HORIZONTAL)
```

**Flip constants:**

| Constant | Effect |
|----------|--------|
| `SDL_FLIP_NONE` | No flip |
| `SDL_FLIP_HORIZONTAL` | Mirror left-right |
| `SDL_FLIP_VERTICAL` | Mirror top-bottom |
| `SDL_FLIP_BOTH` | Mirror both axes |

### Sprite Sheets

Extract a sub-region from a sprite sheet (atlas) with `nl_img_render_texture_sprite`:

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_render_texture_sprite

# Sprite sheet has 32x32 sprites arranged in a grid.
# Draw sprite at column 2, row 1, to screen position (100, 150) at 64x64.
let src_x: int = (* 2 32)
let src_y: int = (* 1 32)
(nl_img_render_texture_sprite renderer sheet_tex src_x src_y 32 32 100 150 64 64)
```

## Querying Texture Size

```nano
from "modules/sdl_image/sdl_image.nano" import
    nl_img_get_texture_width, nl_img_get_texture_height

let tex: int = (nl_img_load_texture renderer "assets/icon.png")
let w: int = (nl_img_get_texture_width tex)
let h: int = (nl_img_get_texture_height tex)
```

## Texture Appearance Modifiers

### Alpha (Transparency)

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_set_texture_alpha

# 0 = fully transparent, 255 = fully opaque
(nl_img_set_texture_alpha tex 128)
```

### Color Tint

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_set_texture_color

# Tint the texture red
(nl_img_set_texture_color tex 255 100 100)
```

### Blend Mode

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_set_texture_blend_mode
from "modules/sdl_image/sdl_image.nano" import SDL_BLENDMODE_BLEND

(nl_img_set_texture_blend_mode tex SDL_BLENDMODE_BLEND)
```

## Batch Loading

Load multiple icon files at once with `nl_img_load_icon_batch`:

```nano
from "modules/sdl_image/sdl_image.nano" import
    nl_img_load_icon_batch, nl_img_destroy_texture_batch

let files: array<string> = ["icons/fire.png", "icons/water.png", "icons/earth.png"]
let textures: array<int> = (nl_img_load_icon_batch renderer files 3)

# ... use textures[0], textures[1], textures[2] ...

(nl_img_destroy_texture_batch textures 3)
```

## Freeing Textures

Always free textures when you are done with them:

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_destroy_texture

(nl_img_destroy_texture tex)
```

## Saving Images

Write a surface back to disk as PNG or JPEG:

```nano
from "modules/sdl_image/sdl_image.nano" import IMG_SavePNG, IMG_SaveJPG

# Save as PNG (lossless)
(IMG_SavePNG surface "screenshot.png")

# Save as JPEG with quality 85 (0–100)
(IMG_SaveJPG surface "screenshot.jpg" 85)
```

## Format Detection

Check whether a file is a supported format before loading it:

```nano
from "modules/sdl_image/sdl_image.nano" import nl_img_can_load, nl_img_get_supported_formats

let ok: int = (nl_img_can_load "mystery.dat")
if (!= ok 0) {
    let tex: int = (nl_img_load_texture renderer "mystery.dat")
}

let formats: array<string> = (nl_img_get_supported_formats)
```

## Complete Example: Loading and Displaying a Sprite

```nano
from "modules/sdl/sdl.nano" import
    SDL_Init, SDL_Quit,
    SDL_CreateWindow, SDL_DestroyWindow,
    SDL_CreateRenderer, SDL_DestroyRenderer,
    SDL_SetRenderDrawColor, SDL_RenderClear, SDL_RenderPresent,
    SDL_PollEvent, SDL_Delay,
    SDL_Window, SDL_Renderer,
    SDL_INIT_VIDEO, SDL_WINDOWPOS_CENTERED, SDL_WINDOW_SHOWN,
    SDL_RENDERER_ACCELERATED, SDL_RENDERER_PRESENTVSYNC

from "modules/sdl_image/sdl_image.nano" import
    IMG_Init, IMG_Quit,
    nl_img_load_texture, nl_img_destroy_texture,
    nl_img_render_texture,
    nl_img_get_texture_width, nl_img_get_texture_height,
    IMG_INIT_PNG

fn run() -> void {
    (SDL_Init SDL_INIT_VIDEO)
    (IMG_Init IMG_INIT_PNG)

    let window: SDL_Window = (SDL_CreateWindow "Sprite Demo"
        SDL_WINDOWPOS_CENTERED SDL_WINDOWPOS_CENTERED
        640 480 SDL_WINDOW_SHOWN)

    let renderer: SDL_Renderer = (SDL_CreateRenderer window -1
        (+ SDL_RENDERER_ACCELERATED SDL_RENDERER_PRESENTVSYNC))

    let sprite: int = (nl_img_load_texture renderer "assets/hero.png")
    let sw: int = (nl_img_get_texture_width sprite)
    let sh: int = (nl_img_get_texture_height sprite)

    let mut x: int = 0

    let mut running: bool = true
    while running {
        let mut ev: int = (SDL_PollEvent 0)
        while (!= ev 0) {
            set ev (SDL_PollEvent 0)
        }

        # Scroll sprite across screen
        set x (% (+ x 2) 640)

        (SDL_SetRenderDrawColor renderer 20 20 40 255)
        (SDL_RenderClear renderer)

        (nl_img_render_texture renderer sprite x 200 sw sh)

        (SDL_RenderPresent renderer)
        (SDL_Delay 16)
    }

    (nl_img_destroy_texture sprite)
    (SDL_DestroyRenderer renderer)
    (SDL_DestroyWindow window)
    (IMG_Quit)
    (SDL_Quit)
}

shadow run { assert true }
```

## API Summary

| Function | Description |
|----------|-------------|
| `IMG_Init(flags)` | Initialize image format support |
| `IMG_Quit()` | Shut down SDL_image |
| `IMG_GetError()` | Return last error string |
| `IMG_Load(file)` | Load image into a surface (int ptr) |
| `IMG_LoadTexture(renderer, file)` | Load image directly into a texture (int) |
| `nl_img_load_texture(renderer, file)` | NanoLang wrapper for IMG_LoadTexture |
| `nl_img_load_png_texture(renderer, file)` | Load PNG with format validation |
| `nl_img_render_texture(renderer, tex, x, y, w, h)` | Draw texture at position |
| `nl_img_render_texture_ex(renderer, tex, x, y, w, h, angle, flip)` | Draw with rotation/flip |
| `nl_img_render_texture_sprite(renderer, tex, sx, sy, sw, sh, dx, dy, dw, dh)` | Draw sub-region (sprite sheet) |
| `nl_img_get_texture_width(tex)` | Get texture width in pixels |
| `nl_img_get_texture_height(tex)` | Get texture height in pixels |
| `nl_img_set_texture_alpha(tex, alpha)` | Set transparency (0–255) |
| `nl_img_set_texture_color(tex, r, g, b)` | Set color tint |
| `nl_img_set_texture_blend_mode(tex, mode)` | Set blend mode |
| `nl_img_load_icon_batch(renderer, files, count)` | Batch load textures |
| `nl_img_destroy_texture(tex)` | Free a texture |
| `nl_img_destroy_texture_batch(textures, count)` | Free multiple textures |
| `nl_img_can_load(file)` | Check if file is a supported format |
| `nl_img_get_supported_formats()` | List supported format names |
| `IMG_SavePNG(surface, file)` | Save surface as PNG |
| `IMG_SaveJPG(surface, file, quality)` | Save surface as JPEG |

---

**Previous:** [16.1 SDL](sdl.html)
**Next:** [16.3 SDL_mixer](sdl_mixer.html)
