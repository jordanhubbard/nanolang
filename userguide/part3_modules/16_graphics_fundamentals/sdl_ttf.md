# 16.4 SDL_ttf - TrueType Fonts

**Render text using TrueType and OpenType fonts.**

SDL_ttf enables high-quality text rendering by rasterizing TrueType (`.ttf`) and OpenType (`.otf`) fonts into SDL surfaces. The resulting surfaces can then be converted to textures and drawn with SDL's renderer. NanoLang exposes SDL_ttf through `modules/sdl_ttf/sdl_ttf.nano`.

## Overview

SDL_ttf provides:

- Loading fonts at arbitrary point sizes
- Three rendering modes: Solid, Blended, and Shaded
- Font metric queries (height, ascent, descent, line skip)
- Text dimension measurement before rendering
- Style flags: bold, italic, underline, strikethrough
- Multiple simultaneous fonts at different sizes

## Initialization

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_Init, TTF_Quit, TTF_GetError, TTF_WasInit

fn init_ttf() -> void {
    if (== (TTF_WasInit) 0) {
        let result: int = (TTF_Init)
        if (!= result 0) {
            (print (TTF_GetError))
        }
    }
}

shadow init_ttf { assert true }
```

Call `TTF_Quit` at shutdown:

```nano
(TTF_Quit)
```

## Loading Fonts

`TTF_OpenFont` loads a font file at a given point size and returns a `TTF_Font` opaque handle.

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_OpenFont, TTF_CloseFont, TTF_Font

fn load_fonts() -> void {
    let body: TTF_Font = (TTF_OpenFont "fonts/OpenSans-Regular.ttf" 16)
    let title: TTF_Font = (TTF_OpenFont "fonts/OpenSans-Bold.ttf" 32)
    let small: TTF_Font = (TTF_OpenFont "fonts/OpenSans-Regular.ttf" 11)

    # ... use fonts ...

    (TTF_CloseFont small)
    (TTF_CloseFont title)
    (TTF_CloseFont body)
}

shadow load_fonts { assert true }
```

The point size determines the rendered size of the font. Smaller values are faster to render; larger values are crisper at large display sizes.

## Rendering Modes

SDL_ttf offers three rendering modes, each returning a surface pointer (`int`) that you then convert to a texture.

### Solid Rendering

Fast and simple — opaque background, aliased edges. Good for frequently-updated text (scores, timers).

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_RenderText_Solid

# White text on transparent background (aliased)
let surface: int = (TTF_RenderText_Solid font "Score: 0" 255 255 255 255)
```

### Blended Rendering

Anti-aliased with alpha blending — smooth edges, transparent background. Best quality for static labels, titles, and UI text.

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_RenderText_Blended

# Yellow text, fully opaque, with smooth anti-aliasing
let surface: int = (TTF_RenderText_Blended font "Player Name" 255 220 0 255)
```

### Shaded Rendering

Anti-aliased text on a solid colored background box. Useful for readability over complex backgrounds.

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_RenderText_Shaded

# White text on dark blue background
let surface: int = (TTF_RenderText_Shaded font "INFO"
    255 255 255 255    # foreground: white
    20 20 80 255)      # background: dark blue
```

**Choosing a mode:**

| Mode | Quality | Speed | Transparency | Use case |
|------|---------|-------|--------------|----------|
| Solid | Low (aliased) | Fast | Transparent BG | Scores, HUD counters |
| Blended | High (anti-aliased) | Slower | Transparent BG | Titles, labels |
| Shaded | High (anti-aliased) | Medium | Opaque BG box | Tooltips, info boxes |

## Surface to Texture

After rendering text to a surface, convert it to a GPU texture for drawing:

```nano
from "modules/sdl/sdl.nano" import SDL_CreateTextureFromSurface, SDL_FreeSurface,
    SDL_RenderCopy, SDL_DestroyTexture, SDL_Texture

fn render_text_at(renderer: SDL_Renderer, surface: int, x: int, y: int) -> void {
    let tex: SDL_Texture = (SDL_CreateTextureFromSurface renderer surface)
    (SDL_FreeSurface surface)   # free CPU surface after uploading

    # dstrect=0 draws at (0,0) full size; use a C shim for positioned rects
    (SDL_RenderCopy renderer tex 0 0)
    (SDL_DestroyTexture tex)
}

shadow render_text_at { assert true }
```

> **Note:** For positioned text rendering, you need a C shim that creates an `SDL_Rect` at the desired `(x, y, w, h)`. The `SDL_RenderCopy` function accepts a rect pointer, and passing `0` means "entire rendering area".

## Measuring Text

Before rendering, measure the pixel dimensions of a string with `TTF_SizeText`. This is essential for centering text or creating dynamic layouts.

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_SizeText

fn measure(font: TTF_Font, text: string) -> void {
    # Pass 0 as out-pointer for w and h — use C shim for actual values
    let result: int = (TTF_SizeText font text 0 0)
    # With a C shim, you would read the w and h from output pointers
}

shadow measure { assert true }
```

## Font Metrics

Query font-level metrics (in pixels) for layout calculations:

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import
    TTF_FontHeight, TTF_FontAscent, TTF_FontDescent, TTF_FontLineSkip

let height: int = (TTF_FontHeight font)        # total height of font
let ascent: int = (TTF_FontAscent font)        # height above baseline
let descent: int = (TTF_FontDescent font)      # depth below baseline (negative)
let line_skip: int = (TTF_FontLineSkip font)   # recommended line spacing
```

Use `TTF_FontLineSkip` rather than `TTF_FontHeight` for multi-line spacing — it accounts for inter-line gaps recommended by the font designer.

## Font Styles

Apply bold, italic, underline, or strikethrough effects programmatically. Styles combine with bitwise OR (addition in NanoLang).

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import
    TTF_SetFontStyle, TTF_GetFontStyle,
    TTF_STYLE_NORMAL, TTF_STYLE_BOLD, TTF_STYLE_ITALIC,
    TTF_STYLE_UNDERLINE, TTF_STYLE_STRIKETHROUGH

# Bold + italic
(TTF_SetFontStyle font (+ TTF_STYLE_BOLD TTF_STYLE_ITALIC))

# Just underline
(TTF_SetFontStyle font TTF_STYLE_UNDERLINE)

# Reset to normal
(TTF_SetFontStyle font TTF_STYLE_NORMAL)

# Query current style
let style: int = (TTF_GetFontStyle font)
```

**Style constants:**

| Constant | Value | Effect |
|----------|-------|--------|
| `TTF_STYLE_NORMAL` | 0 | Regular |
| `TTF_STYLE_BOLD` | 1 | Bold |
| `TTF_STYLE_ITALIC` | 2 | Italic |
| `TTF_STYLE_UNDERLINE` | 4 | Underline |
| `TTF_STYLE_STRIKETHROUGH` | 8 | Strikethrough |

> **Note:** Software-rendered styles (bold, italic) are applied by the SDL_ttf rasterizer. For best quality, use a font file that natively includes those variants (e.g., `OpenSans-Bold.ttf`) rather than relying on synthetic bold.

## Using Multiple Fonts

There is no limit to the number of fonts open simultaneously. Common patterns:

```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_OpenFont, TTF_CloseFont, TTF_Font

# UI font set
let ui_small: TTF_Font = (TTF_OpenFont "fonts/Roboto-Regular.ttf" 12)
let ui_body: TTF_Font = (TTF_OpenFont "fonts/Roboto-Regular.ttf" 16)
let ui_heading: TTF_Font = (TTF_OpenFont "fonts/Roboto-Bold.ttf" 24)
let ui_title: TTF_Font = (TTF_OpenFont "fonts/Roboto-Bold.ttf" 48)

# Monospace font for code/console output
let mono: TTF_Font = (TTF_OpenFont "fonts/JetBrainsMono-Regular.ttf" 14)
```

Always close every font you open:

```nano
(TTF_CloseFont ui_small)
(TTF_CloseFont ui_body)
(TTF_CloseFont ui_heading)
(TTF_CloseFont ui_title)
(TTF_CloseFont mono)
```

## Complete Example: Animated Score Display

```nano
from "modules/sdl/sdl.nano" import
    SDL_Init, SDL_Quit,
    SDL_CreateWindow, SDL_DestroyWindow,
    SDL_CreateRenderer, SDL_DestroyRenderer,
    SDL_SetRenderDrawColor, SDL_RenderClear, SDL_RenderPresent,
    SDL_CreateTextureFromSurface, SDL_RenderCopy, SDL_DestroyTexture,
    SDL_FreeSurface, SDL_Texture, SDL_Renderer, SDL_Window,
    SDL_PollEvent, SDL_Delay, SDL_GetTicks,
    SDL_INIT_VIDEO, SDL_WINDOWPOS_CENTERED, SDL_WINDOW_SHOWN,
    SDL_RENDERER_ACCELERATED, SDL_RENDERER_PRESENTVSYNC

from "modules/sdl_ttf/sdl_ttf.nano" import
    TTF_Init, TTF_Quit,
    TTF_OpenFont, TTF_CloseFont, TTF_Font,
    TTF_RenderText_Blended, TTF_RenderText_Solid

fn main() -> void {
    (SDL_Init SDL_INIT_VIDEO)
    (TTF_Init)

    let window: SDL_Window = (SDL_CreateWindow "Font Demo"
        SDL_WINDOWPOS_CENTERED SDL_WINDOWPOS_CENTERED
        640 480 SDL_WINDOW_SHOWN)

    let renderer: SDL_Renderer = (SDL_CreateRenderer window -1
        (+ SDL_RENDERER_ACCELERATED SDL_RENDERER_PRESENTVSYNC))

    let title_font: TTF_Font = (TTF_OpenFont "fonts/OpenSans-Bold.ttf" 36)
    let body_font: TTF_Font = (TTF_OpenFont "fonts/OpenSans-Regular.ttf" 18)

    let mut score: int = 0
    let mut running: bool = true

    while running {
        let mut ev: int = (SDL_PollEvent 0)
        while (!= ev 0) {
            set ev (SDL_PollEvent 0)
        }

        # Increment score over time
        let t: int = (SDL_GetTicks)
        set score (/ t 100)

        # Clear to dark background
        (SDL_SetRenderDrawColor renderer 15 15 30 255)
        (SDL_RenderClear renderer)

        # Render title (blended = anti-aliased)
        let title_surf: int = (TTF_RenderText_Blended title_font "NanoLang Demo" 255 220 50 255)
        let title_tex: SDL_Texture = (SDL_CreateTextureFromSurface renderer title_surf)
        (SDL_FreeSurface title_surf)
        (SDL_RenderCopy renderer title_tex 0 0)
        (SDL_DestroyTexture title_tex)

        # Cleanup
        (SDL_RenderPresent renderer)
        (SDL_Delay 16)
    }

    (TTF_CloseFont title_font)
    (TTF_CloseFont body_font)
    (SDL_DestroyRenderer renderer)
    (SDL_DestroyWindow window)
    (TTF_Quit)
    (SDL_Quit)
}

shadow main { assert true }
```

## Performance Tips

- **Cache textures**: Pre-render static text (labels, instructions) into textures once. Only re-render when text changes.
- **Dynamic text**: For frequently-changing text (scores, FPS counters), use Solid rendering — it is significantly faster than Blended.
- **Texture atlases**: For many small strings, consider rendering them all into one large texture and using sub-regions.
- **Line skip**: When laying out multiple lines, use `TTF_FontLineSkip` rather than `TTF_FontHeight` to get correct inter-line spacing.

## API Summary

| Function | Description |
|----------|-------------|
| `TTF_Init()` | Initialize SDL_ttf |
| `TTF_Quit()` | Shutdown SDL_ttf |
| `TTF_WasInit()` | Returns 1 if already initialized |
| `TTF_GetError()` | Return last error string |
| `TTF_OpenFont(file, ptsize)` | Load a font at given point size |
| `TTF_CloseFont(font)` | Unload font |
| `TTF_RenderText_Solid(font, text, r, g, b, a)` | Render text fast, aliased |
| `TTF_RenderText_Blended(font, text, r, g, b, a)` | Render text smooth, anti-aliased |
| `TTF_RenderText_Shaded(font, text, fr, fg, fb, fa, br, bg, bb, ba)` | Render on colored background |
| `TTF_SizeText(font, text, w_out, h_out)` | Measure rendered text dimensions |
| `TTF_FontHeight(font)` | Total font height in pixels |
| `TTF_FontAscent(font)` | Height above baseline |
| `TTF_FontDescent(font)` | Depth below baseline |
| `TTF_FontLineSkip(font)` | Recommended line spacing |
| `TTF_GetFontStyle(font)` | Get current style flags |
| `TTF_SetFontStyle(font, style)` | Set style flags |

---

**Previous:** [16.3 SDL_mixer](sdl_mixer.html)
**Next:** [Chapter 17: OpenGL](../17_opengl/index.html)
