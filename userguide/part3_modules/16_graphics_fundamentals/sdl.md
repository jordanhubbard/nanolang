# 16.1 SDL - Simple DirectMedia Layer

**Create windows, handle input, and render 2D graphics.**

SDL (Simple DirectMedia Layer) is a cross-platform library that provides low-level access to audio, keyboard, mouse, joystick, and graphics hardware via OpenGL and Direct3D. NanoLang exposes SDL2 directly through the `modules/sdl/sdl.nano` module, using opaque handle types for windows, renderers, and textures.

## Overview

The SDL module provides:

- Window creation and management
- Hardware-accelerated 2D rendering via `SDL_Renderer`
- Drawing primitives: points, lines, rectangles
- Texture creation and blitting
- Event polling (keyboard, mouse, window events)
- Frame timing and delays
- OpenGL context management (for use with Chapter 17 modules)

## Opaque Types

SDL resources are represented as opaque handles. You cannot inspect their internals — pass them to SDL functions to operate on them.

| Type | Description |
|------|-------------|
| `SDL_Window` | A window on screen |
| `SDL_Renderer` | Hardware-accelerated renderer attached to a window |
| `SDL_Texture` | An image stored in GPU memory |
| `SDL_Surface` | An image stored in CPU memory |
| `SDL_GLContext` | An OpenGL context associated with a window |

## Initialization and Cleanup

SDL must be initialized before any other calls. Pass a bitmask of subsystems to enable:

```nano
from "modules/sdl/sdl.nano" import SDL_Init, SDL_Quit, SDL_GetError,
    SDL_INIT_VIDEO, SDL_INIT_AUDIO, SDL_INIT_EVERYTHING

fn main() -> void {
    let result: int = (SDL_Init SDL_INIT_VIDEO)
    if (!= result 0) {
        let msg: string = (SDL_GetError)
        (print msg)
        return
    }

    # ... do work ...

    (SDL_Quit)
}

shadow main { assert true }
```

**Init flags:**

| Constant | Value | Meaning |
|----------|-------|---------|
| `SDL_INIT_VIDEO` | 32 | Video subsystem (required for windows) |
| `SDL_INIT_AUDIO` | 16 | Audio subsystem |
| `SDL_INIT_TIMER` | 1 | Timer subsystem |
| `SDL_INIT_EVERYTHING` | 62977 | All subsystems |

## Window Creation

`SDL_CreateWindow` opens a window and returns an `SDL_Window` handle. Position it with `SDL_WINDOWPOS_CENTERED` or `SDL_WINDOWPOS_UNDEFINED`.

```nano
from "modules/sdl/sdl.nano" import SDL_CreateWindow, SDL_DestroyWindow,
    SDL_Window, SDL_WINDOWPOS_CENTERED, SDL_WINDOW_SHOWN, SDL_WINDOW_RESIZABLE

fn open_window() -> SDL_Window {
    let window: SDL_Window = (SDL_CreateWindow
        "My NanoLang App"
        SDL_WINDOWPOS_CENTERED
        SDL_WINDOWPOS_CENTERED
        800
        600
        SDL_WINDOW_SHOWN)
    return window
}

shadow open_window { assert true }
```

**Window flags (combinable with bitwise OR via `|`):**

| Constant | Description |
|----------|-------------|
| `SDL_WINDOW_SHOWN` | Window is visible |
| `SDL_WINDOW_RESIZABLE` | User can resize it |
| `SDL_WINDOW_FULLSCREEN` | Exclusive fullscreen |
| `SDL_WINDOW_FULLSCREEN_DESKTOP` | Fullscreen at desktop resolution |
| `SDL_WINDOW_OPENGL` | For use with an OpenGL context |

Destroy the window when done:

```nano
(SDL_DestroyWindow window)
```

You can also update window properties at runtime:

```nano
(SDL_SetWindowTitle window "Updated Title")
(SDL_SetWindowSize window 1024 768)
```

## Renderer Setup

A renderer provides accelerated 2D drawing on top of a window. Use `SDL_RENDERER_ACCELERATED` for GPU rendering, and `SDL_RENDERER_PRESENTVSYNC` to cap to the display refresh rate.

```nano
from "modules/sdl/sdl.nano" import SDL_CreateRenderer, SDL_DestroyRenderer,
    SDL_Renderer, SDL_RENDERER_ACCELERATED, SDL_RENDERER_PRESENTVSYNC

fn create_renderer(window: SDL_Window) -> SDL_Renderer {
    # -1 = use first available driver
    let flags: int = (+ SDL_RENDERER_ACCELERATED SDL_RENDERER_PRESENTVSYNC)
    let renderer: SDL_Renderer = (SDL_CreateRenderer window -1 flags)
    return renderer
}

shadow create_renderer { assert true }
```

Destroy the renderer before the window:

```nano
(SDL_DestroyRenderer renderer)
(SDL_DestroyWindow window)
(SDL_Quit)
```

## Setting Colors

Before drawing, set the active color with `SDL_SetRenderDrawColor`. Components are 0–255; alpha 255 is fully opaque.

```nano
from "modules/sdl/sdl.nano" import SDL_SetRenderDrawColor

# Red
(SDL_SetRenderDrawColor renderer 255 0 0 255)

# Semi-transparent blue
(SDL_SetRenderDrawColor renderer 0 0 255 128)
```

For alpha blending to work, enable blend mode:

```nano
from "modules/sdl/sdl.nano" import SDL_SetRenderDrawBlendMode, SDL_BLENDMODE_BLEND
(SDL_SetRenderDrawBlendMode renderer SDL_BLENDMODE_BLEND)
```

## Clearing and Presenting

At the start of each frame, clear the screen with the current draw color. At the end, present the rendered frame.

```nano
from "modules/sdl/sdl.nano" import SDL_RenderClear, SDL_RenderPresent

# Clear to current color
(SDL_RenderClear renderer)

# ... draw things ...

# Show the frame
(SDL_RenderPresent renderer)
```

## Drawing Primitives

### Points

```nano
from "modules/sdl/sdl.nano" import SDL_RenderDrawPoint

(SDL_SetRenderDrawColor renderer 255 255 0 255)   # Yellow
(SDL_RenderDrawPoint renderer 400 300)
```

### Lines

```nano
from "modules/sdl/sdl.nano" import SDL_RenderDrawLine

(SDL_SetRenderDrawColor renderer 0 255 0 255)   # Green
(SDL_RenderDrawLine renderer 0 0 800 600)       # diagonal
```

### Rectangles

`SDL_RenderDrawRect` draws an outline; `SDL_RenderFillRect` fills it. Both accept a rect pointer (use 0 to mean "entire render target"):

```nano
from "modules/sdl/sdl.nano" import SDL_RenderFillRect, SDL_RenderDrawRect

# Fill the entire renderer with the current color
(SDL_SetRenderDrawColor renderer 100 100 200 255)
(SDL_RenderFillRect renderer 0)
```

> **Note:** Passing `0` as the rect pointer draws the operation over the entire rendering target. For positioned rectangles, use SDL_image helpers or a C shim that creates an `SDL_Rect` struct.

## Texture Operations

Textures are GPU-side images. You typically create them from surfaces (using `SDL_CreateTextureFromSurface`) or load them via the SDL_image module.

```nano
from "modules/sdl/sdl.nano" import SDL_CreateTextureFromSurface, SDL_DestroyTexture,
    SDL_RenderCopy, SDL_FreeSurface, SDL_Texture, SDL_Surface

fn render_surface(renderer: SDL_Renderer, surface: SDL_Surface) -> void {
    let tex: SDL_Texture = (SDL_CreateTextureFromSurface renderer surface)
    (SDL_FreeSurface surface)

    # srcrect=0 means full source; dstrect=0 means full destination
    (SDL_RenderCopy renderer tex 0 0)
    (SDL_DestroyTexture tex)
}

shadow render_surface { assert true }
```

Control texture transparency:

```nano
(SDL_SetTextureAlphaMod texture 128)             # 50% transparent
(SDL_SetTextureBlendMode texture SDL_BLENDMODE_BLEND)
```

## Event Loop

SDL events arrive through `SDL_PollEvent`. Pass an integer pointer (use 0 for simple quit detection, or a C shim for full event data). The function returns 1 if an event was dequeued, 0 otherwise.

```nano
from "modules/sdl/sdl.nano" import SDL_PollEvent, SDL_Delay

fn run_loop(renderer: SDL_Renderer) -> void {
    let mut running: bool = true

    while running {
        # Drain the event queue
        let mut has_event: int = (SDL_PollEvent 0)
        while (!= has_event 0) {
            # With a 0 pointer, we cannot inspect event fields.
            # Use a C shim or the nl_sdl helpers for typed event access.
            set has_event (SDL_PollEvent 0)
        }

        # Render frame: dark grey background
        (SDL_SetRenderDrawColor renderer 30 30 30 255)
        (SDL_RenderClear renderer)
        (SDL_RenderPresent renderer)

        # ~60 fps
        (SDL_Delay 16)
    }
}

shadow run_loop { assert true }
```

> **Tip:** For real applications, use a C shim that returns the event type integer so you can branch on `SDL_QUIT` (value 256), `SDL_KEYDOWN` (768), and `SDL_MOUSEBUTTONDOWN` (1025).

## Timing

```nano
from "modules/sdl/sdl.nano" import SDL_GetTicks, SDL_Delay

let start: int = (SDL_GetTicks)
(SDL_Delay 16)
let elapsed: int = (- (SDL_GetTicks) start)
```

`SDL_GetTicks` returns milliseconds since SDL was initialized. `SDL_Delay` sleeps for approximately the given number of milliseconds.

## Complete Minimal Example

This puts everything together: a window that fills the background with a color that cycles over time.

```nano
from "modules/sdl/sdl.nano" import
    SDL_Init, SDL_Quit, SDL_GetError,
    SDL_CreateWindow, SDL_DestroyWindow,
    SDL_CreateRenderer, SDL_DestroyRenderer,
    SDL_SetRenderDrawColor, SDL_RenderClear, SDL_RenderPresent,
    SDL_PollEvent, SDL_GetTicks, SDL_Delay,
    SDL_Window, SDL_Renderer,
    SDL_INIT_VIDEO, SDL_WINDOWPOS_CENTERED, SDL_WINDOW_SHOWN,
    SDL_RENDERER_ACCELERATED, SDL_RENDERER_PRESENTVSYNC

fn run() -> void {
    let init_result: int = (SDL_Init SDL_INIT_VIDEO)
    if (!= init_result 0) {
        (print (SDL_GetError))
        return
    }

    let window: SDL_Window = (SDL_CreateWindow
        "NanoLang SDL Demo"
        SDL_WINDOWPOS_CENTERED SDL_WINDOWPOS_CENTERED
        640 480
        SDL_WINDOW_SHOWN)

    let renderer: SDL_Renderer = (SDL_CreateRenderer window -1
        (+ SDL_RENDERER_ACCELERATED SDL_RENDERER_PRESENTVSYNC))

    let mut running: bool = true
    while running {
        # Drain events (quit detection requires a C shim in practice)
        let mut ev: int = (SDL_PollEvent 0)
        while (!= ev 0) {
            set ev (SDL_PollEvent 0)
        }

        # Cycle background color using time
        let t: int = (SDL_GetTicks)
        let r: int = (% t 256)
        let g: int = (% (+ t 85) 256)
        let b: int = (% (+ t 170) 256)

        (SDL_SetRenderDrawColor renderer r g b 255)
        (SDL_RenderClear renderer)
        (SDL_RenderPresent renderer)
        (SDL_Delay 16)
    }

    (SDL_DestroyRenderer renderer)
    (SDL_DestroyWindow window)
    (SDL_Quit)
}

shadow run { assert true }
```

## OpenGL Context Support

SDL can also create an OpenGL context for use with the Chapter 17 modules. Set GL attributes before creating the window, then call `SDL_GL_CreateContext`:

```nano
from "modules/sdl/sdl.nano" import
    SDL_GL_SetAttribute, SDL_GL_CreateContext, SDL_GL_SwapWindow,
    SDL_GL_SetSwapInterval, SDL_GL_DeleteContext, SDL_GLContext,
    SDL_GL_CONTEXT_MAJOR_VERSION, SDL_GL_CONTEXT_MINOR_VERSION,
    SDL_GL_DOUBLEBUFFER, SDL_WINDOW_OPENGL

# Request OpenGL 3.3 with double buffering
(SDL_GL_SetAttribute SDL_GL_CONTEXT_MAJOR_VERSION 3)
(SDL_GL_SetAttribute SDL_GL_CONTEXT_MINOR_VERSION 3)
(SDL_GL_SetAttribute SDL_GL_DOUBLEBUFFER 1)

let window: SDL_Window = (SDL_CreateWindow "GL Window"
    SDL_WINDOWPOS_CENTERED SDL_WINDOWPOS_CENTERED
    800 600 SDL_WINDOW_OPENGL)

let ctx: SDL_GLContext = (SDL_GL_CreateContext window)
(SDL_GL_SetSwapInterval 1)   # vsync

# ... render with OpenGL, then:
(SDL_GL_SwapWindow window)

# Cleanup
(SDL_GL_DeleteContext ctx)
```

## API Summary

| Function | Description |
|----------|-------------|
| `SDL_Init(flags)` | Initialize SDL subsystems |
| `SDL_Quit()` | Shut down all SDL subsystems |
| `SDL_GetError()` | Return last error string |
| `SDL_GetTicks()` | Milliseconds since init |
| `SDL_Delay(ms)` | Sleep for ms milliseconds |
| `SDL_CreateWindow(title, x, y, w, h, flags)` | Open a window |
| `SDL_DestroyWindow(window)` | Close a window |
| `SDL_SetWindowTitle(window, title)` | Change window title |
| `SDL_SetWindowSize(window, w, h)` | Resize window |
| `SDL_CreateRenderer(window, index, flags)` | Create a renderer |
| `SDL_DestroyRenderer(renderer)` | Destroy a renderer |
| `SDL_SetRenderDrawColor(r, r, g, b, a)` | Set draw color |
| `SDL_RenderClear(renderer)` | Clear with draw color |
| `SDL_RenderPresent(renderer)` | Flip to screen |
| `SDL_RenderDrawPoint(renderer, x, y)` | Draw a point |
| `SDL_RenderDrawLine(renderer, x1, y1, x2, y2)` | Draw a line |
| `SDL_RenderDrawRect(renderer, rect)` | Draw rect outline |
| `SDL_RenderFillRect(renderer, rect)` | Draw filled rect |
| `SDL_RenderCopy(renderer, tex, src, dst)` | Blit texture |
| `SDL_CreateTextureFromSurface(renderer, surface)` | Surface -> texture |
| `SDL_DestroyTexture(texture)` | Free texture |
| `SDL_FreeSurface(surface)` | Free surface |
| `SDL_PollEvent(event_ptr)` | Poll event queue |
| `SDL_SetRenderDrawBlendMode(renderer, mode)` | Set blend mode |

---

**Previous:** [Chapter 16 Overview](index.html)
**Next:** [16.2 SDL_image](sdl_image.html)
