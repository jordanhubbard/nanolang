# Chapter 16: Graphics Fundamentals (SDL)

**2D graphics, windows, events, and rendering with SDL.**

SDL (Simple DirectMedia Layer) provides cross-platform access to graphics, audio, and input devices.

## 16.1 Window Creation

```nano
from "modules/sdl/sdl.nano" import init, create_window, destroy_window, Window

fn create_app_window() -> Window {
    (init)
    let window: Window = (create_window "My App" 800 600)
    return window
}

shadow create_app_window {
    # Would test with actual SDL
    assert true
}
```

## 16.2 Drawing Primitives

```nano
from "modules/sdl/sdl.nano" import draw_rect, draw_circle, draw_line, clear_screen, present

fn draw_shapes(window: Window) -> void {
    (clear_screen window)
    (draw_rect window 100 100 200 150)
    (draw_circle window 400 300 50)
    (draw_line window 0 0 800 600)
    (present window)
}

shadow draw_shapes {
    assert true
}
```

## 16.3 Event Loop

```nano
from "modules/sdl/sdl.nano" import poll_event, Event, EVENT_QUIT

fn run_event_loop(window: Window) -> void {
    let mut running: bool = true
    
    while running {
        let event: Event = (poll_event)
        if (== event.type EVENT_QUIT) {
            set running false
        }
        
        (clear_screen window)
        (present window)
        (delay 16)
    }
}

shadow run_event_loop {
    assert true
}
```

## 16.4 Colors and Rendering

```nano
from "modules/sdl/sdl.nano" import set_draw_color, clear_screen

fn render_with_color(window: Window) -> void {
    # Set color (R, G, B, A)
    (set_draw_color window 255 0 0 255)  # Red
    (clear_screen window)
    (present window)
}

shadow render_with_color {
    assert true
}
```

## Summary

SDL provides:
- ✅ Window management
- ✅ 2D drawing primitives
- ✅ Event handling
- ✅ Color rendering

---

**Previous:** [Chapter 15: Web & Networking](../15_web_networking/index.md)  
**Next:** [Chapter 17: OpenGL Graphics](../17_opengl_graphics/index.md)
