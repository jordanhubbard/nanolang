# My SDL Event Handling

## Overview

I centralize my SDL event handling in `modules/sdl_helpers/sdl_helpers.c`. I provide a consistent, frame-based event polling system that works reliably across all my SDL examples.

## My Design Principles

### 1. Frame-Based Event Buffering

I poll events once per frame. I define a frame as a 16ms interval, which is approximately 60 FPS. I buffer these events internally. This prevents:
- Event loss when multiple event polls occur within the same frame.
- Unbounded buffer growth from unconsumed events.
- Race conditions between different event types.

### 2. Special Event Handling

I treat different event types according to their needs:

**SDL_QUIT (Window Close)**
- **Persistent within frame**: Once I receive this, it remains true for the entire frame.
- **Multiple polls supported**: You can check this multiple times per frame safely.
- **Never lost**: I ensure this is preserved for your application shutdown.

**SDL_MOUSEMOTION**
- **Coalesced**: I only keep the most recent motion event.
- **Not buffered**: I do not let motion events consume my buffer space.
- **High frequency**: I handle motion events that occur hundreds of times per second without bloat.

**Other Events (Keyboard, Mouse Click, etc.)**
- **Consumed on first poll**: I remove these from my buffer when you retrieve them.
- **FIFO order**: I process events in the order I received them.
- **Buffer limited**: I allow a maximum of 256 events per frame.

## Implementation Details

### My Static State Variables

```c
static SDL_Event nl_sdl_event_buf[NL_SDL_EVENT_BUF_CAP];  // Event buffer
static int nl_sdl_event_buf_len = 0;                      // Buffer length
static uint32_t nl_sdl_event_buf_last_ticks = UINT32_MAX; // Last drain time
static int nl_sdl_events_drained_this_tick = 0;           // Drain flag
static int nl_sdl_quit_received = 0;                      // Quit flag (persistent)
static SDL_Event nl_sdl_last_mousemotion;                 // Last motion event
static int nl_sdl_has_mousemotion = 0;                    // Motion flag
```

### My Event Draining Logic

```c
static void nl__sdl_drain_events(void) {
    uint32_t now = (uint32_t)SDL_GetTicks();
    int time_advanced = (now >= nl_sdl_event_buf_last_ticks + 16);
    
    if (time_advanced) {
        // New frame - reset all state
        nl_sdl_event_buf_len = 0;
        nl_sdl_events_drained_this_tick = 0;
        nl_sdl_quit_received = 0;  // Reset quit flag
    }
    
    if (nl_sdl_events_drained_this_tick) {
        return;  // Already drained, reuse buffered events
    }
    
    nl_sdl_events_drained_this_tick = 1;
    
    // Poll all events
    SDL_PumpEvents();
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            nl_sdl_quit_received = 1;  // Mark quit as received
            continue;  // Don't buffer
        }
        // ... handle other event types
    }
}
```

### My Quit Event Handling

```c
int64_t nl_sdl_poll_event_quit(void) {
    nl__sdl_drain_events();
    return nl_sdl_quit_received ? 1 : 0;  // Persistent within frame
}
```

## My Usage Patterns

### My Standard Event Loop

```nano
while running {
    # Check quit FIRST - it's persistent and safe to check multiple times
    if (== (nl_sdl_poll_event_quit) 1) {
        set running false
    } else {}
    
    # Check other events (these are consumed)
    let key: int = (nl_sdl_poll_keypress)
    let mouse: int = (nl_sdl_poll_mouse_click)
    
    # Render frame
    # ...
    
    unsafe { (SDL_Delay 16) }  # Approximately 60 FPS
}
```

### Multiple Quit Checks

```nano
# Safe - all will return 1 if quit was received
let quit1: int = (nl_sdl_poll_event_quit)
let quit2: int = (nl_sdl_poll_event_quit)
let quit3: int = (nl_sdl_poll_event_quit)
```

### Event Order Independence

My frame-based buffering ensures that the order of event polling within a frame does not matter:

```nano
# Both orderings work identically
# Order 1:
let key: int = (nl_sdl_poll_keypress)
let quit: int = (nl_sdl_poll_event_quit)

# Order 2:
let quit: int = (nl_sdl_poll_event_quit)
let key: int = (nl_sdl_poll_keypress)
```

## My Available Event Functions

### Critical Events
- `nl_sdl_poll_event_quit() -> int` - I return 1 if a window close was requested. This is persistent within the frame.

### Mouse Events
- `nl_sdl_poll_mouse_click() -> int` - I return x*10000+y for a left click, or -1.
- `nl_sdl_poll_mouse_up() -> int` - I return x*10000+y for a left release, or -1.
- `nl_sdl_poll_mouse_state() -> int` - I return x*10000+y if the button is held, or -1.
- `nl_sdl_poll_mouse_motion() -> int` - I return x*10000+y for mouse movement, or -1.
- `nl_sdl_poll_mouse_wheel() -> int` - I return the scroll delta. Positive is up, negative is down.

### Keyboard Events
- `nl_sdl_poll_keypress() -> int` - I return the SDL scancode if a key was pressed, or -1.
- `nl_sdl_key_state(scancode: int) -> int` - I return 1 if the key is currently held, or 0.

## Common Pitfalls I Avoid

### Problem: Lost Quit Events
```c
// BUGGY CODE:
// First call consumes quit event from buffer
if (nl__sdl_take_first_event(SDL_QUIT, NULL)) { ... }
// Second call finds empty buffer, returns 0
if (nl__sdl_take_first_event(SDL_QUIT, NULL)) { ... }  // FAILS!
```

### My Solution: Persistent Quit Flag
```c
// CORRECT CODE:
// Both calls check the same persistent flag
if (nl_sdl_quit_received) { ... }  // Returns 1
if (nl_sdl_quit_received) { ... }  // Returns 1 (still works!)
```

### Problem: Buffer Clearing Between Calls
```c
// BUGGY CODE:
nl__sdl_drain_events();  // Fills buffer
nl__sdl_drain_events();  // Clears buffer if time changed!
```

### My Solution: Once-Per-Frame Draining
```c
// CORRECT CODE:
nl__sdl_drain_events();  // Fills buffer, sets drained flag
nl__sdl_drain_events();  // Returns early, preserves buffer
```

## My Performance Characteristics

- **Frame rate**: I am designed for 60 FPS with a 16ms frame time.
- **Buffer capacity**: I allow 256 events per frame.
- **Overhead**: I use a single timestamp check per drain call.
- **Latency**: I have a maximum event response latency of approximately one frame (16ms).

## Thread Safety

I am **not thread-safe**. You must call all my SDL event functions from the main thread. This is an SDL limitation, not my own.

## My Future Improvements

- [ ] Configurable frame time threshold.
- [ ] Event priority system for critical events.
- [ ] Event queue overflow handling.
- [ ] Performance metrics and profiling.

## See Also

- `modules/sdl_helpers/sdl_helpers.c` - My implementation.
- `modules/sdl_helpers/sdl_helpers.nano` - My NanoLang interface.
- `modules/sdl/sdl.nano` - My core SDL bindings.


