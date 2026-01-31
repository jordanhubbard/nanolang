# SDL Event Handling in NanoLang

## Overview

SDL event handling in NanoLang is centralized in `modules/sdl_helpers/sdl_helpers.c` to provide a consistent, frame-based event polling system that works reliably across all SDL examples.

## Key Design Principles

### 1. Frame-Based Event Buffering

Events are polled once per "frame" (defined as a 16ms interval, approximately 60 FPS) and buffered internally. This prevents:
- Event loss when multiple event polls occur within the same frame
- Unbounded buffer growth from unconsumed events
- Race conditions between different event types

### 2. Special Event Handling

Different event types receive different treatment:

**SDL_QUIT (Window Close)**
- **Persistent within frame**: Once received, remains true for the entire frame
- **Multiple polls supported**: Can be checked multiple times per frame safely
- **Never lost**: Critical for application shutdown

**SDL_MOUSEMOTION**
- **Coalesced**: Only the most recent motion event is kept
- **Not buffered**: Doesn't consume buffer space
- **High frequency**: Motion events can occur hundreds of times per second

**Other Events (Keyboard, Mouse Click, etc.)**
- **Consumed on first poll**: Removed from buffer when retrieved
- **FIFO order**: Events processed in order received
- **Buffer limited**: Maximum 256 events per frame

## Implementation Details

### Static State Variables

```c
static SDL_Event nl_sdl_event_buf[NL_SDL_EVENT_BUF_CAP];  // Event buffer
static int nl_sdl_event_buf_len = 0;                      // Buffer length
static uint32_t nl_sdl_event_buf_last_ticks = UINT32_MAX; // Last drain time
static int nl_sdl_events_drained_this_tick = 0;           // Drain flag
static int nl_sdl_quit_received = 0;                      // Quit flag (persistent)
static SDL_Event nl_sdl_last_mousemotion;                 // Last motion event
static int nl_sdl_has_mousemotion = 0;                    // Motion flag
```

### Event Draining Logic

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

### Quit Event Handling

```c
int64_t nl_sdl_poll_event_quit(void) {
    nl__sdl_drain_events();
    return nl_sdl_quit_received ? 1 : 0;  // Persistent within frame
}
```

## Usage Patterns

### Standard Event Loop

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

The frame-based buffering ensures that the order of event polling within a frame doesn't matter:

```nano
# Both orderings work identically
# Order 1:
let key: int = (nl_sdl_poll_keypress)
let quit: int = (nl_sdl_poll_event_quit)

# Order 2:
let quit: int = (nl_sdl_poll_event_quit)
let key: int = (nl_sdl_poll_keypress)
```

## Available Event Functions

### Critical Events
- `nl_sdl_poll_event_quit() -> int` - Returns 1 if window close requested (persistent)

### Mouse Events
- `nl_sdl_poll_mouse_click() -> int` - Returns x*10000+y for left click, -1 otherwise
- `nl_sdl_poll_mouse_up() -> int` - Returns x*10000+y for left release, -1 otherwise
- `nl_sdl_poll_mouse_state() -> int` - Returns x*10000+y if button held, -1 otherwise
- `nl_sdl_poll_mouse_motion() -> int` - Returns x*10000+y for mouse movement, -1 otherwise
- `nl_sdl_poll_mouse_wheel() -> int` - Returns scroll delta (positive=up, negative=down)

### Keyboard Events
- `nl_sdl_poll_keypress() -> int` - Returns SDL scancode if key pressed, -1 otherwise
- `nl_sdl_key_state(scancode: int) -> int` - Returns 1 if key currently held, 0 otherwise

## Common Pitfalls (Avoided by Current Design)

### ❌ Old Problem: Lost Quit Events
```c
// OLD BUGGY CODE:
// First call consumes quit event from buffer
if (nl__sdl_take_first_event(SDL_QUIT, NULL)) { ... }
// Second call finds empty buffer, returns 0
if (nl__sdl_take_first_event(SDL_QUIT, NULL)) { ... }  // FAILS!
```

### ✅ Current Solution: Persistent Quit Flag
```c
// NEW CORRECT CODE:
// Both calls check the same persistent flag
if (nl_sdl_quit_received) { ... }  // Returns 1
if (nl_sdl_quit_received) { ... }  // Returns 1 (still works!)
```

### ❌ Old Problem: Buffer Clearing Between Calls
```c
// OLD BUGGY CODE:
nl__sdl_drain_events();  // Fills buffer
nl__sdl_drain_events();  // Clears buffer if time changed!
```

### ✅ Current Solution: Once-Per-Frame Draining
```c
// NEW CORRECT CODE:
nl__sdl_drain_events();  // Fills buffer, sets drained flag
nl__sdl_drain_events();  // Returns early, preserves buffer
```

## Performance Characteristics

- **Frame rate**: Designed for 60 FPS (16ms frame time)
- **Buffer capacity**: 256 events per frame
- **Overhead**: Minimal - single timestamp check per drain call
- **Latency**: ~1 frame (16ms) maximum for event response

## Thread Safety

**Not thread-safe**. All SDL event functions must be called from the main thread. This is an SDL limitation, not a NanoLang limitation.

## Future Improvements

- [ ] Configurable frame time threshold (currently hardcoded to 16ms)
- [ ] Event priority system for critical events
- [ ] Event queue overflow handling
- [ ] Performance metrics and profiling

## See Also

- `modules/sdl_helpers/sdl_helpers.c` - Implementation
- `modules/sdl_helpers/sdl_helpers.nano` - NanoLang interface
- `modules/sdl/sdl.nano` - Core SDL bindings

