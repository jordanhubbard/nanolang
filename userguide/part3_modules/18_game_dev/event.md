# 18.1 event — Event System

**Drive your game loop with a high-performance async event system.**

The `event` module wraps [libevent](https://libevent.org/), a portable asynchronous event notification library. In NanoLang games it serves as the heartbeat of the application: you create an event base, register a timer that fires every frame, and dispatch the event loop. When you're ready to quit, you break out of the loop and clean up.

## Quick Start

```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_dispatch,
                                        nl_event_base_free, nl_evtimer_new,
                                        nl_evtimer_add_timeout, nl_event_free,
                                        nl_event_get_version

fn main() -> int {
    # Check the libevent version at startup
    let ver: string = (nl_event_get_version)
    (println (+ "libevent version: " ver))

    # Create the event loop base
    let base: int = (nl_event_base_new)

    # Create a one-shot timer, fire in 1 second
    let timer: int = (nl_evtimer_new base)
    (nl_evtimer_add_timeout timer 1)

    # Block until all events fire (or loopbreak/loopexit is called)
    (nl_event_base_dispatch base)

    # Cleanup
    (nl_event_free timer)
    (nl_event_base_free base)

    return 0
}

shadow main { assert true }
```

## Concepts

### The Event Base

The event base is the core object — it holds the state of the event loop and keeps track of all registered events. All other functions take an `int` handle that refers to this base.

```nano
let base: int = (nl_event_base_new)   # allocate
# ... register events, dispatch ...
(nl_event_base_free base)             # always free when done
```

The base is backed by the best available I/O notification mechanism on the current platform (kqueue on macOS, epoll on Linux, etc.). You can query the method in use:

```nano
let method: string = (nl_event_base_get_method base)
(println (+ "Using backend: " method))
```

### Timer Events

Timer events fire after a delay expressed in whole seconds. They are the most common event type in games because they give you the regular tick you need to advance your simulation.

```nano
let timer: int = (nl_evtimer_new base)   # create timer attached to base
(nl_evtimer_add_timeout timer 0)         # arm it: fire after 0 seconds
# ... timer fires, your callback runs ...
(nl_event_del timer)                     # disarm without freeing
(nl_event_free timer)                    # free the event object
```

> **Note on granularity:** `nl_evtimer_add_timeout` takes whole seconds. For sub-second game ticks you typically call `nl_event_base_dispatch` in a tight loop and use `nl_event_sleep` for fractional delays, or integrate with your OS's high-resolution timer.

### Dispatching the Loop

Two functions drive the loop:

| Function | Behaviour |
|---|---|
| `nl_event_base_dispatch(base)` | Run until no more events are registered. Blocks. |
| `nl_event_base_loop(base, flags)` | Run with flags (e.g. `EVLOOP_ONCE = 1` for a single pass). |

For a game that runs until the player quits, `nl_event_base_dispatch` is the right choice:

```nano
(nl_event_base_dispatch base)   # blocks here until the loop exits
```

### Exiting the Loop

From inside a callback (or from any thread) you can stop the loop two ways:

```nano
# Exit after a timeout (in seconds); 0 means "exit now at next opportunity"
(nl_event_base_loopexit base 0)

# Break immediately, even if events are pending
(nl_event_base_loopbreak base)
```

`loopbreak` is the right choice inside a game-quit handler because it stops the loop without waiting for the current dispatch cycle to finish.

### Sleeping Inside the Loop

For simple timing without a dedicated timer event:

```nano
(nl_event_sleep base 1)   # sleep 1 second, keeping the base alive
```

This is useful for rate-limiting a polling loop.

## API Reference

### Event Base Management

```
nl_event_base_new() -> int
```
Allocate and return a new event base. Returns a handle (positive int) on success.

```
nl_event_base_free(base: int) -> void
```
Free all resources associated with the event base. Call this after dispatch returns.

```
nl_event_base_dispatch(base: int) -> int
```
Enter the event loop and block until there are no more pending events, or until `loopbreak`/`loopexit` is called. Returns 0 on normal exit, -1 on error.

```
nl_event_base_loop(base: int, flags: int) -> int
```
Like `dispatch`, but accepts flags. Pass `1` for EVLOOP_ONCE (process one batch then return) or `2` for EVLOOP_NONBLOCK (process only already-pending events).

```
nl_event_base_loopexit(base: int, timeout_secs: int) -> int
```
Schedule the event loop to exit after `timeout_secs` seconds. Pass `0` to exit as soon as the current callback returns.

```
nl_event_base_loopbreak(base: int) -> int
```
Break out of the event loop immediately, even if events are waiting. Returns `0` on success.

### Event Base Information

```
nl_event_base_get_method(base: int) -> string
```
Return the name of the I/O notification method in use (e.g. `"kqueue"`, `"epoll"`).

```
nl_event_base_get_num_events(base: int) -> int
```
Return the number of events currently registered with the base.

```
nl_event_get_version() -> string
```
Return the libevent version string (e.g. `"2.1.12-stable"`).

```
nl_event_get_version_number() -> int
```
Return the libevent version as a packed integer for programmatic comparison.

### Timer Events

```
nl_evtimer_new(base: int) -> int
```
Create a new timer event attached to `base`. Returns a timer handle.

```
nl_evtimer_add_timeout(event: int, timeout_secs: int) -> int
```
Arm the timer to fire after `timeout_secs` seconds. Call again inside the callback to create a repeating timer. Returns `0` on success.

```
nl_event_del(event: int) -> int
```
Disarm the event (stop it from firing) without freeing it.

```
nl_event_free(event: int) -> void
```
Free the event object. Always call this after you are done with a timer.

### Utilities

```
nl_event_enable_debug_mode() -> void
```
Enable libevent's internal debug mode. Call before creating any event base. Useful for diagnosing event lifecycle issues.

```
nl_event_sleep(base: int, seconds: int) -> int
```
Suspend execution for `seconds` seconds while keeping the event base alive.

## Examples

### Repeating Timer (Fixed-Rate Game Loop)

The key pattern for a fixed-rate game loop is to re-arm the timer at the end of each callback:

```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_dispatch,
                                        nl_event_base_free, nl_event_base_loopbreak,
                                        nl_evtimer_new, nl_evtimer_add_timeout,
                                        nl_event_free

let mut tick_count: int = 0
let mut game_timer: int = 0
let mut game_base: int = 0

fn on_tick() -> void {
    set tick_count (+ tick_count 1)
    (println (+ "tick " (int_to_string tick_count)))

    if (>= tick_count 5) {
        (nl_event_base_loopbreak game_base)
    } else {
        # Re-arm the timer for the next tick (1 second intervals)
        (nl_evtimer_add_timeout game_timer 1)
    }
}

shadow on_tick {
    set tick_count 0
    (on_tick)
    assert (== tick_count 1)
}

fn main() -> int {
    set game_base (nl_event_base_new)
    set game_timer (nl_evtimer_new game_base)
    (nl_evtimer_add_timeout game_timer 1)

    (nl_event_base_dispatch game_base)

    (nl_event_free game_timer)
    (nl_event_base_free game_base)
    return 0
}

shadow main { assert true }
```

### Checking Available Events

```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_get_num_events,
                                        nl_evtimer_new, nl_event_base_free, nl_event_free,
                                        nl_event_base_get_method

fn show_event_info() -> void {
    let base: int = (nl_event_base_new)
    let method: string = (nl_event_base_get_method base)
    (println (+ "Backend: " method))

    let t: int = (nl_evtimer_new base)
    let n: int = (nl_event_base_get_num_events base)
    (println (+ "Registered events: " (int_to_string n)))

    (nl_event_free t)
    (nl_event_base_free base)
}

shadow show_event_info {
    (show_event_info)
}
```

---

**Previous:** [Chapter 18 Overview](index.html)
**Next:** [18.2 vector2d](vector2d.html)
