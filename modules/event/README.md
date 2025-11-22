# libevent Module for nanolang

Asynchronous event notification library for building high-performance network servers.

## Installation

**macOS:**
```bash
brew install libevent
```

**Ubuntu/Debian:**
```bash
sudo apt install libevent-dev
```

## Usage

```nano
import "modules/event/event.nano"

fn main() -> int {
    # Get version
    (print "libevent version: ")
    (println (nl_event_get_version))
    
    # Create event base
    let base: int = (nl_event_base_new)
    
    # Get backend method (epoll, kqueue, select, etc.)
    (print "Backend: ")
    (println (nl_event_base_get_method base))
    
    # Get number of active events
    let num_events: int = (nl_event_base_get_num_events base)
    (print "Active events: ")
    (println num_events)
    
    # Clean up
    (nl_event_base_free base)
    
    return 0
}

shadow main {
    # Skip - uses extern functions
}
```

## Features

- **Event loop**: Efficient async event processing
- **Multiple backends**: epoll, kqueue, select, poll
- **Timer events**: Schedule timed operations
- **High performance**: Optimized for network servers
- **Cross-platform**: macOS, Linux, BSD support

## Example

See `examples/event_example.nano` for comprehensive usage examples.

## API Reference

### Event Base (Event Loop)
- `nl_event_base_new() -> int` - Create event loop
- `nl_event_base_free(base: int) -> void` - Free event loop
- `nl_event_base_dispatch(base: int) -> int` - Run event loop (blocks)
- `nl_event_base_loop(base: int, flags: int) -> int` - Run with flags (0=block, 1=nonblock, 2=once)
- `nl_event_base_loopexit(base: int, timeout: int) -> int` - Exit after timeout
- `nl_event_base_loopbreak(base: int) -> int` - Exit immediately

### Event Base Information
- `nl_event_get_version() -> string` - libevent version
- `nl_event_base_get_method(base: int) -> string` - Backend name (epoll/kqueue/etc)
- `nl_event_base_get_num_events(base: int) -> int` - Number of active events

### Timer Events
- `nl_evtimer_new(base: int) -> int` - Create timer event
- `nl_event_free(event: int) -> void` - Free event
- `nl_evtimer_add_timeout(event: int, secs: int) -> int` - Schedule timer
- `nl_event_del(event: int) -> int` - Remove event

### Utilities
- `nl_event_enable_debug_mode() -> void` - Enable debugging
- `nl_event_sleep(base: int, seconds: int) -> int` - Event-based sleep

## Notes

**Backend Methods:**
- **epoll** (Linux) - Most efficient on Linux
- **kqueue** (macOS/BSD) - Most efficient on macOS
- **poll** - Portable fallback
- **select** - Universal fallback

**Event Loop Modes:**
- `0` = Block until event occurs
- `1` = Non-blocking (return immediately)
- `2` = Run once then return

## Use Cases

- Network servers
- High-performance I/O
- Async event processing
- Real-time systems
- Game servers
