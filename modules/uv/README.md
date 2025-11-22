# libuv Module for nanolang

Cross-platform asynchronous I/O library that powers Node.js event loop.

## Installation

**macOS:**
```bash
brew install libuv
```

**Ubuntu/Debian:**
```bash
sudo apt install libuv1-dev
```

## Usage

```nano
import "modules/uv/uv.nano"

fn main() -> int {
    # Get system information
    (print "libuv version: ")
    (println (nl_uv_version_string))
    
    (print "CPUs: ")
    (println (nl_uv_cpu_count))
    
    (print "Hostname: ")
    (println (nl_uv_os_gethostname))
    
    # Create event loop
    let loop: int = (nl_uv_default_loop)
    
    # Get current time in milliseconds
    let now: int = (nl_uv_now loop)
    (print "Current time (ms): ")
    (println now)
    
    # High-resolution timer
    let hrtime: int = (nl_uv_hrtime)
    (print "High-res time (ns): ")
    (println hrtime)
    
    return 0
}

shadow main {
    # Skip - uses extern functions
}
```

## Features

- **Event loop**: Node.js-style async event processing
- **System info**: CPU count, memory, hostname, process IDs
- **High-resolution timers**: Nanosecond precision timing
- **Cross-platform**: Works on macOS, Linux, Windows
- **Non-blocking I/O**: Foundation for async operations

## Example

See `examples/uv_example.nano` for comprehensive usage examples.

## API Reference

### Event Loop
- `nl_uv_default_loop() -> int` - Get default event loop
- `nl_uv_loop_new() -> int` - Create new event loop
- `nl_uv_loop_close(loop: int) -> int` - Close loop
- `nl_uv_run(loop: int, mode: int) -> int` - Run loop (mode: 0=default, 1=once, 2=nowait)
- `nl_uv_stop(loop: int) -> void` - Stop loop
- `nl_uv_loop_alive(loop: int) -> int` - Check if loop has active handles

### Timing
- `nl_uv_now(loop: int) -> int` - Current time in milliseconds
- `nl_uv_hrtime() -> int` - High-resolution time in nanoseconds
- `nl_uv_sleep(msec: int) -> void` - Blocking sleep

### System Information
- `nl_uv_version_string() -> string` - libuv version
- `nl_uv_cpu_count() -> int` - Number of CPUs
- `nl_uv_get_total_memory() -> int` - Total RAM in bytes
- `nl_uv_get_free_memory() -> int` - Free RAM in bytes
- `nl_uv_loadavg_1min() -> int` - 1-minute load average * 100
- `nl_uv_os_getpid() -> int` - Current process ID
- `nl_uv_os_getppid() -> int` - Parent process ID
- `nl_uv_cwd() -> string` - Current working directory
- `nl_uv_os_gethostname() -> string` - System hostname

### Error Handling
- `nl_uv_strerror(err: int) -> string` - Get error message
- `nl_uv_err_name(err: int) -> string` - Get error name

## Use Cases

- Async I/O operations
- System monitoring and diagnostics
- High-precision timing
- Cross-platform utilities
- Foundation for network servers
