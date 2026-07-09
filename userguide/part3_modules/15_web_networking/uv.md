# 15.3 uv - Async I/O (libuv bindings)

**Asynchronous I/O, event loops, timers, and system information via libuv.**

The `uv` module exposes NanoLang bindings to [libuv](https://libuv.org/), the cross-platform async I/O library that powers Node.js. Use it when you need an event-driven programming model, high-resolution timing, or access to OS-level information such as CPU count, memory, hostname, and process IDs. All functions are `extern` and must be called inside `unsafe` blocks.

libuv's core concept is the **event loop**: a loop that polls for I/O events, fires callbacks, and advances timers. In NanoLang you acquire the default loop, run work that registers I/O or timer handles on it, and then drive the loop with `nl_uv_run`.

## Installation

libuv must be available on the system (`libuv1-dev` on Debian/Ubuntu, `libuv` via Homebrew on macOS).

Import the functions you need:

```nano
from "modules/uv/uv.nano" import
    nl_uv_version_string, nl_uv_version,
    nl_uv_default_loop, nl_uv_loop_new, nl_uv_loop_close,
    nl_uv_run, nl_uv_stop, nl_uv_loop_alive, nl_uv_loop_get_active_handles,
    nl_uv_now, nl_uv_update_time, nl_uv_hrtime,
    nl_uv_sleep, nl_uv_backend_timeout,
    nl_uv_strerror, nl_uv_err_name, nl_uv_translate_sys_error,
    nl_uv_get_total_memory, nl_uv_get_free_memory, nl_uv_cpu_count,
    nl_uv_loadavg_1min, nl_uv_os_getpid, nl_uv_os_getppid,
    nl_uv_cwd, nl_uv_os_gethostname
```

## Quick Start

```nano
from "modules/uv/uv.nano" import nl_uv_version_string, nl_uv_default_loop,
    nl_uv_cpu_count, nl_uv_get_total_memory, nl_uv_os_gethostname

fn print_system_info() -> void {
    unsafe {
        let ver: string = (nl_uv_version_string)
        let cpus: int = (nl_uv_cpu_count)
        let mem: int = (nl_uv_get_total_memory)
        let host: string = (nl_uv_os_gethostname)

        (println (+ "libuv version: " ver))
        (println (+ "Host: " host))
        (println (+ "CPUs: " (int_to_string cpus)))
        (println (+ "Total memory (bytes): " (int_to_string mem)))
    }
    return
}

shadow print_system_info { assert true }
```

---

## API Reference

### Version Information

#### `extern fn nl_uv_version_string() -> string`

Return the libuv version as a human-readable string, e.g. `"1.46.0"`.

#### `extern fn nl_uv_version() -> int`

Return the libuv version packed as an integer: `(major << 16) | (minor << 8) | patch`. For example, version 1.46.0 returns `0x012e00`.

```nano
unsafe {
    let ver: string = (nl_uv_version_string)
    let ver_int: int = (nl_uv_version)
    (println (+ "libuv " ver))
}
```

---

### Event Loop Management

The event loop is the central dispatch mechanism. A single default loop is available per process, or you can create isolated loops for specific subsystems.

#### `extern fn nl_uv_default_loop() -> int`

Return a handle to the process-wide default event loop. The returned handle is valid for the lifetime of the process. Returns `0` on failure (extremely rare).

```nano
unsafe {
    let loop: int = (nl_uv_default_loop)
}
```

#### `extern fn nl_uv_loop_new() -> int`

Allocate and initialize a new, independent event loop. Use this when you need a loop that is separate from the default. Returns a non-zero handle on success, `0` on allocation failure. Must be closed with `nl_uv_loop_close` when done.

```nano
unsafe {
    let loop: int = (nl_uv_loop_new)
    if (== loop 0) {
        (println "Failed to create event loop")
    } else {
        # ... use loop ...
        (nl_uv_loop_close loop)
    }
}
```

#### `extern fn nl_uv_loop_close(_loop: int) -> int`

Close and free a loop created with `nl_uv_loop_new`. Returns `0` on success. Will fail (non-zero) if there are still active handles on the loop — stop them first.

#### `extern fn nl_uv_run(_loop: int, _mode: int) -> int`

Run the event loop. The `_mode` parameter controls how the loop runs:

| Mode value | Meaning |
|------------|---------|
| `0` | `UV_RUN_DEFAULT` — run until there are no more active handles or requests |
| `1` | `UV_RUN_ONCE` — poll for I/O once; block if there are no pending callbacks |
| `2` | `UV_RUN_NOWAIT` — poll for I/O once without blocking; return immediately |

Returns non-zero if there are still active handles after the call (relevant for modes `1` and `2`).

```nano
unsafe {
    let loop: int = (nl_uv_default_loop)
    # Register handles/timers on the loop here
    (nl_uv_run loop 0)   # Run until loop is empty
}
```

#### `extern fn nl_uv_stop(_loop: int) -> void`

Stop the event loop. After all currently queued callbacks have been processed, `nl_uv_run` returns. Call this from within a callback to shut down the loop cleanly.

#### `extern fn nl_uv_loop_alive(_loop: int) -> int`

Returns non-zero if the loop has active handles or requests. Use this to check whether there is still pending work before deciding to exit.

```nano
unsafe {
    let loop: int = (nl_uv_default_loop)
    let alive: int = (nl_uv_loop_alive loop)
    if (!= alive 0) {
        (println "Loop has active work")
    } else {
        (println "Loop is idle")
    }
}
```

#### `extern fn nl_uv_loop_get_active_handles(_loop: int) -> int`

Return the number of active handles currently registered on the loop. Useful for diagnostics.

---

### Time Functions

#### `extern fn nl_uv_now(_loop: int) -> int`

Return the cached time in milliseconds. libuv caches the current time to avoid repeated system calls; call `nl_uv_update_time` to refresh it.

#### `extern fn nl_uv_update_time(_loop: int) -> void`

Update the loop's cached time by calling `gettimeofday` (or equivalent). Normally called automatically by `nl_uv_run` between iterations.

#### `extern fn nl_uv_hrtime() -> int`

Return a high-resolution timestamp in nanoseconds. This is a monotonic clock suitable for measuring elapsed time. Not related to wall-clock time.

```nano
unsafe {
    let start_ns: int = (nl_uv_hrtime)
    # ... do work ...
    let end_ns: int = (nl_uv_hrtime)
    let elapsed_ms: int = (/ (- end_ns start_ns) 1000000)
    (println (+ "Elapsed: " (+ (int_to_string elapsed_ms) "ms")))
}
```

#### `extern fn nl_uv_sleep(_msec: int) -> void`

Sleep the current thread for `_msec` milliseconds. This is a blocking sleep — it does not yield to the event loop. Use sparingly; prefer timer-based approaches in event-driven code.

```nano
unsafe {
    (println "Waiting 500ms...")
    (nl_uv_sleep 500)
    (println "Done")
}
```

#### `extern fn nl_uv_backend_timeout(_loop: int) -> int`

Return the number of milliseconds the loop should block waiting for I/O events during the next `nl_uv_run` call. Returns `0` if the loop should not block, `-1` if it should block indefinitely.

---

### Error Handling

libuv returns negative integers for error codes. Use these functions to translate them into human-readable form.

#### `extern fn nl_uv_strerror(_err: int) -> string`

Return a descriptive string for a libuv error code (e.g. `"connection refused"` for `UV_ECONNREFUSED`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `_err` | `int` | A negative libuv error code |

**Returns:** `string` — human-readable error description.

#### `extern fn nl_uv_err_name(_err: int) -> string`

Return the symbolic name for a libuv error code (e.g. `"ECONNREFUSED"`).

#### `extern fn nl_uv_translate_sys_error(_sys_errno: int) -> int`

Translate a POSIX `errno` value into a libuv error code. Useful when integrating with C functions that return system errors.

```nano
unsafe {
    let rc: int = (nl_uv_loop_close loop)
    if (!= rc 0) {
        let name: string = (nl_uv_err_name rc)
        let msg: string = (nl_uv_strerror rc)
        (println (+ name (+ ": " msg)))
    } else {
        (print "")
    }
}
```

---

### System Information

These functions expose OS-level information without requiring any loop handle.

#### `extern fn nl_uv_get_total_memory() -> int`

Return total physical memory in bytes.

#### `extern fn nl_uv_get_free_memory() -> int`

Return available (free) physical memory in bytes.

#### `extern fn nl_uv_cpu_count() -> int`

Return the number of logical CPU cores.

#### `extern fn nl_uv_loadavg_1min() -> int`

Return the 1-minute load average, scaled as an integer (multiply by the platform scale factor to get the actual value). Availability varies by platform.

#### `extern fn nl_uv_os_getpid() -> int`

Return the process ID of the current process.

#### `extern fn nl_uv_os_getppid() -> int`

Return the process ID of the parent process.

#### `extern fn nl_uv_cwd() -> string`

Return the current working directory of the process.

#### `extern fn nl_uv_os_gethostname() -> string`

Return the hostname of the machine.

```nano
unsafe {
    let pid: int = (nl_uv_os_getpid)
    let host: string = (nl_uv_os_gethostname)
    let cwd: string = (nl_uv_cwd)
    let total_mem: int = (nl_uv_get_total_memory)
    let free_mem: int = (nl_uv_get_free_memory)
    let cpus: int = (nl_uv_cpu_count)

    (println (+ "PID: " (int_to_string pid)))
    (println (+ "Host: " host))
    (println (+ "CWD: " cwd))
    (println (+ "CPUs: " (int_to_string cpus)))
    (println (+ "Memory: " (+ (int_to_string free_mem) (+ "/" (int_to_string total_mem)))))
}
```

---

## Examples

### Example 1: Basic Event Loop

```nano
from "modules/uv/uv.nano" import nl_uv_default_loop, nl_uv_run, nl_uv_loop_alive,
    nl_uv_version_string

fn run_loop() -> int {
    unsafe {
        let loop: int = (nl_uv_default_loop)
        (println (+ "libuv " (nl_uv_version_string)))

        # With no handles registered, the loop exits immediately
        let alive: int = (nl_uv_loop_alive loop)
        if (!= alive 0) {
            (nl_uv_run loop 0)
        } else {
            (println "No active handles, loop would exit immediately")
        }

        return 0
    }
}

shadow run_loop { assert true }
```

### Example 2: High-Resolution Benchmarking

```nano
from "modules/uv/uv.nano" import nl_uv_hrtime

fn benchmark_string_concat(n: int) -> int {
    unsafe {
        let start: int = (nl_uv_hrtime)

        let mut result: string = ""
        for i in (range 0 n) {
            set result (+ result "x")
        }

        let end: int = (nl_uv_hrtime)
        let elapsed_us: int = (/ (- end start) 1000)
        (println (+ "Concatenated " (+ (int_to_string n) (+ " times in " (+ (int_to_string elapsed_us) "us")))))
        return elapsed_us
    }
}

shadow benchmark_string_concat {
    unsafe {
        let us: int = (benchmark_string_concat 100)
        assert (>= us 0)
    }
}
```

### Example 3: System Information Report

```nano
from "modules/uv/uv.nano" import nl_uv_os_getpid, nl_uv_os_getppid, nl_uv_os_gethostname,
    nl_uv_cwd, nl_uv_cpu_count, nl_uv_get_total_memory, nl_uv_get_free_memory,
    nl_uv_version_string

fn print_process_info() -> void {
    unsafe {
        (println "=== Process Information ===")
        (println (+ "PID:  " (int_to_string (nl_uv_os_getpid))))
        (println (+ "PPID: " (int_to_string (nl_uv_os_getppid))))
        (println (+ "CWD:  " (nl_uv_cwd)))
        (println (+ "Host: " (nl_uv_os_gethostname)))
        (println "")
        (println "=== Hardware ===")
        (println (+ "CPUs:       " (int_to_string (nl_uv_cpu_count))))
        (println (+ "Total RAM:  " (int_to_string (/ (nl_uv_get_total_memory) 1048576)) " MB"))
        (println (+ "Free RAM:   " (int_to_string (/ (nl_uv_get_free_memory) 1048576)) " MB"))
        (println "")
        (println (+ "libuv: " (nl_uv_version_string)))
    }
    return
}

shadow print_process_info { assert true }
```

### Example 4: Throttled Loop with Sleep

```nano
from "modules/uv/uv.nano" import nl_uv_sleep, nl_uv_hrtime, nl_uv_now, nl_uv_default_loop

fn poll_until_done(max_iterations: int, interval_ms: int) -> int {
    let mut iterations: int = 0
    let mut done: bool = false
    unsafe {
        let start: int = (nl_uv_hrtime)

        while (and (not done) (< iterations max_iterations)) {
            # Simulate checking a condition
            set iterations (+ iterations 1)
            if (== iterations max_iterations) {
                set done true
            } else {
                (nl_uv_sleep interval_ms)
            }
        }

        let end: int = (nl_uv_hrtime)
        let elapsed_ms: int = (/ (- end start) 1000000)
        (println (+ "Finished after " (+ (int_to_string iterations) (+ " iterations, " (+ (int_to_string elapsed_ms) "ms")))))
    }
    return iterations
}

shadow poll_until_done {
    unsafe {
        let count: int = (poll_until_done 3 1)
        assert (== count 3)
    }
}
```

### Example 5: Custom Event Loop with Error Handling

```nano
from "modules/uv/uv.nano" import nl_uv_loop_new, nl_uv_loop_close, nl_uv_run,
    nl_uv_loop_alive, nl_uv_loop_get_active_handles,
    nl_uv_strerror, nl_uv_err_name

fn run_isolated_loop() -> bool {
    unsafe {
        let loop: int = (nl_uv_loop_new)
        if (== loop 0) {
            (println "Failed to create event loop")
            return false
        } else {
            (print "")
        }

        let handles: int = (nl_uv_loop_get_active_handles loop)
        (println (+ "Active handles before run: " (int_to_string handles)))

        # Run the loop (exits immediately since no handles are registered)
        (nl_uv_run loop 0)

        let rc: int = (nl_uv_loop_close loop)
        if (!= rc 0) {
            let name: string = (nl_uv_err_name rc)
            let msg: string = (nl_uv_strerror rc)
            (println (+ "Loop close error: " (+ name (+ " - " msg))))
            return false
        } else {
            (print "")
        }

        return true
    }
}

shadow run_isolated_loop {
    unsafe {
        assert (run_isolated_loop)
    }
}
```

### Example 6: Using uv with http_server for Diagnostics

```nano
from "modules/uv/uv.nano" import nl_uv_os_getpid, nl_uv_os_gethostname,
    nl_uv_cpu_count, nl_uv_get_total_memory, nl_uv_get_free_memory,
    nl_uv_version_string, nl_uv_hrtime
from "modules/http_server/http_server.nano" import send_ok_json, HttpRequest, HttpResponse
from "modules/std/json/json.nano" import new_object, new_string, new_int, object_set, stringify, free, Json

let mut g_start_time: int = 0

fn init_diagnostics() -> void {
    unsafe {
        set g_start_time (nl_uv_hrtime)
    }
    return
}

fn handle_diagnostics(req: HttpRequest, res: HttpResponse) -> void {
    unsafe {
        let now: int = (nl_uv_hrtime)
        let uptime_sec: int = (/ (- now g_start_time) 1000000000)

        let obj: Json = (new_object)
        (object_set obj "pid" (new_int (nl_uv_os_getpid)))
        (object_set obj "host" (new_string (nl_uv_os_gethostname)))
        (object_set obj "cpus" (new_int (nl_uv_cpu_count)))
        (object_set obj "free_memory_mb" (new_int (/ (nl_uv_get_free_memory) 1048576)))
        (object_set obj "total_memory_mb" (new_int (/ (nl_uv_get_total_memory) 1048576)))
        (object_set obj "uptime_seconds" (new_int uptime_sec))
        (object_set obj "libuv_version" (new_string (nl_uv_version_string)))

        let json: string = (stringify obj)
        (free obj)
        (send_ok_json res json)
    }
    return
}

shadow handle_diagnostics { assert true }
```

---

## Common Pitfalls

### Pitfall 1: Forgetting unsafe blocks

Every `extern` function in the `uv` module requires an `unsafe` block. There are no safe wrappers in this module.

```nano
# WRONG — compile error
let cpus: int = (nl_uv_cpu_count)

# CORRECT
unsafe {
    let cpus: int = (nl_uv_cpu_count)
}
```

### Pitfall 2: Closing the default loop

The default loop (obtained via `nl_uv_default_loop`) is process-global and must not be closed with `nl_uv_loop_close`. Only loops created with `nl_uv_loop_new` should be closed.

```nano
# WRONG
let loop: int = (nl_uv_default_loop)
(nl_uv_loop_close loop)   # Do not close the default loop

# CORRECT
let loop: int = (nl_uv_loop_new)
# ... use loop ...
(nl_uv_loop_close loop)
```

### Pitfall 3: Closing a loop with active handles

`nl_uv_loop_close` fails if there are still active handles on the loop. Stop all handles before closing.

```nano
# Check for active handles before closing
unsafe {
    let handles: int = (nl_uv_loop_get_active_handles loop)
    if (!= handles 0) {
        (println (+ (int_to_string handles) " handles still active — stopping loop"))
        (nl_uv_stop loop)
        (nl_uv_run loop 0)   # Drain remaining callbacks
    } else {
        (print "")
    }
    (nl_uv_loop_close loop)
}
```

### Pitfall 4: Using nl_uv_sleep in event-driven code

`nl_uv_sleep` blocks the entire thread, preventing the event loop from processing any I/O or timer callbacks. In event-driven code, use a timer handle instead. `nl_uv_sleep` is only appropriate for simple sequential scripts.

### Pitfall 5: Treating nl_uv_now as wall clock time

`nl_uv_now` returns milliseconds since an arbitrary epoch (typically process start), not Unix time. For wall-clock timestamps use the system clock directly. For elapsed time measurement use `nl_uv_hrtime`.

### Pitfall 6: Interpreting nl_uv_loadavg_1min directly

The load average returned by `nl_uv_loadavg_1min` is platform-specific and may be scaled differently on different operating systems. On some platforms it returns `0` or is not meaningful. Do not rely on it for portable load checking.

---

## Best Practices

- Use `nl_uv_hrtime` for precise elapsed-time measurement and benchmarking — it is monotonic and nanosecond-resolution.
- Use `nl_uv_get_free_memory` / `nl_uv_get_total_memory` in diagnostic endpoints to expose runtime memory statistics.
- Expose `nl_uv_os_getpid` and `nl_uv_os_gethostname` in server startup logs to make distributed deployments easier to trace.
- Reserve `nl_uv_sleep` for simple scripts and tests. Event-driven servers should use timer callbacks instead.
- When creating custom loops, always check the return value of `nl_uv_loop_new` before using the handle.
- Drain the loop with `nl_uv_run(loop, 0)` before calling `nl_uv_loop_close` to ensure all pending callbacks fire and handles are released cleanly.

---

**Previous:** [15.2 http_server - Building Web Services](http_server.html)
**Next:** [Chapter 16: Graphics Fundamentals](../16_graphics_fundamentals/index.html)
