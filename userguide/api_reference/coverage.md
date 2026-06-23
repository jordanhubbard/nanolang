# coverage API Reference

The `coverage` module is an importable NanoLang standard library module that provides runtime instrumentation for collecting execution traces, coverage data, and performance timing. It is designed for use during testing and development — particularly to validate code paths, identify untested branches, and measure performance.

Import the module with:

```nano
import "stdlib/coverage"
```

The module exposes three independent subsystems:

- **Coverage tracking** — records which source locations were executed and how many times
- **Performance timing** — measures elapsed time for labeled sections of code
- **Execution tracing** — records a chronological log of function calls, returns, and line events

---

## Types

### CoveragePoint

```nano
struct CoveragePoint {
    file: string,
    line: int,
    column: int,
    hit_count: int
}
```

Represents a single instrumented location in source code and the number of times it was executed.

**Fields:**
- `file` — the source file name or identifier
- `line` — the line number
- `column` — the column number
- `hit_count` — the number of times this location was executed

---

### TimingPoint

```nano
struct TimingPoint {
    label: string,
    start_time_ms: int,
    total_time_ms: int,
    call_count: int
}
```

Accumulates timing data for a named section of code across multiple invocations.

**Fields:**
- `label` — the name given to this timing section
- `start_time_ms` — the timestamp of the most recent start (in milliseconds)
- `total_time_ms` — the cumulative elapsed time across all calls (in milliseconds)
- `call_count` — the total number of times this section was timed

---

### TraceEvent

```nano
struct TraceEvent {
    timestamp_ms: int,
    event_type: string,
    location: string,
    details: string
}
```

Represents a single entry in the execution trace log.

**Fields:**
- `timestamp_ms` — timestamp in milliseconds at the time the event was recorded
- `event_type` — one of `"CALL"`, `"RETURN"`, or `"LINE"`
- `location` — the function name or source location where the event occurred
- `details` — additional context (e.g., argument values or return values as strings)

---

## Coverage Functions

### coverage_init

```nano
fn coverage_init() -> void
```

Initializes coverage tracking. Enables recording and clears any previously collected data. Call this before the code you want to instrument.

**Example:**
```nano
(coverage_init)
```

---

### coverage_record

```nano
fn coverage_record(file: string, line: int, column: int) -> void
```

Records a hit at the specified source location. If the location has been recorded before, its `hit_count` is incremented. Has no effect when coverage tracking is disabled.

**Parameters:**
- `file` — the source file name or identifier
- `line` — the line number of the instrumented location
- `column` — the column number of the instrumented location

**Example:**
```nano
fn my_function(x: int) -> int {
    (coverage_record "my_file.nano" 10 5)
    return (* x 2)
}
```

---

### coverage_get_hit_count

```nano
fn coverage_get_hit_count() -> int
```

Returns the total number of distinct coverage points that have been hit at least once since the last `coverage_init` or `coverage_reset`.

**Returns:** The count of distinct recorded locations.

**Example:**
```nano
(coverage_init)
(coverage_record "file.nano" 1 1)
(coverage_record "file.nano" 2 1)
let n: int = (coverage_get_hit_count)  # 2
```

---

### coverage_get_line_hits

```nano
fn coverage_get_line_hits(file: string, line: int) -> int
```

Returns the number of times a specific source line was executed.

**Parameters:**
- `file` — the source file name or identifier
- `line` — the line number to query

**Returns:** The hit count for that line, or `0` if it was never recorded.

**Example:**
```nano
(coverage_init)
(coverage_record "app.nano" 42 1)
(coverage_record "app.nano" 42 1)
let hits: int = (coverage_get_line_hits "app.nano" 42)  # 2
```

---

### coverage_report

```nano
fn coverage_report() -> void
```

Prints a formatted coverage report to standard output listing all recorded locations and their hit counts.

**Example output:**
```
========================================
Coverage Report
========================================
Total coverage points: 2

File:Line:Column - Hit Count
----------------------------------------
app.nano:10:5 - 3 hits
app.nano:15:1 - 1 hits
========================================
```

**Example:**
```nano
(coverage_init)
# ... run instrumented code ...
(coverage_report)
```

---

### coverage_reset

```nano
fn coverage_reset() -> void
```

Clears all recorded coverage data without disabling coverage tracking. Use this to reset between test runs while keeping coverage active.

**Example:**
```nano
(coverage_reset)
```

---

### coverage_disable

```nano
fn coverage_disable() -> void
```

Disables coverage recording. Subsequent calls to `coverage_record` have no effect until `coverage_init` is called again.

**Example:**
```nano
(coverage_disable)
```

---

## Timing Functions

### timing_start

```nano
fn timing_start(label: string) -> int
```

Records the current time in milliseconds. Pass the returned value to `timing_end` to measure the elapsed time for a section of code.

**Parameters:**
- `label` — a name for this timing section (informational; used in `timing_end` for matching)

**Returns:** The current time in milliseconds.

**Example:**
```nano
let t: int = (timing_start "parse_phase")
# ... do work ...
(timing_end "parse_phase" t)
```

---

### timing_end

```nano
fn timing_end(label: string, start_time: int) -> void
```

Records the elapsed time since `start_time` under the given label. If a timing entry for `label` already exists, the elapsed time and call count are accumulated.

**Parameters:**
- `label` — the name of the timing section (should match the label used in `timing_start`)
- `start_time` — the value returned by the corresponding `timing_start` call

**Example:**
```nano
let t: int = (timing_start "render")
# ... render frame ...
(timing_end "render" t)
```

---

### timing_report

```nano
fn timing_report() -> void
```

Prints a formatted performance timing report to standard output showing each labeled section's total call count, total elapsed time, and average time per call.

**Example output:**
```
========================================
Performance Timing Report
========================================
Label - Calls - Total Time (ms) - Avg Time (ms)
----------------------------------------
parse_phase - 10 calls - 320 ms - 32 ms avg
render - 60 calls - 1200 ms - 20 ms avg
========================================
```

**Example:**
```nano
(timing_report)
```

---

### timing_reset

```nano
fn timing_reset() -> void
```

Clears all accumulated timing data.

**Example:**
```nano
(timing_reset)
```

---

## Trace Functions

### trace_init

```nano
fn trace_init() -> void
```

Initializes the execution tracer. Enables trace recording and clears any previously collected events. Uses the default event limit of 10,000 events.

**Example:**
```nano
(trace_init)
```

---

### trace_init_with_limit

```nano
fn trace_init_with_limit(max_events: int) -> void
```

Initializes the execution tracer with a custom maximum event count. Once the limit is reached, additional events are silently dropped. A warning is shown in `trace_report` if the limit was hit.

**Parameters:**
- `max_events` — the maximum number of trace events to store

**Example:**
```nano
(trace_init_with_limit 500)
```

---

### trace_record

```nano
fn trace_record(event_type: string, location: string, details: string) -> void
```

Records a single trace event. Has no effect when tracing is disabled or when the event limit has been reached.

**Parameters:**
- `event_type` — the kind of event: `"CALL"`, `"RETURN"`, or `"LINE"`
- `location` — the function name or source location
- `details` — additional context, such as argument or return values serialized as strings

**Example:**
```nano
fn my_function(x: int) -> int {
    (trace_record "CALL" "my_function" (int_to_string x))
    let result: int = (* x 2)
    (trace_record "RETURN" "my_function" (int_to_string result))
    return result
}
```

---

### trace_report

```nano
fn trace_report() -> void
```

Prints the full execution trace to standard output in chronological order. If the trace limit was reached, a warning is printed.

**Example output:**
```
========================================
Execution Trace
========================================
Total events: 3

Timestamp - Type - Location - Details
----------------------------------------
0 - CALL - my_function - 5
0 - RETURN - my_function - 10
0 - CALL - my_function - 7
========================================
```

**Example:**
```nano
(trace_report)
```

---

### trace_disable

```nano
fn trace_disable() -> void
```

Disables trace recording. Subsequent calls to `trace_record` have no effect until `trace_init` or `trace_init_with_limit` is called again.

**Example:**
```nano
(trace_disable)
```

---

### trace_reset

```nano
fn trace_reset() -> void
```

Clears all recorded trace events without disabling the tracer.

**Example:**
```nano
(trace_reset)
```

---

## Global State

The module uses module-level mutable globals to hold state across function calls:

| Variable | Type | Description |
|----------|------|-------------|
| `g_coverage_enabled` | `bool` | Whether coverage recording is active |
| `g_coverage_points` | `array<CoveragePoint>` | All recorded coverage points |
| `g_timing_points` | `array<TimingPoint>` | All accumulated timing entries |
| `g_trace_enabled` | `bool` | Whether trace recording is active |
| `g_trace_events` | `array<TraceEvent>` | All recorded trace events |
| `g_trace_max_events` | `int` | Maximum trace events before dropping (default: 10000) |

---

## Usage Pattern

A typical instrumented function looks like:

```nano
fn my_function(x: int) -> int {
    (trace_record "CALL" "my_function" (int_to_string x))
    (coverage_record "my_file.nano" 10 5)

    let result: int = (* x 2)

    (trace_record "RETURN" "my_function" (int_to_string result))
    return result
}

fn main() -> int {
    (coverage_init)
    (trace_init)
    let t: int = (timing_start "my_function")

    let r: int = (my_function 21)

    (timing_end "my_function" t)
    (coverage_report)
    (timing_report)
    (trace_report)
    return 0
}
```
