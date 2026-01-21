# Chapter 12: System & Runtime

**Execute processes, track coverage, and interact with the runtime.**

This chapter covers system-level operations: process execution, coverage tracking, and runtime utilities.

## 12.1 Process Execution

Execute external commands and capture their output.

### Running Commands (Simple)

```nano
from "stdlib/process.nano" import exec_command, CommandResult

fn run_simple_command() -> int {
    let result: CommandResult = (exec_command "echo hello")
    return result.exit_code
}

shadow run_simple_command {
    assert (== (run_simple_command) 0)
}
```

**Returns:** `CommandResult` struct with:
- `exit_code: int` - Process exit code (0 = success)
- `stdout: string` - Standard output
- `stderr: string` - Standard error

### Capturing Output

```nano
from "stdlib/process.nano" import exec_command, CommandResult

fn get_command_output(command: string) -> string {
    let result: CommandResult = (exec_command command)
    if (== result.exit_code 0) {
        return result.stdout
    }
    return result.stderr
}

shadow get_command_output {
    let output: string = (get_command_output "echo test")
    assert (> (str_length output) 0)
}
```

### Checking Exit Codes

```nano
from "stdlib/process.nano" import exec_command, CommandResult

fn command_succeeded(command: string) -> bool {
    let result: CommandResult = (exec_command command)
    return (== result.exit_code 0)
}

shadow command_succeeded {
    assert (command_succeeded "true")
    assert (not (command_succeeded "false"))
}
```

## 12.2 Advanced Process Management

For more control, use the `modules/std/process.nano` module.

### Running with Output

```nano
from "modules/std/process.nano" import run, Output

fn run_with_details() -> Output {
    let result: Output = (run "ls -la")
    return result
}

shadow run_with_details {
    let output: Output = (run "echo test")
    assert (== output.code 0)
}
```

### Executing Without Capture

```nano
from "modules/std/process.nano" import exec

fn execute_command(command: string) -> bool {
    let code: int = (exec command)
    return (== code 0)
}

shadow execute_command {
    assert (execute_command "true")
}
```

**Use `exec` when:**
- Don't need output capture
- Want process to write directly to terminal
- Faster execution (no buffering)

### Spawning Background Processes

```nano
from "modules/std/process.nano" import spawn, wait, is_running

fn run_background_task() -> int {
    let pid: int = (spawn "sleep 0.1")
    
    if (< pid 0) {
        return -1  # Spawn failed
    }
    
    # Check if running
    let status: int = (is_running pid)
    
    # Wait for completion
    let exit_code: int = (wait pid)
    return exit_code
}

shadow run_background_task {
    assert (== (run_background_task) 0)
}
```

**Functions:**
- `spawn(command) -> int` - Returns PID or -1 on error
- `is_running(pid) -> int` - Returns 1 if running, 0 if exited, -1 on error
- `wait(pid) -> int` - Blocks until process exits, returns exit code

### Example: Long-Running Process

```nano
from "modules/std/process.nano" import spawn, is_running, wait

fn monitor_process(command: string, max_seconds: int) -> bool {
    let pid: int = (spawn command)
    if (< pid 0) {
        return false
    }
    
    let mut elapsed: int = 0
    while (and (< elapsed max_seconds) (== (is_running pid) 1)) {
        # Sleep simulation (simplified)
        set elapsed (+ elapsed 1)
    }
    
    if (== (is_running pid) 1) {
        # Still running after timeout
        return false
    }
    
    let exit_code: int = (wait pid)
    return (== exit_code 0)
}

shadow monitor_process {
    assert (monitor_process "true" 5)
}
```

## 12.3 Coverage Tracking

Track code execution for testing and validation.

### Initializing Coverage

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_report

fn start_coverage() -> void {
    (coverage_init)
}

shadow start_coverage {
    (start_coverage)
}
```

### Recording Coverage Points

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_get_hit_count

fn track_execution() -> int {
    (coverage_init)
    (coverage_record "test.nano" 10 5)
    (coverage_record "test.nano" 15 10)
    return (coverage_get_hit_count)
}

shadow track_execution {
    assert (== (track_execution) 2)
}
```

**Signature:** `coverage_record(file: string, line: int, column: int)`
- Records a single execution point
- Increments hit count on repeated calls
- Tracks per-file, per-line, per-column

### Querying Coverage

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_get_line_hits

fn check_line_hits() -> int {
    (coverage_init)
    (coverage_record "main.nano" 42 0)
    (coverage_record "main.nano" 42 0)
    (coverage_record "main.nano" 42 0)
    
    return (coverage_get_line_hits "main.nano" 42)
}

shadow check_line_hits {
    assert (== (check_line_hits) 3)
}
```

### Generating Reports

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_report

fn generate_coverage_report() -> void {
    (coverage_init)
    (coverage_record "src/main.nano" 10 5)
    (coverage_record "src/main.nano" 15 10)
    (coverage_record "src/util.nano" 20 8)
    
    (coverage_report)
}

shadow generate_coverage_report {
    (generate_coverage_report)
}
```

**Output format:**
```
Coverage Report:
  src/main.nano:10:5 - hit 1 times
  src/main.nano:15:10 - hit 1 times
  src/util.nano:20:8 - hit 1 times
```

## 12.4 Range Iteration

The `range` function creates iterators for `for` loops.

### Basic Range

```nano
fn count_to_ten() -> int {
    let mut sum: int = 0
    for i in (range 0 10) {
        set sum (+ sum i)
    }
    return sum
}

shadow count_to_ten {
    assert (== (count_to_ten) 45)  # 0+1+2+...+9
}
```

**Syntax:** `range(start, end)`
- `start` - Inclusive beginning
- `end` - Exclusive ending
- Returns iterator (only usable in `for` loops)

### Range Examples

```nano
fn range_examples() -> bool {
    # Range 0..5 (0, 1, 2, 3, 4)
    let mut count1: int = 0
    for i in (range 0 5) {
        set count1 (+ count1 1)
    }
    
    # Range 5..10 (5, 6, 7, 8, 9)
    let mut count2: int = 0
    for i in (range 5 10) {
        set count2 (+ count2 1)
    }
    
    # Empty range
    let mut count3: int = 0
    for i in (range 10 10) {
        set count3 (+ count3 1)
    }
    
    return (and (== count1 5) (and (== count2 5) (== count3 0)))
}

shadow range_examples {
    assert (range_examples)
}
```

## 12.5 Practical Examples

### Example 1: Build Script

```nano
from "stdlib/process.nano" import exec_command, CommandResult

fn build_project() -> bool {
    # Compile source files
    let compile: CommandResult = (exec_command "make compile")
    if (!= compile.exit_code 0) {
        (println "Compile failed:")
        (println compile.stderr)
        return false
    }
    
    # Run tests
    let test: CommandResult = (exec_command "make test")
    if (!= test.exit_code 0) {
        (println "Tests failed:")
        (println test.stderr)
        return false
    }
    
    return true
}

shadow build_project {
    # Would test with actual build system
    assert true
}
```

### Example 2: Code Coverage

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_get_line_hits

fn factorial_with_coverage(n: int) -> int {
    (coverage_record "factorial.nano" 1 0)
    if (<= n 1) {
        (coverage_record "factorial.nano" 2 0)
        return 1
    }
    (coverage_record "factorial.nano" 4 0)
    return (* n (factorial_with_coverage (- n 1)))
}

fn test_factorial_coverage() -> bool {
    (coverage_init)
    let result: int = (factorial_with_coverage 3)
    
    # Check all lines were hit
    let hits1: int = (coverage_get_line_hits "factorial.nano" 1)
    let hits2: int = (coverage_get_line_hits "factorial.nano" 2)
    let hits4: int = (coverage_get_line_hits "factorial.nano" 4)
    
    return (and (> hits1 0) (and (> hits2 0) (> hits4 0)))
}

shadow test_factorial_coverage {
    assert (test_factorial_coverage)
}
```

### Example 3: Command Pipeline

```nano
from "stdlib/process.nano" import exec_command, CommandResult

fn count_lines(file: string) -> int {
    let command: string = (+ "wc -l " file)
    let result: CommandResult = (exec_command command)
    
    if (!= result.exit_code 0) {
        return 0
    }
    
    # Parse output (format: "   123 filename")
    let output: string = result.stdout
    # Simplified parsing
    return (string_to_int output)
}

shadow count_lines {
    # Would test with actual file
    assert true
}
```

### Example 4: Process Monitor

```nano
from "modules/std/process.nano" import spawn, is_running, wait

fn run_with_timeout(command: string, timeout: int) -> int {
    let pid: int = (spawn command)
    if (< pid 0) {
        return -1
    }
    
    let mut checks: int = 0
    let max_checks: int = (* timeout 10)  # Check 10 times per second
    
    while (< checks max_checks) {
        if (== (is_running pid) 0) {
            # Process finished
            return (wait pid)
        }
        set checks (+ checks 1)
    }
    
    # Timeout - process still running
    return -2
}

shadow run_with_timeout {
    let code: int = (run_with_timeout "true" 1)
    assert (== code 0)
}
```

### Example 5: Test Runner

```nano
from "stdlib/process.nano" import exec_command, CommandResult
from "stdlib/coverage.nano" import coverage_init, coverage_report

fn run_tests(test_files: array<string>) -> bool {
    (coverage_init)
    
    let mut passed: int = 0
    let mut failed: int = 0
    let len: int = (array_length test_files)
    
    for i in (range 0 len) {
        let test: string = (at test_files i)
        let command: string = (+ "./test_runner " test)
        let result: CommandResult = (exec_command command)
        
        if (== result.exit_code 0) {
            set passed (+ passed 1)
        } else {
            set failed (+ failed 1)
            (println (+ "FAILED: " test))
        }
    }
    
    (println (+ "Passed: " (int_to_string passed)))
    (println (+ "Failed: " (int_to_string failed)))
    (coverage_report)
    
    return (== failed 0)
}

shadow run_tests {
    # Would test with actual test files
    assert true
}
```

## 12.6 Best Practices

### ✅ DO

**1. Check exit codes:**

```nano
from "stdlib/process.nano" import exec_command, CommandResult

fn safe_exec(command: string) -> bool {
    let result: CommandResult = (exec_command command)
    if (!= result.exit_code 0) {
        (println "Command failed:")
        (println result.stderr)
        return false
    }
    return true
}

shadow safe_exec {
    assert (safe_exec "true")
}
```

**2. Initialize coverage before tracking:**

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record

fn proper_coverage_use() -> void {
    (coverage_init)  # Always init first
    (coverage_record "file.nano" 10 5)
}

shadow proper_coverage_use {
    (proper_coverage_use)
}
```

**3. Wait for spawned processes:**

```nano
from "modules/std/process.nano" import spawn, wait

fn spawn_and_wait(command: string) -> int {
    let pid: int = (spawn command)
    if (< pid 0) {
        return -1
    }
    return (wait pid)
}

shadow spawn_and_wait {
    assert (== (spawn_and_wait "true") 0)
}
```

### ❌ DON'T

**1. Don't ignore process failures:**

```nano
# ❌ Bad: Ignores errors
from "stdlib/process.nano" import exec_command
let result: CommandResult = (exec_command "might_fail")

# ✅ Good: Checks exit code
if (!= result.exit_code 0) {
    (println "Error occurred")
}
```

**2. Don't spawn without cleanup:**

```nano
# ❌ Bad: Zombie process
from "modules/std/process.nano" import spawn
let pid: int = (spawn "long_running")
# Never waits!

# ✅ Good: Always wait
let exit_code: int = (wait pid)
```

**3. Don't hardcode commands:**

```nano
# ❌ Bad: Platform-specific
from "stdlib/process.nano" import exec_command
let result: CommandResult = (exec_command "dir")  # Windows only!

# ✅ Good: Portable
# Check platform or use cross-platform tools
```

## Summary

In this chapter, you learned:
- ✅ Process execution: `exec_command`, `run`, `exec`
- ✅ Background processes: `spawn`, `is_running`, `wait`
- ✅ Coverage tracking: `coverage_init`, `coverage_record`, `coverage_report`
- ✅ Coverage queries: `coverage_get_hit_count`, `coverage_get_line_hits`
- ✅ Range iteration: `range(start, end)` for loops
- ✅ Practical patterns: build scripts, test runners, monitors

### Quick Reference

| Operation | Function | Module |
|-----------|----------|--------|
| **Run command** | `exec_command(cmd)` | `stdlib/process` |
| **Run (advanced)** | `run(cmd)` | `modules/std/process` |
| **Execute** | `exec(cmd)` | `modules/std/process` |
| **Spawn process** | `spawn(cmd)` | `modules/std/process` |
| **Check running** | `is_running(pid)` | `modules/std/process` |
| **Wait for exit** | `wait(pid)` | `modules/std/process` |
| **Init coverage** | `coverage_init()` | `stdlib/coverage` |
| **Record hit** | `coverage_record(f,l,c)` | `stdlib/coverage` |
| **Get hits** | `coverage_get_line_hits(f,l)` | `stdlib/coverage` |
| **Report** | `coverage_report()` | `stdlib/coverage` |
| **Loop range** | `range(start, end)` | Built-in |

### Process Execution Decision Tree

```
Need to capture output?
  ├─ Yes → Use exec_command or run
  └─ No → Use exec

Need background execution?
  ├─ Yes → Use spawn + wait
  └─ No → Use exec_command/run/exec

Need to monitor progress?
  └─ Yes → Use spawn + is_running + wait
```

---

**Previous:** [Chapter 11: I/O & Filesystem](11_io_filesystem.md)  
**Next:** [Chapter 13: Text Processing](../part3_modules/13_text_processing/index.md)
