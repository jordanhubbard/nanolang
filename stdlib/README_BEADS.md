# Beads Module Documentation

The `stdlib/beads.nano` module provides programmatic access to the Beads issue tracker from NanoLang code. It enables automatic issue creation, querying, and management directly from your programs.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Future Features](#future-features)

## Overview

The beads module wraps the `bd` command-line tool, providing a type-safe NanoLang API for:

- **Querying** issues by status, priority, and other criteria
- **Creating** issues programmatically with full metadata
- **Closing** issues with reasons
- **Getting** project statistics
- **Automatic issue creation** from failing assertions (killer feature!)

## Core Concepts

### Bead Structure

A `Bead` represents a single issue:

```nano
struct Bead {
    id: string,              # e.g., "nanolang-abc123"
    title: string,           # Issue title
    description: string,     # Full description
    status: string,          # open, in_progress, blocked, closed
    priority: int,           # 0=P0, 1=P1, 2=P2, 3=P3, 4=P4
    issue_type: string,      # bug, feature, task, chore, epic
    created_at: string,      # ISO timestamp
    updated_at: string,      # ISO timestamp
    labels: array<string>,   # Tags/categories
    close_reason: string,    # Why it was closed
    dependency_count: int,   # How many dependencies
    dependent_count: int     # How many depend on this
}
```

### Priority Levels

- **P0**: Critical - Drop everything and fix
- **P1**: High - Fix this sprint
- **P2**: Medium - Fix soon
- **P3**: Low - Nice to have
- **P4**: Backlog - Eventually

### Issue Types

- **bug**: Something is broken
- **feature**: New functionality
- **task**: Work item
- **chore**: Maintenance
- **epic**: Large multi-task project

### Status Values

- **open**: Not started
- **in_progress**: Currently being worked on
- **blocked**: Waiting on dependencies
- **closed**: Completed

## API Reference

### Querying Functions

#### `bd_list(status: string) -> array<Bead>`
List beads filtered by status.

```nano
let open_beads: array<Bead> = (bd_list "open")
let all_beads: array<Bead> = (bd_list "")  # Empty string = all
```

#### `bd_open() -> array<Bead>`
Get all open beads.

```nano
let open: array<Bead> = (bd_open)
```

#### `bd_ready() -> array<Bead>`
Get beads ready to work (no blockers).

```nano
let ready: array<Bead> = (bd_ready)
```

#### `bd_by_priority(priority: int) -> array<Bead>`
Get beads of a specific priority.

```nano
let p0_beads: array<Bead> = (bd_by_priority 0)
let p1_beads: array<Bead> = (bd_by_priority 1)
```

#### `bd_show(id: string) -> Bead`
Get details of a specific bead.

```nano
let bead: Bead = (bd_show "nanolang-abc123")
```

#### `bd_stats() -> BeadStats`
Get project statistics.

```nano
let stats: BeadStats = (bd_stats)
(println (+ "Open: " (int_to_string stats.open)))
(println (+ "Closed: " (int_to_string stats.closed)))
```

### Creation Functions

#### `bd_create(title: string, description: string, priority: int, issue_type: string) -> string`
Create a new bead (returns bead ID).

```nano
let bead_id: string = (bd_create
    "Fix memory leak in parser"
    "Parser allocates but never frees tokens"
    0
    "bug"
)
```

#### `bd_create_with_options(opts: BeadCreateOptions) -> string`
Create with full options including labels.

```nano
let labels: array<string> = (array_new 2 "")
(array_set labels 0 "compiler")
(array_set labels 1 "performance")

let opts: BeadCreateOptions = BeadCreateOptions {
    title: "Optimize typechecker",
    description: "Reduce O(n²) to O(n log n)",
    priority: 2,
    issue_type: "feature",
    labels: labels
}

let id: string = (bd_create_with_options opts)
```

### Management Functions

#### `bd_close(id: string, reason: string) -> bool`
Close a bead with a reason.

```nano
let success: bool = (bd_close
    "nanolang-abc123"
    "Fixed by commit 1234567"
)
```

### Assertion Functions (Killer Feature!)

#### `assert_with_bead(condition: bool, title: string, priority: int, description: string) -> bool`
Assert that creates a bead if condition fails.

```nano
fn divide(a: int, b: int) -> int {
    # If b is zero, creates a P0 bug automatically!
    (assert_with_bead
        (!= b 0)
        "Division by zero"
        0
        "Attempted to divide by zero"
    )
    return (/ a b)
}
```

#### `assert_with_bead_context(condition: bool, title: string, priority: int, file: string, line: int, context: string) -> bool`
Enhanced version with file/line context.

```nano
(assert_with_bead_context
    (> value 0)
    "Negative value detected"
    1
    "validator.nano"
    42
    "Value must be positive for calculations"
)
```

## Examples

### Example 1: Basic Usage

```nano
from "stdlib/beads.nano" import bd_stats, bd_open, bd_create

fn main() -> int {
    # Get statistics
    let stats: BeadStats = (bd_stats)
    (println (+ "Total issues: " (int_to_string stats.total)))
    
    # List open issues
    let open: array<Bead> = (bd_open)
    (println (+ "Open issues: " (int_to_string (array_length open))))
    
    # Create a new issue
    let id: string = (bd_create
        "Add new feature"
        "Detailed description here"
        2
        "feature"
    )
    (println (+ "Created: " id))
    
    return 0
}
```

### Example 2: Automatic Issue Creation

```nano
from "stdlib/beads.nano" import assert_with_bead

fn validate_config(timeout: int) -> bool {
    # Creates P0 bug if timeout is invalid
    return (assert_with_bead
        (and (> timeout 0) (< timeout 60000))
        "Invalid timeout configuration"
        0
        (+ "Timeout must be 1-60000ms, got: " (int_to_string timeout))
    )
}
```

### Example 3: Workflow Automation

```nano
from "stdlib/beads.nano" import bd_by_priority, bd_ready

fn check_urgent_work() -> bool {
    # Check for P0 issues
    let p0_beads: array<Bead> = (bd_by_priority 0)
    
    if (> (array_length p0_beads) 0) {
        (println "⚠️  URGENT: P0 issues need attention!")
        return true
    }
    
    # Check if work is available
    let ready: array<Bead> = (bd_ready)
    (println (+ (int_to_string (array_length ready)) " issues ready to work"))
    
    return false
}
```

### Example 4: Test Suite Integration

```nano
from "stdlib/beads.nano" import assert_with_bead

fn test_sorting() -> bool {
    let arr: array<int> = (sort (array_new 5 0))
    
    # If sorting fails, creates a bug automatically
    return (and
        (assert_with_bead
            (is_sorted arr)
            "Sorting algorithm broken"
            0
            "Array not sorted after sort() call"
        )
        (assert_with_bead
            (== (array_length arr) 5)
            "Array length changed during sort"
            1
            "Sort should preserve array length"
        )
    )
}
```

## Future Features

### Planned Enhancements

1. **Auto-capture stack traces**: Include call stack in bug descriptions
2. **Smart deduplication**: Detect duplicate issues before creating
3. **Metric tracking**: Count assertion failures over time
4. **CI/CD integration**: Automatically close beads on successful builds
5. **Watch mode**: Monitor for new beads in real-time
6. **Bulk operations**: Update multiple beads at once
7. **Templates**: Pre-defined issue templates
8. **Dependencies**: Link related beads programmatically

### Future API Extensions

```nano
# Planned future functions:
fn bd_add_dependency(id: string, depends_on: string) -> bool
fn bd_add_label(id: string, label: string) -> bool
fn bd_update_priority(id: string, new_priority: int) -> bool
fn bd_reopen(id: string, reason: string) -> bool
fn bd_search(query: string) -> array<Bead>
fn bd_watch(callback: fn(Bead) -> void) -> void
```

## Dependencies

- **stdlib/process.nano**: Execute shell commands
- **stdlib/json.nano**: Parse JSON output from bd
- **bd command**: Must be installed and in PATH

## Implementation Status

### ✅ Complete
- Core data structures (Bead, BeadStats, BeadCreateOptions)
- Function signatures and API design
- Type-safe wrappers
- Shadow tests
- Documentation

### 🚧 In Progress
- C FFI for command execution (`stdlib/process.nano`)
- JSON parsing (`stdlib/json.nano`)
- String parsing helpers

### 📋 TODO
- Integration tests with real bd command
- Performance optimization
- Error handling improvements
- Command timeout handling

## Testing

Run the test suite:

```bash
./bin/nanoc tests/test_beads_module.nano -o bin/test_beads
./bin/test_beads
```

Run examples:

```bash
./bin/nanoc examples/advanced/beads_basic_usage.nano -o bin/beads_basic
./bin/beads_basic
```

## Contributing

The beads module is designed to grow with community needs. Contributions welcome for:

- Additional query filters
- Better error messages
- Performance improvements
- New automation workflows
- Integration examples

## License

Part of the NanoLang standard library. Same license as NanoLang itself.
