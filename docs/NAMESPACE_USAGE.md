# Namespace System Usage Guide

## Overview

NanoLang now supports a hierarchical namespace system for organizing code into modules with proper visibility control. This allows you to create large-scale applications with clear module boundaries and prevents naming conflicts.

## Module Declarations

Every module should declare its namespace at the top of the file:

```nano
module my_app_core

/* Module code here */
```

### Naming Conventions

- Use snake_case for module names: `std_io`, `my_app_utils`
- Hierarchical modules use underscores: `std_math_vector2d`
- Module names must be valid C identifiers (alphanumeric + underscore)

## Import Syntax

### Basic Import

Import entire module:

```nano
import "modules/sdl/sdl.nano"
```

### Selective Import

Import specific symbols from a module:

```nano
from "modules/std/io/stdio.nano" import fopen, fclose, file_exists
```

### Import with Alias

Give an imported module a shorter name:

```nano
import "modules/std/math/vector2d.nano" as vec
```

### Wildcard Import (Not Recommended)

Import all symbols from a module:

```nano
from "modules/std/math/extended.nano" import *
```

**Warning:** Wildcard imports can lead to naming conflicts and make code harder to understand. Use selective imports instead.

## Visibility Control

### Public Functions

Use `pub fn` to make a function accessible from other modules:

```nano
pub fn add(a: int, b: int) -> int {
    return (+ a b)
}
```

### Private Functions

Functions without `pub` are module-private:

```nano
fn helper(x: int) -> int {
    return (* x 2)
}
```

### Public Types

Structs, enums, and unions can be made public:

```nano
pub struct Vector2D {
    x: float,
    y: float
}

pub enum Status {
    Ok,
    Error
}

pub union Result {
    success{int},
    failure{string}
}
```

## Standard Library Organization

The standard library is organized hierarchically:

```
modules/std/
├── lib.nano              # Re-export facade
├── collections/
│   └── stringbuilder.nano
├── io/
│   └── stdio.nano
└── math/
    ├── extended.nano
    └── vector2d.nano
```

### Importing from Standard Library

**I/O Operations:**

```nano
from "modules/std/io/stdio.nano" import fopen, fclose, file_size, FILE_MODE_READ
```

**Extended Math:**

```nano
from "modules/std/math/extended.nano" import atan2, log, exp, PI, E
```

**Vector Math:**

```nano
from "modules/std/math/vector2d.nano" import Vector2D, vec_new, vec_add, vec_length
```

**String Building:**

```nano
from "modules/std/collections/stringbuilder.nano" import StringBuilder, new, append, to_string
```

## Complete Example

```nano
/* My game module */
module my_game

/* Import standard library modules */
from "modules/std/math/vector2d.nano" import Vector2D, vec_new, vec_add, vec_length
from "modules/std/io/stdio.nano" import println

/* Private helper function */
fn calculate_distance(p1: Vector2D, p2: Vector2D) -> float {
    let diff: Vector2D = (vec_sub p2 p1)
    return (vec_length diff)
}

/* Public game function */
pub fn update_player(pos: Vector2D, velocity: Vector2D) -> Vector2D {
    return (vec_add pos velocity)
}

/* Main entry point */
pub fn main() -> int {
    let pos: Vector2D = (vec_new 100.0 200.0)
    let vel: Vector2D = (vec_new 5.0 0.0)
    let new_pos: Vector2D = (update_player pos vel)
    (println "Player updated!")
    return 0
}
```

## Module Search Paths

NanoLang searches for modules in these locations (in order):

1. Relative to the current file
2. `NANO_MODULE_PATH` environment variable (colon-separated)
3. `./modules` directory
4. `/usr/local/lib/nano/modules` (if it exists)

### Setting Module Path

```bash
export NANO_MODULE_PATH="/opt/nano/modules:$HOME/nano/modules:./modules"
```

## Re-Exports and Facades

You can create facade modules that re-export symbols from other modules:

```nano
module std

/* Re-export from collections */
pub use std_collections_stringbuilder::StringBuilder
pub use std_collections_stringbuilder::new as sb_new

/* Re-export from io */
pub use std_io::fopen
pub use std_io::fclose
```

This allows users to write:

```nano
from "modules/std/lib.nano" import StringBuilder, fopen
```

Instead of:

```nano
from "modules/std/collections/stringbuilder.nano" import StringBuilder
from "modules/std/io/stdio.nano" import fopen
```

## Best Practices

### 1. One Module Per File

Each `.nano` file should contain exactly one module declaration.

### 2. Use Selective Imports

Be explicit about what you're importing:

```nano
/* Good */
from "modules/math.nano" import sin, cos, atan2

/* Avoid */
from "modules/math.nano" import *
```

### 3. Minimize Public API Surface

Only export what's necessary:

```nano
/* Public API */
pub fn parse(input: string) -> Result {
    return (internal_parse input)
}

/* Private implementation detail */
fn internal_parse(input: string) -> Result {
    /* ... */
}
```

### 4. Group Related Functionality

Organize modules by functionality, not by type:

```
modules/game/
├── physics.nano      # Physics simulation
├── rendering.nano    # Graphics rendering
└── input.nano        # Input handling
```

Not:

```
modules/game/
├── structs.nano      # All structs
├── functions.nano    # All functions
└── constants.nano    # All constants
```

### 5. Use Hierarchical Naming

For large projects, use hierarchical module names:

```
modules/my_app/
├── core/
│   ├── config.nano
│   └── state.nano
├── ui/
│   ├── widgets.nano
│   └── layout.nano
└── network/
    ├── client.nano
    └── server.nano
```

## Migration Guide

### From Flat Imports to Namespaces

**Before:**

```nano
import "stdio.nano"
let file: int = (fopen "data.txt" "r")
```

**After:**

```nano
from "modules/std/io/stdio.nano" import fopen
let file: int = (fopen "data.txt" "r")
```

### Making Existing Modules Namespace-Aware

1. Add module declaration at the top:

```nano
module my_module
```

2. Mark public functions with `pub`:

```nano
pub fn public_function() -> int { /* ... */ }
fn private_function() -> int { /* ... */ }
```

3. Update imports to use selective imports:

```nano
from "modules/other.nano" import needed_function
```

## Known Limitations

- **No nested modules:** Module names are flat (use underscores for hierarchy)
- **No generic re-exports:** Cannot re-export with type parameters
- **Module per file:** Each file can only declare one module

## Troubleshooting

### "Module file not found"

- Check that the module path is correct
- Verify the file exists in one of the search paths
- Use project-relative paths: `"modules/mymodule.nano"`

### "Function is private to module"

- Add `pub` keyword to the function definition
- Or import the module that contains the public wrapper

### "Undefined function"

- Verify you've imported the function: `from "..." import function_name`
- Check that the function is marked `pub` in the source module
- Ensure module paths are correct

## See Also

- [Module System Overview](MODULE_SYSTEM.md)
- [Standard Library Reference](STDLIB.md)
- [Quick Reference](QUICK_REFERENCE.md)

