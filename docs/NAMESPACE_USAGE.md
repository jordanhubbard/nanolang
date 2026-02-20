# My Namespace and Module System

This is my guide to modules, namespaces, imports, and exports. I describe how I organize code and how I manage what is visible between files.

## Table of Contents

1. [Overview](#overview)
2. [Module Declaration](#module-declaration)
3. [Import Syntax](#import-syntax)
4. [Export Control (pub)](#export-control-pub)
5. [Qualified Access](#qualified-access)
6. [Module Structure](#module-structure)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

---

## Overview

My module system provides:
- **Explicit imports** - I require you to import only what you need.
- **Namespace control** - I distinguish between public and private symbols.
- **Path-based modules** - I use a file-based module system.
- **Selective imports** - I allow importing specific symbols or entire modules.
- **Aliasing** - I let you rename imports to avoid conflicts.

### Key Concepts

- **Module**: A `.nano` file that I can import.
- **Namespace**: I declare these via the `module` keyword.
- **Public symbols**: I mark these with `pub`. They are accessible to importers.
- **Private symbols**: I treat symbols without `pub` as private to the module.
- **Import**: I bring external symbols into the current scope.

---

## Module Declaration

I let you declare a module namespace. This is optional, but I recommend it for larger projects.

```nano
module my_app

# Your code here
```

**Benefits:**
- I prevent symbol collisions.
- I make module boundaries explicit.
- I enable better code organization.

**Note:** My `module` declaration is optional. I treat files without it as valid modules.

---

## Import Syntax

### Selective Import (Recommended)

I let you import specific symbols from a module.

```nano
from "modules/readline/readline.nano" import rl_readline, rl_add_history
```

**Syntax:**
```nano
from "path/to/module.nano" import symbol1, symbol2, symbol3
```

**Use when:**
- I only need to see a few functions.
- You want to keep your imports explicit.
- You want to avoid polluting my namespace.

### Import All (Use Sparingly)

I can import all public symbols from a module.

```nano
import "path/to/module.nano"
```

**Caution:** This brings all public symbols into my namespace. It can cause name conflicts.

### Import with Alias

I allow importing and renaming to avoid conflicts.

```nano
from "modules/math_ext.nano" import sqrt as math_sqrt
from "modules/graphics.nano" import sqrt as gfx_sqrt
```

I also allow aliasing the entire module.

```nano
import "modules/std/io/stdio.nano" as io

# Use qualified names:
let file: int = (io.fopen "test.txt" "r")
```

---

## Export Control (pub)

I control what is visible to importers using the `pub` keyword.

### Public Functions

```nano
# Exported - visible to importers
pub fn exported_function(x: int) -> int {
    return (* x 2)
}

# Private - module-only
fn internal_helper(x: int) -> int {
    return (+ x 1)
}
```

### Public Types

```nano
# Exported struct
pub struct Config {
    enabled: bool,
    timeout: int
}

# Private struct
struct InternalState {
    counter: int
}
```

---

## Best Practices

### 1. Be Explicit with Imports

**Good:
```nano
from "utils.nano" import add, multiply, divide
```

**Avoid:
```nano
import "utils.nano"  # Imports everything
```

### 2. Use Public Sparingly

I only expose what consumers need.

```nano
pub fn api_function() -> int { ... }
fn internal_helper() -> int { ... }  # Private
```

### 3. Group Related Imports

```nano
# Standard library
from "modules/std/io/stdio.nano" import fopen, fclose
from "modules/std/collections/list.nano" import List_new

# Third-party modules
from "modules/json/json.nano" import parse, stringify

# Local modules
from "./types.nano" import Config
from "./utils.nano" import helper
```

---

## Related Documentation

- [MODULE_SYSTEM.md](MODULE_SYSTEM.md) - My module build system and module.json.
- [EXTERN_FFI.md](EXTERN_FFI.md) - My C FFI and external functions.
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - My syntax cheat sheet.
- [SPECIFICATION.md](SPECIFICATION.md) - My complete language specification.

---

**Last Updated:** February 20, 2026
**Status:** Complete
**Version:** 0.2.0+

