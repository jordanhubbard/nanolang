# Namespace and Module System Usage Guide

Complete guide to NanoLang's module system, namespaces, imports, and exports.

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

NanoLang's module system provides:
- **Explicit imports** - Import only what you need
- **Namespace control** - Public vs private symbols
- **Path-based modules** - File-based module system
- **Selective imports** - Import specific symbols or entire modules
- **Aliasing** - Rename imports to avoid conflicts

### Key Concepts

- **Module**: A `.nano` file that can be imported
- **Namespace**: Declared via `module` keyword (optional)
- **Public symbols**: Marked with `pub`, accessible to importers
- **Private symbols**: Not marked `pub`, module-only
- **Import**: Bringing external symbols into scope

---

## Module Declaration

Declare a module namespace (optional but recommended for larger projects):

\`\`\`nano
module my_app

# Your code here
\`\`\`

**Benefits:**
- Prevents symbol collisions
- Makes module boundaries explicit
- Enables better code organization

**Note:** `module` declaration is optional. Files without it are still valid modules.

---

## Import Syntax

### Selective Import (Recommended)

Import specific symbols from a module:

\`\`\`nano
from "modules/readline/readline.nano" import rl_readline, rl_add_history
\`\`\`

**Syntax:**
\`\`\`nano
from "path/to/module.nano" import symbol1, symbol2, symbol3
\`\`\`

**Use when:**
- You only need a few functions
- You want to keep imports explicit
- You want to avoid namespace pollution

### Import All (Use Sparingly)

Import all public symbols from a module:

\`\`\`nano
import "path/to/module.nano"
\`\`\`

**Caution:** This brings ALL public symbols into your namespace. Can cause name conflicts.

### Import with Alias

Import and rename to avoid conflicts:

\`\`\`nano
from "modules/math_ext.nano" import sqrt as math_sqrt
from "modules/graphics.nano" import sqrt as gfx_sqrt
\`\`\`

Or alias the entire module:

\`\`\`nano
import "modules/std/io/stdio.nano" as io

# Use qualified names:
let file: int = (io.fopen "test.txt" "r")
\`\`\`

---

## Export Control (pub)

Control what's visible to importers using the `pub` keyword.

### Public Functions

\`\`\`nano
# Exported - visible to importers
pub fn exported_function(x: int) -> int {
    return (* x 2)
}

# Private - module-only
fn internal_helper(x: int) -> int {
    return (+ x 1)
}
\`\`\`

### Public Types

\`\`\`nano
# Exported struct
pub struct Config {
    enabled: bool,
    timeout: int
}

# Private struct
struct InternalState {
    counter: int
}
\`\`\`

---

## Best Practices

### 1. Be Explicit with Imports

✅ **Good:**
\`\`\`nano
from "utils.nano" import add, multiply, divide
\`\`\`

❌ **Avoid:**
\`\`\`nano
import "utils.nano"  # Imports everything
\`\`\`

### 2. Use Public Sparingly

Only expose what consumers need:

\`\`\`nano
pub fn api_function() -> int { ... }
fn internal_helper() -> int { ... }  # Private
\`\`\`

### 3. Group Related Imports

\`\`\`nano
# Standard library
from "modules/std/io/stdio.nano" import fopen, fclose
from "modules/std/collections/list.nano" import List_new

# Third-party modules
from "modules/json/json.nano" import parse, stringify

# Local modules
from "./types.nano" import Config
from "./utils.nano" import helper
\`\`\`

---

## Related Documentation

- [MODULE_SYSTEM.md](MODULE_SYSTEM.md) - Module build system and module.json
- [EXTERN_FFI.md](EXTERN_FFI.md) - C FFI and external functions
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Syntax cheat sheet
- [SPECIFICATION.md](SPECIFICATION.md) - Complete language specification

---

**Last Updated:** January 25, 2026
**Status:** Complete
**Version:** 0.2.0+
