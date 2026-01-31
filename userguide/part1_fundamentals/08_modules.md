# Chapter 8: Modules & Imports

**Learn how to organize code with NanoLang's module system.**

Modules let you organize code into reusable units. This chapter covers how to import modules, use selective imports, and create your own modules.

## 8.1 Importing Modules

NanoLang supports two import styles: full module imports and selective imports.

### Basic Import Syntax

**Full module import with alias:**

```nano
import "path/to/module.nano" as alias

fn use_module() -> int {
    return (alias.function_name arg)
}
```

**Example:**

```nano
import "modules/std/json/json.nano" as json

fn parse_json_data(text: string) -> Json {
    return (json.parse text)
}

shadow parse_json_data {
    # Requires json module for real test
    assert true
}
```

### Module Paths

Paths are relative to the project root:

```nano
# Standard library module
import "stdlib/log.nano" as log

# External module
import "modules/sdl/sdl.nano" as sdl

# Local module in same directory
import "helper.nano" as helper

# Module in subdirectory
import "utils/math.nano" as math
```

### Standard Library Imports

Common standard library modules:

```nano
import "stdlib/log.nano" as log
import "stdlib/StringBuilder.nano" as sb
import "stdlib/regex.nano" as regex
import "stdlib/coverage.nano" as coverage
```

### External Module Imports

External modules live in `modules/` directory:

```nano
# Graphics
import "modules/sdl/sdl.nano" as sdl

# Database  
import "modules/sqlite/sqlite.nano" as sqlite

# HTTP
import "modules/curl/curl.nano" as curl

# Terminal UI
import "modules/ncurses/ncurses.nano" as ncurses
```

### Using Imported Functions

Call functions with the module alias:

```nano
import "stdlib/log.nano" as log

fn logging_example() -> void {
    (log.log_info "app" "Application started")
    (log.log_debug "app" "Debug message")
}

shadow logging_example {
    (logging_example)
}
```

## 8.2 Selective Imports (from...import)

Import specific functions without needing a module alias.

### Importing Specific Functions

```nano
from "module/path.nano" import function1, function2, function3

fn use_functions() -> int {
    return (function1 arg)  # No prefix needed
}
```

### Example: Log Functions

```nano
from "stdlib/log.nano" import log_info, log_debug, log_error

fn app_startup() -> void {
    (log_info "app" "Starting application")
    (log_debug "app" "Loading configuration")
    (log_info "app" "Ready")
}

shadow app_startup {
    (app_startup)
}
```

### When to Use Selective Imports

**Use selective imports when:**
- You only need a few functions
- Function names are unique and clear
- You want cleaner call sites

```nano
from "stdlib/regex.nano" import compile, matches, free

fn validate_email(email: string) -> bool {
    let pattern: Regex = (compile "[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-z]+")
    let result: bool = (matches pattern email)
    (free pattern)
    return result
}

shadow validate_email {
    assert (validate_email "user@example.com")
    assert (not (validate_email "invalid"))
}
```

**Use module imports when:**
- Importing many functions
- Risk of name collisions
- Want to show where functions come from

```nano
import "stdlib/log.nano" as log

fn use_many_log_functions() -> void {
    (log.log_trace "cat" "msg")
    (log.log_debug "cat" "msg")
    (log.log_info "cat" "msg")
    (log.log_warn "cat" "msg")
    (log.log_error "cat" "msg")
}

shadow use_many_log_functions {
    (use_many_log_functions)
}
```

### Examples

**Filesystem operations:**

```nano
from "modules/std/fs.nano" import read, write, exists, mkdir

fn save_data(path: string, data: string) -> bool {
    if (not (exists (dirname path))) {
        (mkdir (dirname path))
    }
    (write path data)
    return true
}

shadow save_data {
    # Would test with actual file operations
    assert true
}
```

**String operations:**

```nano
from "stdlib/StringBuilder.nano" import sb_new, sb_append, sb_to_string

fn build_html(title: string, body: string) -> string {
    let sb: StringBuilder = (sb_new)
    (sb_append sb "<html><head><title>")
    (sb_append sb title)
    (sb_append sb "</title></head><body>")
    (sb_append sb body)
    (sb_append sb "</body></html>")
    return (sb_to_string sb)
}

shadow build_html {
    let html: string = (build_html "Test" "Content")
    assert (str_contains html "<title>Test</title>")
}
```

## 8.3 Creating Your Own Modules

Any NanoLang file can be a module. Simply define functions and import them from other files.

### Module File Structure

**my_math.nano:**

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
}

fn multiply(a: int, b: int) -> int {
    return (* a b)
}

shadow multiply {
    assert (== (multiply 3 4) 12)
}
```

### Exporting Functions

All functions in a module are automatically exported. There's no `export` keyword - if it's defined, it's available.

### Public vs Private Functions

**Currently:** All functions are public.

**Future:** Private functions may use a naming convention (e.g., `_private_function`).

**Workaround:** Use separate internal modules for private functions.

### Examples

**Creating a utilities module:**

**utils.nano:**

```nano
fn clamp(value: int, min_val: int, max_val: int) -> int {
    if (< value min_val) { return min_val }
    if (> value max_val) { return max_val }
    return value
}

shadow clamp {
    assert (== (clamp 5 0 10) 5)
    assert (== (clamp -5 0 10) 0)
    assert (== (clamp 15 0 10) 10)
}

fn lerp(a: float, b: float, t: float) -> float {
    return (+ a (* t (- b a)))
}

shadow lerp {
    assert (== (lerp 0.0 10.0 0.5) 5.0)
    assert (== (lerp 0.0 10.0 0.0) 0.0)
    assert (== (lerp 0.0 10.0 1.0) 10.0)
}
```

**Using the module:**

**main.nano:**

```nano
from "utils.nano" import clamp, lerp

fn main() -> int {
    let clamped: int = (clamp 100 0 50)
    (println (int_to_string clamped))
    
    let interpolated: float = (lerp 0.0 100.0 0.25)
    (println (float_to_string interpolated))
    
    return 0
}

shadow main { assert true }
```

### Module Dependencies

Modules can import other modules:

**parser.nano:**

```nano
from "lexer.nano" import tokenize, Token

fn parse(input: string) -> AST {
    let tokens: array<Token> = (tokenize input)
    # Parse tokens into AST
    return ast
}

shadow parse {
    assert true
}
```

## 8.4 Module Structure & Best Practices

How to organize larger projects.

### Project Organization

```
myproject/
├── main.nano           # Entry point
├── lib/
│   ├── parser.nano     # Parser module
│   ├── lexer.nano      # Lexer module
│   └── ast.nano        # AST definitions
├── utils/
│   ├── string.nano     # String utilities
│   └── array.nano      # Array utilities
└── tests/
    └── test_parser.nano
```

**main.nano:**

```nano
from "lib/parser.nano" import parse
from "lib/lexer.nano" import tokenize

fn main() -> int {
    let input: string = "print 42"
    let ast: AST = (parse input)
    return 0
}

shadow main { assert true }
```

### Module Dependencies

**Principle:** Keep dependencies acyclic.

```
✅ Good (acyclic):
main.nano → parser.nano → lexer.nano → ast.nano

❌ Bad (circular):
parser.nano → lexer.nano → parser.nano
```

### Avoiding Circular Dependencies

**Strategy 1: Extract common definitions**

```
parser.nano ↘
              ast.nano (common types)
lexer.nano  ↗
```

**Strategy 2: One-way dependencies**

```
high_level.nano → low_level.nano
(never the reverse)
```

**Strategy 3: Use interfaces**

Define interfaces/types in separate module that both depend on.

### Best Practices

**1. One module per file**

```
✅ Good:
- math.nano
- string_utils.nano
- parser.nano

❌ Bad:
- utils.nano (too general)
- all_helpers.nano (kitchen sink)
```

**2. Clear naming**

```nano
✅ Good names:
- json_parser.nano
- http_client.nano
- file_utils.nano

❌ Unclear names:
- helpers.nano
- misc.nano
- stuff.nano
```

**3. Minimal exports**

Only define functions you want to be public. Keep helper functions in the same file.

**4. Import what you use**

```nano
✅ Good:
from "module.nano" import specific_function

❌ Less good:
import "module.nano" as m
# Then only use m.one_function
```

### Example: Multi-Module Project

**types.nano:**

```nano
struct User {
    id: int,
    name: string,
    email: string
}

struct Post {
    id: int,
    author_id: int,
    title: string,
    content: string
}
```

**database.nano:**

```nano
from "types.nano" import User, Post

fn save_user(u: User) -> bool {
    # Save to database
    return true
}

shadow save_user {
    let user: User = User { id: 1, name: "Alice", email: "alice@example.com" }
    assert (save_user user)
}

fn find_user(id: int) -> User {
    # Query database
    return User { id: id, name: "Unknown", email: "" }
}

shadow find_user {
    let user: User = (find_user 1)
    assert (== user.id 1)
}
```

**app.nano:**

```nano
from "types.nano" import User, Post
from "database.nano" import save_user, find_user

fn create_and_save_user(name: string, email: string) -> bool {
    let user: User = User { id: 0, name: name, email: email }
    return (save_user user)
}

shadow create_and_save_user {
    assert (create_and_save_user "Bob" "bob@example.com")
}

fn main() -> int {
    let user: User = (find_user 1)
    (println user.name)
    return 0
}

shadow main { assert true }
```

### Summary

In this chapter, you learned:
- ✅ `import "path" as alias` for full module imports
- ✅ `from "path" import func` for selective imports
- ✅ Module paths are relative to project root
- ✅ All functions are automatically exported
- ✅ Organize projects with subdirectories
- ✅ Avoid circular dependencies

### Practice Exercises

**Exercise 1: Create a geometry module**

**geometry.nano:**

```nano
struct Circle {
    radius: float
}

struct Rectangle {
    width: float,
    height: float
}

fn circle_area(c: Circle) -> float {
    return (* 3.14159 (* c.radius c.radius))
}

shadow circle_area {
    let c: Circle = Circle { radius: 5.0 }
    let area: float = (circle_area c)
    assert (and (> area 78.5) (< area 78.6))
}

fn rectangle_area(r: Rectangle) -> float {
    return (* r.width r.height)
}

shadow rectangle_area {
    let r: Rectangle = Rectangle { width: 4.0, height: 5.0 }
    assert (== (rectangle_area r) 20.0)
}
```

**Exercise 2: Use the geometry module**

**main.nano:**

```nano
from "geometry.nano" import Circle, Rectangle, circle_area, rectangle_area

fn compare_areas() -> bool {
    let c: Circle = Circle { radius: 3.0 }
    let r: Rectangle = Rectangle { width: 5.0, height: 5.0 }
    
    let area_c: float = (circle_area c)
    let area_r: float = (rectangle_area r)
    
    return (> area_r area_c)
}

shadow compare_areas {
    assert (compare_areas)
}

fn main() -> int {
    if (compare_areas) {
        (println "Rectangle is larger")
    }
    return 0
}

shadow main { assert true }
```

---

**Previous:** [Chapter 7: Data Structures](07_data_structures.html)  
**Next:** [Chapter 9: Core Utilities](../part2_stdlib/09_core_utilities.html)
