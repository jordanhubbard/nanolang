# Tutorial 3: Module System

Learn how to organize code into modules, use the standard library, and leverage external libraries.

## What are Modules?

Modules are reusable collections of functions, types, and constants. They help organize code and enable code sharing.

### Module Types

1. **Standard Library** (`std/`) - Built-in utilities
2. **External Modules** (`modules/`) - Community/system libraries  
3. **Your Own Modules** - Project-specific code

## Importing Modules

### Basic Import

```nano
import "std/io/stdio.nano" as IO

fn main() -> int {
    (IO.println "Hello from a module!")
    return 0
}
```

### Multiple Imports

```nano
import "std/io/stdio.nano" as IO
import "std/math/extended.nano" as Math
import "modules/sqlite/sqlite.nano" as DB

fn main() -> int {
    let result: float = (Math.sqrt 16.0)
    (IO.println (IO.float_to_string result))
    return 0
}
```

### Selective Imports

Import specific functions instead of the entire module:

```nano
from "std/math/extended.nano" import sqrt, pow, sin, cos

fn distance(x: float, y: float) -> float {
    return (sqrt (+ (* x x) (* y y)))
}
```

## Standard Library

### stdio - Console I/O

```nano
import "std/io/stdio.nano" as IO

fn demo_stdio() -> void {
    // Printing
    (IO.println "Hello!")
    (IO.print "No newline")
    
    // Type conversion
    let num_str: string = (IO.int_to_string 42)
    let float_str: string = (IO.float_to_string 3.14)
    
    // String building
    let msg: string = (IO.string_concat "Answer: " num_str)
    (IO.println msg)
}
```

### collections/stringbuilder - Efficient String Building

```nano
import "std/collections/stringbuilder.nano" as SB

fn build_message(name: string, age: int) -> string {
    let sb: StringBuilder = (SB.sb_new)
    let sb = (SB.sb_append sb "Name: ")
    let sb = (SB.sb_append sb name)
    let sb = (SB.sb_append sb ", Age: ")
    let sb = (SB.sb_append sb (int_to_string age))
    return (SB.sb_to_string sb)
}
```

### math/extended - Math Operations

```nano
import "std/math/extended.nano" as Math

fn calculate() -> void {
    let distance: float = (Math.sqrt 25.0)  // 5.0
    let power: float = (Math.pow 2.0 8.0)   // 256.0
    let angle: float = (Math.sin 1.57)      // ≈1.0
    
    let max_val: int = (Math.max 10 20)     // 20
    let min_val: int = (Math.min 10 20)     // 10
    let absolute: int = (Math.abs (- 5))    // 5
}
```

## External Modules

### SQLite Database

```nano
import "modules/sqlite/sqlite.nano" as DB

fn create_and_query() -> int {
    // Open database
    let db: opaque = (DB.nl_sqlite3_open "test.db")
    
    // Create table
    let sql: string = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)"
    (DB.nl_sqlite3_exec db sql)
    
    // Insert data
    let insert: string = "INSERT INTO users (name) VALUES ('Alice')"
    (DB.nl_sqlite3_exec db insert)
    
    // Query
    let query: string = "SELECT id, name FROM users"
    let stmt: opaque = (DB.nl_sqlite3_prepare_v2 db query)
    
    while (== (DB.nl_sqlite3_step stmt) DB.SQLITE_ROW) {
        let id: int = (DB.nl_sqlite3_column_int stmt 0)
        let name: string = (DB.nl_sqlite3_column_text stmt 1)
        (println (string_concat (int_to_string id) ": " name))
    }
    
    (DB.nl_sqlite3_finalize stmt)
    (DB.nl_sqlite3_close db)
    return 0
}
```

### Filesystem Operations

```nano
import "modules/filesystem/filesystem.nano" as FS

fn file_demo() -> int {
    // Check if file exists
    if (== (FS.nl_fs_file_exists "data.txt") 1) {
        // Read file
        let content: string = (FS.nl_fs_read_file "data.txt")
        (println content)
        
        // Get file size
        let size: int = (FS.nl_fs_file_size "data.txt")
        (println (string_concat "Size: " (int_to_string size)))
    } else {
        // Write file
        (FS.nl_fs_write_file "data.txt" "Hello, World!")
    }
    
    // List files in directory
    let files: array<string> = (FS.nl_fs_list_files "." ".nano")
    let mut i: int = 0
    while (< i (len files)) {
        (println (get files i))
        set i (+ i 1)
    }
    
    return 0
}
```

### SDL Graphics

```nano
import "modules/sdl/sdl.nano" as SDL

fn graphics_demo() -> int {
    (SDL.SDL_Init SDL.SDL_INIT_VIDEO)
    
    let window: opaque = (SDL.SDL_CreateWindow "Demo" 100 100 800 600 0)
    let renderer: opaque = (SDL.SDL_CreateRenderer window (- 1) 0)
    
    // Main loop
    let mut running: int = 1
    while (== running 1) {
        // Event handling
        let event: opaque = (SDL.SDL_Event_new)
        while (== (SDL.SDL_PollEvent event) 1) {
            if (== (SDL.SDL_Event_get_type event) SDL.SDL_QUIT) {
                let running = 0
            }
        }
        
        // Render
        (SDL.SDL_SetRenderDrawColor renderer 0 0 0 255)
        (SDL.SDL_RenderClear renderer)
        (SDL.SDL_RenderPresent renderer)
    }
    
    (SDL.SDL_DestroyRenderer renderer)
    (SDL.SDL_DestroyWindow window)
    (SDL.SDL_Quit)
    return 0
}
```

## Creating Your Own Modules

### Simple Module

Create `mymath.nano`:

```nano
// mymath.nano - Custom math utilities

pub fn square(x: int) -> int {
    return (* x x)
}

pub fn cube(x: int) -> int {
    return (* x (* x x))
}

fn helper(x: int) -> int {
    // Private function (no 'pub')
    return (+ x 1)
}

shadow square {
    assert (== (square 5) 25)
    assert (== (square 0) 0)
}

shadow cube {
    assert (== (cube 3) 27)
}
```

Use it:

```nano
import "mymath.nano" as Math

fn main() -> int {
    let sq: int = (Math.square 5)    // 25
    let cb: int = (Math.cube 3)      // 27
    // let x = (Math.helper 1)       // ERROR: helper is private
    return 0
}
```

### Module with Types

```nano
// geometry.nano

pub struct Circle {
    radius: float
}

pub struct Rectangle {
    width: float
    height: float
}

pub fn circle_area(c: Circle) -> float {
    return (* 3.14159 (* c.radius c.radius))
}

pub fn rectangle_area(r: Rectangle) -> float {
    return (* r.width r.height)
}

shadow circle_area {
    let c: Circle = Circle { radius: 5.0 }
    let area: float = (circle_area c)
    assert (> area 78.0)
    assert (< area 79.0)
}
```

## Module Visibility

### Public vs Private

```nano
// Only 'pub' items are accessible from outside
pub fn public_function() -> void {
    (println "Everyone can call this")
}

fn private_function() -> void {
    (println "Only this module can call this")
}

pub struct PublicStruct {
    pub field1: int      // Public field
    field2: int          // Private field (default)
}
```

### Re-exports

```nano
// math_utils.nano
from "std/math/extended.nano" import sqrt, pow
pub use "std/math/extended.nano" as Math  // Re-export module under a public alias

pub fn hypotenuse(a: float, b: float) -> float {
    return (sqrt (+ (* a a) (* b b)))
}
```

## Module Organization

### Recommended Project Structure

```
my_project/
├── src/
│   ├── main.nano          # Entry point
│   ├── lib.nano           # Public API
│   ├── models/
│   │   ├── user.nano
│   │   └── post.nano
│   └── utils/
│       ├── string.nano
│       └── math.nano
├── tests/
│   ├── test_user.nano
│   └── test_utils.nano
└── modules/               # External dependencies
    └── (third-party modules)
```

### Multi-File Module Example

**models/user.nano**:
```nano
pub struct User {
    id: int
    username: string
    email: string
}

pub fn create_user(username: string, email: string) -> User {
    return User {
        id: 0
        username: username
        email: email
    }
}
```

**models/post.nano**:
```nano
pub struct Post {
    id: int
    title: string
    content: string
    author_id: int
}

pub fn create_post(title: string, content: string, author: int) -> Post {
    return Post {
        id: 0
        title: title
        content: content
        author_id: author
    }
}
```

**main.nano**:
```nano
import "models/user.nano" as User
import "models/post.nano" as Post

fn main() -> int {
    let user: User.User = (User.create_user "alice" "alice@example.com")
    let post: Post.Post = (Post.create_post "Hello" "Content" user.id)
    
    (println user.username)
    (println post.title)
    return 0
}
```

## Module Search Paths

Nanolang searches for modules in:

1. **Relative to current file** - `import "utils/math.nano"`
2. **std/ directory** - `import "std/io/stdio.nano"`
3. **modules/ directory** - `import "modules/sqlite/sqlite.nano"`
4. **NANO_MODULE_PATH** environment variable

## Best Practices

### 1. One Module Per File

```nano
// ✅ Good: Clear, focused module
// user.nano
pub struct User { ... }
pub fn create_user() -> User { ... }

// ❌ Avoid: Kitchen sink module
// everything.nano - contains User, Post, Comment, Tag, etc.
```

### 2. Use Descriptive Module Names

```nano
// ✅ Clear intent
import "database/connection.nano" as DB
import "utils/string_helpers.nano" as StrUtil

// ❌ Vague names
import "stuff.nano" as X
import "helpers.nano" as H
```

### 3. Document Public APIs

```nano
// Parse a date string in ISO 8601 format
// Returns Option<Date> - Some(date) if valid, None if invalid
pub fn parse_date(input: string) -> Option<Date> {
    // Implementation
}
```

### 4. Keep Modules Cohesive

Each module should have a single responsibility:

- `user.nano` - User management
- `auth.nano` - Authentication
- `database.nano` - Database operations

### 5. Use Shadow Tests in Modules

```nano
pub fn validate_email(email: string) -> bool {
    // Implementation
    return true
}

shadow validate_email {
    assert (== (validate_email "user@example.com") true)
    assert (== (validate_email "invalid") false)
    assert (== (validate_email "") false)
}
```

## Common Patterns

### Service Pattern

```nano
// user_service.nano
import "database.nano" as DB

pub struct UserService {
    db: DB.Connection
}

pub fn new_user_service(conn: DB.Connection) -> UserService {
    return UserService { db: conn }
}

pub fn find_user(service: UserService, id: int) -> Option<User> {
    // Implementation
}
```

### Factory Pattern

```nano
// config.nano
pub struct Config {
    debug: bool
    port: int
}

pub fn production_config() -> Config {
    return Config { debug: false, port: 80 }
}

pub fn development_config() -> Config {
    return Config { debug: true, port: 3000 }
}
```

## Next Steps

Now you understand the module system! Continue to:

- [FFI Guide](../FFI_GUIDE.md) - Call C libraries
- [Multi-File Projects](../MULTI_FILE_PROJECTS.md) - Real-world project structure
- [Module Reference](../MODULES.md) - Available libraries

## Quick Reference

### Import Syntax

```nano
// Named import
import "path/to/module.nano" as Name

// Selective import
from "path/to/module.nano" import func1, func2, Type

// Multiple imports
import "module1.nano" as M1
import "module2.nano" as M2
```

### Visibility Modifiers

```nano
pub fn public_func() -> void { }    // Exported
fn private_func() -> void { }       // Internal only

pub struct PublicStruct { }         // Exported
struct PrivateStruct { }            // Internal only
```

### Module Paths

```nano
"std/io/stdio.nano"                 // Standard library
"modules/sqlite/sqlite.nano"        // External module
"./local_module.nano"               // Relative path
"../parent_module.nano"             // Parent directory
```

Happy modular programming! 🎯

