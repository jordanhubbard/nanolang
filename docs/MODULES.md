# Nanolang Modules Reference

Comprehensive guide to all available modules in nanolang.

## Table of Contents

- [Standard Library](#standard-library)
- [Database](#database)
- [File System](#file-system)
- [Graphics & UI](#graphics--ui)
- [Audio](#audio)
- [Networking](#networking)
- [Math Extensions](#math-extensions)
- [Module System](#module-system)

---

## Standard Library

### std/io/stdio

Standard I/O operations for console input/output.

**Import:**
```nano
import "std/io/stdio.nano" as IO
```

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `println` | `(str: string) -> void` | Print string with newline |
| `print` | `(str: string) -> void` | Print string without newline |
| `int_to_string` | `(n: int) -> string` | Convert integer to string |
| `float_to_string` | `(f: float) -> string` | Convert float to string |
| `string_concat` | `(a: string, b: string) -> string` | Concatenate two strings |

**Example:**
```nano
import "std/io/stdio.nano" as IO

fn main() -> int {
    (IO.println "Hello, nanolang!")
    let count: int = 42
    (IO.println (IO.string_concat "Count: " (IO.int_to_string count)))
    return 0
}
```

### std/collections/stringbuilder

Efficient string building for concatenating multiple strings.

**Import:**
```nano
import "std/collections/stringbuilder.nano" as SB
```

**Functions:**
- `sb_new() -> StringBuilder` - Create new string builder
- `sb_append(sb: StringBuilder, s: string) -> StringBuilder` - Append string
- `sb_to_string(sb: StringBuilder) -> string` - Convert to final string

**Example:**
```nano
import "std/collections/stringbuilder.nano" as SB

fn build_message(name: string, age: int) -> string {
    let sb: StringBuilder = (SB.sb_new)
    let sb = (SB.sb_append sb "Name: ")
    let sb = (SB.sb_append sb name)
    let sb = (SB.sb_append sb ", Age: ")
    return (SB.sb_to_string sb)
}
```

### std/math/extended

Extended math operations beyond basic arithmetic.

**Functions:**
- `abs(x: int) -> int` - Absolute value
- `min(a: int, b: int) -> int` - Minimum of two values
- `max(a: int, b: int) -> int` - Maximum of two values
- `pow(base: float, exp: float) -> float` - Power function
- `sqrt(x: float) -> float` - Square root
- `sin(x: float) -> float` - Sine
- `cos(x: float) -> float` - Cosine
- `tan(x: float) -> float` - Tangent

---

## Database

### sqlite

SQLite embedded database interface.

**Import:**
```nano
import "modules/sqlite/sqlite.nano" as DB
```

**Core Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `nl_sqlite3_open` | `(filename: string) -> opaque` | Open/create database |
| `nl_sqlite3_close` | `(db: opaque) -> int` | Close database connection |
| `nl_sqlite3_exec` | `(db: opaque, sql: string) -> int` | Execute SQL statement |
| `nl_sqlite3_prepare_v2` | `(db: opaque, sql: string) -> opaque` | Prepare SQL statement |
| `nl_sqlite3_step` | `(stmt: opaque) -> int` | Execute prepared statement |
| `nl_sqlite3_finalize` | `(stmt: opaque) -> int` | Finalize statement |

**Query Functions:**
- `nl_sqlite3_column_int(stmt: opaque, col: int) -> int` - Get integer column
- `nl_sqlite3_column_text(stmt: opaque, col: int) -> string` - Get text column
- `nl_sqlite3_column_double(stmt: opaque, col: int) -> float` - Get float column

**Constants:**
- `SQLITE_OK` = 0 - Success
- `SQLITE_ROW` = 100 - Row available
- `SQLITE_DONE` = 101 - No more rows

**Example:**
```nano
import "modules/sqlite/sqlite.nano" as DB

fn create_users_table(db: opaque) -> int {
    let sql: string = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
    let result: int = (DB.nl_sqlite3_exec db sql)
    if (== result DB.SQLITE_OK) {
        return 0
    } else {
        return 1
    }
}

fn insert_user(db: opaque, name: string, age: int) -> int {
    let sql: string = "INSERT INTO users (name, age) VALUES (?, ?)"
    let stmt: opaque = (DB.nl_sqlite3_prepare_v2 db sql)
    (DB.nl_sqlite3_bind_text stmt 1 name)
    (DB.nl_sqlite3_bind_int stmt 2 age)
    let result: int = (DB.nl_sqlite3_step stmt)
    (DB.nl_sqlite3_finalize stmt)
    return result
}

fn query_users(db: opaque) -> int {
    let sql: string = "SELECT id, name, age FROM users"
    let stmt: opaque = (DB.nl_sqlite3_prepare_v2 db sql)
    
    while (== (DB.nl_sqlite3_step stmt) DB.SQLITE_ROW) {
        let id: int = (DB.nl_sqlite3_column_int stmt 0)
        let name: string = (DB.nl_sqlite3_column_text stmt 1)
        let age: int = (DB.nl_sqlite3_column_int stmt 2)
        // Process row...
    }
    
    (DB.nl_sqlite3_finalize stmt)
    return 0
}
```

---

## File System

### filesystem

File and directory operations.

**Import:**
```nano
import "modules/filesystem/filesystem.nano" as FS
```

**File Operations:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `nl_fs_read_file` | `(path: string) -> string` | Read entire file as string |
| `nl_fs_write_file` | `(path: string, content: string) -> int` | Write string to file |
| `nl_fs_file_exists` | `(path: string) -> int` | Check if file exists (1=yes, 0=no) |
| `nl_fs_file_size` | `(path: string) -> int` | Get file size in bytes |
| `nl_fs_is_directory` | `(path: string) -> int` | Check if path is directory |
| `nl_fs_delete_file` | `(path: string) -> int` | Delete file |

**Directory Operations:**
- `nl_fs_list_files(path: string, extension: string) -> array<string>` - List files
- `nl_fs_list_dirs(path: string) -> array<string>` - List directories
- `nl_fs_create_directory(path: string) -> int` - Create directory
- `nl_fs_delete_directory(path: string) -> int` - Delete directory

**Example:**
```nano
import "modules/filesystem/filesystem.nano" as FS
import "std/io/stdio.nano" as IO

fn process_text_files(directory: string) -> int {
    let files: array<string> = (FS.nl_fs_list_files directory ".txt")
    let i: int = 0
    while (< i (len files)) {
        let filename: string = (get files i)
        if (== (FS.nl_fs_file_exists filename) 1) {
            let content: string = (FS.nl_fs_read_file filename)
            (IO.println content)
        }
        let i = (+ i 1)
    }
    return 0
}

fn backup_file(source: string) -> int {
    if (== (FS.nl_fs_file_exists source) 1) {
        let content: string = (FS.nl_fs_read_file source)
        let backup: string = (string_concat source ".bak")
        return (FS.nl_fs_write_file backup content)
    } else {
        return 1
    }
}
```

---

## Graphics & UI

### sdl

SDL2 core functionality for graphics, events, and windowing.

**Import:**
```nano
import "modules/sdl/sdl.nano" as SDL
```

**Initialization:**
- `SDL_Init(flags: int) -> int` - Initialize SDL subsystems
- `SDL_Quit() -> void` - Cleanup SDL

**Window Management:**
- `SDL_CreateWindow(title: string, x: int, y: int, w: int, h: int, flags: int) -> opaque` - Create window
- `SDL_DestroyWindow(window: opaque) -> void` - Destroy window

**Renderer:**
- `SDL_CreateRenderer(window: opaque, index: int, flags: int) -> opaque` - Create renderer
- `SDL_RenderClear(renderer: opaque) -> int` - Clear screen
- `SDL_RenderPresent(renderer: opaque) -> void` - Present rendered frame
- `SDL_SetRenderDrawColor(renderer: opaque, r: int, g: int, b: int, a: int) -> int` - Set draw color
- `SDL_RenderFillRect(renderer: opaque, rect: opaque) -> int` - Fill rectangle

**Events:**
- `SDL_PollEvent(event: opaque) -> int` - Poll for events
- Event types: `SDL_QUIT`, `SDL_KEYDOWN`, `SDL_KEYUP`, `SDL_MOUSEBUTTONDOWN`, etc.

**Example:**
```nano
import "modules/sdl/sdl.nano" as SDL

fn main() -> int {
    (SDL.SDL_Init SDL.SDL_INIT_VIDEO)
    
    let window: opaque = (SDL.SDL_CreateWindow "My Window" 100 100 800 600 0)
    let renderer: opaque = (SDL.SDL_CreateRenderer window (- 1) SDL.SDL_RENDERER_ACCELERATED)
    
    let running: int = 1
    while (== running 1) {
        // Event handling
        let event: opaque = (SDL.SDL_Event_new)
        while (== (SDL.SDL_PollEvent event) 1) {
            let type: int = (SDL.SDL_Event_get_type event)
            if (== type SDL.SDL_QUIT) {
                let running = 0
            }
        }
        
        // Rendering
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

### sdl_helpers

High-level SDL wrappers for common operations.

**Import:**
```nano
import "modules/sdl_helpers/sdl_helpers.nano" as SDL
```

**Simplified Functions:**
- `nl_sdl_render_fill_rect(renderer, x, y, w, h) -> int` - Draw filled rectangle
- `nl_sdl_render_draw_line(renderer, x1, y1, x2, y2) -> int` - Draw line
- `nl_sdl_render_draw_circle(renderer, x, y, radius) -> int` - Draw circle

### sdl_ttf

TrueType font rendering for SDL.

**Import:**
```nano
import "modules/sdl_ttf/sdl_ttf.nano" as TTF
```

**Functions:**
- `TTF_Init() -> int` - Initialize TTF
- `TTF_OpenFont(file: string, ptsize: int) -> opaque` - Load font
- `nl_render_text_blended_to_texture(renderer, font, text, r, g, b, a) -> int` - Render text to texture
- `nl_draw_text_blended(renderer, font, text, x, y, r, g, b, a) -> int` - Draw text at position

### ui_widgets

High-level UI components built on SDL.

**Widgets:**
- `nl_ui_button(renderer, x, y, w, h, text, clicked) -> int` - Interactive button
- `nl_ui_label(renderer, x, y, text, font) -> void` - Text label
- `nl_ui_text_input(renderer, x, y, w, h, buffer, max_len) -> int` - Text input field
- `nl_ui_scrollable_list(renderer, x, y, w, h, items, selected_index) -> int` - Scrollable list

---

## Audio

### sdl_mixer

SDL_mixer for audio playback and mixing.

**Import:**
```nano
import "modules/sdl_mixer/sdl_mixer.nano" as Mix
```

**Functions:**
- `Mix_OpenAudio(frequency, format, channels, chunksize) -> int` - Initialize audio
- `Mix_LoadWAV(file: string) -> opaque` - Load WAV file
- `Mix_PlayChannel(channel, chunk, loops) -> int` - Play sound
- `Mix_Music_Load(file: string) -> opaque` - Load music file
- `Mix_PlayMusic(music, loops) -> int` - Play music
- `Mix_CloseAudio() -> void` - Cleanup audio

---

## Networking

### curl

HTTP client using libcurl.

**Import:**
```nano
import "modules/curl/curl.nano" as CURL
```

**Functions:**
- `nl_curl_get(url: string) -> string` - Simple GET request
- `nl_curl_post(url: string, data: string) -> string` - POST request
- `curl_easy_init() -> opaque` - Initialize CURL handle
- `curl_easy_setopt(handle, option, value) -> int` - Set option
- `curl_easy_perform(handle) -> int` - Perform request

---

## Math Extensions

### vector2d

2D vector math operations.

**Import:**
```nano
import "std/math/vector2d.nano" as Vec2
import "modules/vector2d/vector2d.nano" as Vec2
```

**Struct:**
```nano
struct Vec2 {
    x: float
    y: float
}
```

**Functions:**
- `vec2_add(a: Vec2, b: Vec2) -> Vec2` - Vector addition
- `vec2_sub(a: Vec2, b: Vec2) -> Vec2` - Vector subtraction
- `vec2_mul(v: Vec2, scalar: float) -> Vec2` - Scalar multiplication
- `vec2_dot(a: Vec2, b: Vec2) -> float` - Dot product
- `vec2_length(v: Vec2) -> float` - Vector length/magnitude
- `vec2_normalize(v: Vec2) -> Vec2` - Normalize vector

---

## Module System

### Creating Modules

Modules are directories in `modules/` containing:
1. `*.nano` - Nanolang interface files
2. `*.c` - C implementation (optional)
3. `module.json` - Build metadata

**module.json Example:**
```json
{
    "name": "my_module",
    "version": "1.0.0",
    "description": "My custom module",
    "c_sources": ["my_module.c"],
    "system_libs": ["pthread"],
    "pkg_config": ["glib-2.0"]
}
```

### Importing Modules

**Absolute Import:**
```nano
import "modules/sqlite/sqlite.nano" as DB
```

**Relative Import:**
```nano
import "std/io/stdio.nano" as IO
```

**Selective Import:**
```nano
from "std/math/extended.nano" use (sqrt, pow, sin)
```

---

## Additional Modules

- **ncurses** - Terminal UI library
- **preferences** - Persistent key-value storage
- **glfw** - Modern OpenGL window management
- **glew** - OpenGL extension wrangling
- **glut** - OpenGL utility toolkit
- **uv** - libuv async I/O
- **onnx** - ONNX machine learning runtime

For detailed documentation on these modules, see their respective README.md files in `modules/<module_name>/`.

---

## Best Practices

1. **Always check return values** - Most C FFI functions return error codes
2. **Close resources** - Call cleanup functions (sqlite3_close, SDL_DestroyWindow, etc.)
3. **Use opaque types** - For C pointers (SDL_Window*, etc.)
4. **Import with aliases** - Prevents namespace collisions (as IO, as DB, etc.)
5. **Check module.json** - For build dependencies and system requirements

---

## Getting Help

- Main documentation: `docs/FEATURES.md`
- Example programs: `examples/`
- Module-specific docs: `modules/<name>/README.md`
- Build system: `docs/MODULE_FORMAT.md`

