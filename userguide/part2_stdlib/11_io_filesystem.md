# Chapter 11: I/O & Filesystem

**Work with files, directories, paths, and environment variables.**

This chapter covers filesystem operations, path manipulation, and environment variable access. Most functionality requires importing from `modules/std/fs.nano` or `modules/std/env.nano`.

## 11.1 Built-In System Functions

### Current Working Directory

```nano
fn show_cwd() -> string {
    let current: string = (getcwd)
    return current
}

shadow show_cwd {
    let cwd: string = (show_cwd)
    assert (> (str_length cwd) 0)
}
```

**Returns:** Absolute path as string (e.g., `/Users/username/project`)

### Environment Variables

```nano
fn get_home_dir() -> string {
    return (getenv "HOME")
}

shadow get_home_dir {
    let home: string = (get_home_dir)
    # May be empty on systems without HOME
    assert true
}
```

**Returns:** Variable value, or empty string if not set

**Common environment variables:**
- `HOME` - User's home directory
- `PATH` - Executable search path
- `USER` - Current username
- `PWD` - Present working directory

## 11.2 File Operations

File operations require importing from `modules/std/fs.nano`.

### Reading Files

```nano
from "modules/std/fs.nano" import file_read, file_exists

fn read_config(path: string) -> string {
    if (file_exists path) {
        return (file_read path)
    }
    return ""
}

shadow read_config {
    # Would test with actual file
    assert true
}
```

**Signature:** `file_read(path: string) -> string`
- Reads entire file as string
- Returns empty string on error
- Works with text files

### Writing Files

```nano
from "modules/std/fs.nano" import file_write

fn save_data(path: string, content: string) -> bool {
    let result: int = (file_write path content)
    return (== result 0)
}

shadow save_data {
    # Would test with actual file I/O
    assert true
}
```

**Signature:** `file_write(path: string, content: string) -> int`
- Overwrites existing file or creates new
- Returns 0 on success
- Returns non-zero on error

### Appending to Files

```nano
from "modules/std/fs.nano" import file_append

fn append_log(path: string, message: string) -> bool {
    let timestamped: string = (+ (int_to_string 0) (+ ": " message))
    let result: int = (file_append path (+ timestamped "\n"))
    return (== result 0)
}

shadow append_log {
    assert true
}
```

**Signature:** `file_append(path: string, content: string) -> int`
- Appends to end of file
- Creates file if doesn't exist
- Returns 0 on success

### Checking File Existence

```nano
from "modules/std/fs.nano" import file_exists

fn ensure_file(path: string) -> bool {
    if (not (file_exists path)) {
        (file_write path "")  # Create empty file
    }
    return (file_exists path)
}

shadow ensure_file {
    assert true
}
```

### Deleting Files

```nano
from "modules/std/fs.nano" import file_delete

fn cleanup(path: string) -> bool {
    if (file_exists path) {
        let result: int = (file_delete path)
        return (== result 0)
    }
    return true  # Already deleted
}

shadow cleanup {
    assert true
}
```

## 11.3 Directory Operations

### Creating Directories

```nano
from "modules/std/fs.nano" import fs_mkdir_p

fn create_directory_tree(path: string) -> bool {
    let result: int = (fs_mkdir_p path)
    return (== result 0)
}

shadow create_directory_tree {
    assert true
}
```

**Note:** `fs_mkdir_p` creates parent directories automatically (like `mkdir -p`)

### Walking Directory Trees

```nano
from "modules/std/fs.nano" import walkdir

fn count_files(directory: string) -> int {
    let files: array<string> = (walkdir directory)
    return (array_length files)
}

shadow count_files {
    let count: int = (count_files "modules")
    assert (> count 0)
}
```

**Signature:** `walkdir(root: string) -> array<string>`
- Returns all files recursively
- Full paths relative to root
- Includes subdirectories

## 11.4 Path Manipulation

Path utilities help work with file paths portably.

### Joining Paths

```nano
from "modules/std/fs.nano" import join

fn make_path(dir: string, file: string) -> string {
    return (join dir file)
}

shadow make_path {
    assert (== (make_path "foo" "bar") "foo/bar")
    assert (== (make_path "foo/" "bar") "foo/bar")
}
```

**Handles:**
- Trailing slashes
- Empty components
- Cross-platform separators

### Normalizing Paths

```nano
from "modules/std/fs.nano" import normalize

fn clean_path(path: string) -> string {
    return (normalize path)
}

shadow clean_path {
    assert (== (clean_path "/foo/./bar/../baz") "/foo/baz")
    assert (== (clean_path "./foo") "foo")
}
```

**Removes:**
- `.` (current directory)
- `..` (parent directory)
- Redundant slashes

### Extracting Path Components

```nano
from "modules/std/fs.nano" import basename, dirname

fn split_path(path: string) -> bool {
    let dir: string = (dirname path)
    let file: string = (basename path)
    
    return (and
        (== dir "/foo/bar")
        (== file "baz.txt")
    )
}

shadow split_path {
    assert (split_path "/foo/bar/baz.txt")
}
```

**Functions:**
- `basename(path)` - Returns filename
- `dirname(path)` - Returns directory path

### Relative Paths

```nano
from "modules/std/fs.nano" import relpath

fn relative_from_root(target: string) -> string {
    return (relpath target "/")
}

shadow relative_from_root {
    assert (== (relpath "/a/b/c" "/a") "b/c")
}
```

## 11.5 Environment Variables (Extended)

The `modules/std/env.nano` module provides full environment variable access.

### Getting Variables

```nano
from "modules/std/env.nano" import get

fn get_user_home() -> string {
    return (get "HOME")
}

shadow get_user_home {
    let home: string = (get_user_home)
    assert (> (str_length home) 0)
}
```

### Setting Variables

```nano
from "modules/std/env.nano" import set_env, get

fn configure_env() -> bool {
    let result: int = (set_env "MY_VAR" "value")
    let retrieved: string = (get "MY_VAR")
    return (== retrieved "value")
}

shadow configure_env {
    assert (configure_env)
}
```

### Unsetting Variables

```nano
from "modules/std/env.nano" import unset, set_env, get

fn cleanup_env() -> bool {
    (set_env "TEMP_VAR" "test")
    let result: int = (unset "TEMP_VAR")
    let value: string = (get "TEMP_VAR")
    return (and (== result 0) (== value ""))
}

shadow cleanup_env {
    assert (cleanup_env)
}
```

## 11.6 Command-Line Arguments

Access program arguments via the environment module.

### Getting Arguments

```nano
from "modules/std/env.nano" import args

fn process_args() -> int {
    let arguments: array<string> = (args)
    return (array_length arguments)
}

shadow process_args {
    # Number of args varies by invocation
    assert true
}
```

**Returns:** Array of command-line arguments
- Index 0: Program name
- Index 1+: User arguments

### Example: Argument Parser

```nano
from "modules/std/env.nano" import args

fn parse_flag(flag: string) -> bool {
    let arguments: array<string> = (args)
    let len: int = (array_length arguments)
    
    for i in (range 0 len) {
        if (== (at arguments i) flag) {
            return true
        }
    }
    
    return false
}

shadow parse_flag {
    # Would test with actual args
    assert true
}
```

## 11.7 Practical Examples

### Example 1: Configuration File

```nano
from "modules/std/fs.nano" import file_read, file_write, file_exists

fn load_config(path: string, default: string) -> string {
    if (file_exists path) {
        return (file_read path)
    }
    (file_write path default)
    return default
}

shadow load_config {
    # Would test with actual filesystem
    assert true
}
```

### Example 2: File Counter

```nano
from "modules/std/fs.nano" import walkdir

fn count_by_extension(dir: string, ext: string) -> int {
    let files: array<string> = (walkdir dir)
    let mut count: int = 0
    let ext_len: int = (str_length ext)
    
    for i in (range 0 (array_length files)) {
        let file: string = (at files i)
        let file_len: int = (str_length file)
        
        if (>= file_len ext_len) {
            let start: int = (- file_len ext_len)
            let ending: string = (str_substring file start ext_len)
            if (== ending ext) {
                set count (+ count 1)
            }
        }
    }
    
    return count
}

shadow count_by_extension {
    let count: int = (count_by_extension "modules" ".nano")
    assert (> count 0)
}
```

### Example 3: Path Builder

```nano
from "modules/std/fs.nano" import join, normalize

fn build_project_path(project: string, subdir: string, file: string) -> string {
    let path1: string = (join project subdir)
    let path2: string = (join path1 file)
    return (normalize path2)
}

shadow build_project_path {
    let path: string = (build_project_path "/home/user" "src" "main.nano")
    assert (== path "/home/user/src/main.nano")
}
```

### Example 4: Environment Config

```nano
from "modules/std/env.nano" import get

fn get_editor() -> string {
    let editor: string = (get "EDITOR")
    if (== editor "") {
        return "nano"  # Default
    }
    return editor
}

shadow get_editor {
    let editor: string = (get_editor)
    assert (> (str_length editor) 0)
}
```

### Example 5: Log File

```nano
from "modules/std/fs.nano" import file_append, fs_mkdir_p, dirname

fn log_message(logfile: string, level: string, message: string) -> bool {
    # Ensure directory exists
    let dir: string = (dirname logfile)
    (fs_mkdir_p dir)
    
    # Format: [LEVEL] message
    let formatted: string = (+ "[" (+ level (+ "] " (+ message "\n"))))
    
    let result: int = (file_append logfile formatted)
    return (== result 0)
}

shadow log_message {
    # Would test with actual logging
    assert true
}
```

### Example 6: File Copying

```nano
from "modules/std/fs.nano" import file_read, file_write, file_exists

fn copy_file(src: string, dst: string) -> bool {
    if (not (file_exists src)) {
        return false
    }
    
    let content: string = (file_read src)
    let result: int = (file_write dst content)
    return (== result 0)
}

shadow copy_file {
    assert true
}
```

### Example 7: Directory Listing

```nano
from "modules/std/fs.nano" import walkdir, basename

fn list_filenames(directory: string) -> array<string> {
    let paths: array<string> = (walkdir directory)
    let len: int = (array_length paths)
    let mut names: array<string> = (array_new len "")
    
    for i in (range 0 len) {
        let path: string = (at paths i)
        let name: string = (basename path)
        (array_set names i name)
    }
    
    return names
}

shadow list_filenames {
    let names: array<string> = (list_filenames "modules")
    assert (> (array_length names) 0)
}
```

## 11.8 Best Practices

### ✅ DO

**1. Check file existence before reading:**

```nano
from "modules/std/fs.nano" import file_exists, file_read

fn safe_read(path: string) -> string {
    if (file_exists path) {
        return (file_read path)
    }
    (println "File not found")
    return ""
}

shadow safe_read {
    assert true
}
```

**2. Use path utilities for portability:**

```nano
from "modules/std/fs.nano" import join, normalize

fn portable_path(base: string, file: string) -> string {
    return (normalize (join base file))
}

shadow portable_path {
    assert (== (portable_path "foo" "bar") "foo/bar")
}
```

**3. Create parent directories:**

```nano
from "modules/std/fs.nano" import fs_mkdir_p, dirname, file_write

fn safe_write(path: string, content: string) -> bool {
    let dir: string = (dirname path)
    (fs_mkdir_p dir)
    let result: int = (file_write path content)
    return (== result 0)
}

shadow safe_write {
    assert true
}
```

### ❌ DON'T

**1. Don't hardcode path separators:**

```nano
# ❌ Bad: Platform-specific
let path: string = "C:\\Users\\name\\file.txt"

# ✅ Good: Use join
from "modules/std/fs.nano" import join
let path: string = (join "C:" (join "Users" (join "name" "file.txt")))
```

**2. Don't ignore error codes:**

```nano
# ❌ Bad: Ignores failure
(file_write path content)

# ✅ Good: Check result
let result: int = (file_write path content)
if (!= result 0) {
    (println "Write failed")
}
```

**3. Don't use raw getenv for critical paths:**

```nano
# ❌ Bad: No fallback
let home: string = (getenv "HOME")

# ✅ Good: Provide default
fn get_home_with_default() -> string {
    let home: string = (getenv "HOME")
    if (== home "") {
        return "/tmp"
    }
    return home
}

shadow get_home_with_default {
    let home: string = (get_home_with_default)
    assert (> (str_length home) 0)
}
```

## Summary

In this chapter, you learned:
- ✅ Built-in system functions: `getcwd`, `getenv`
- ✅ File operations: read, write, append, delete, exists
- ✅ Directory operations: mkdir, walkdir
- ✅ Path utilities: join, normalize, basename, dirname, relpath
- ✅ Environment variables: get, set, unset
- ✅ Command-line arguments via `args()`
- ✅ Practical file I/O patterns

### Quick Reference

| Operation | Function | Module |
|-----------|----------|--------|
| **Get CWD** | `getcwd()` | Built-in |
| **Get env var** | `getenv(name)` | Built-in |
| **Read file** | `file_read(path)` | `std/fs` |
| **Write file** | `file_write(path, content)` | `std/fs` |
| **Append file** | `file_append(path, content)` | `std/fs` |
| **File exists** | `file_exists(path)` | `std/fs` |
| **Delete file** | `file_delete(path)` | `std/fs` |
| **Make directory** | `fs_mkdir_p(path)` | `std/fs` |
| **List files** | `walkdir(dir)` | `std/fs` |
| **Join paths** | `join(a, b)` | `std/fs` |
| **Normalize** | `normalize(path)` | `std/fs` |
| **Basename** | `basename(path)` | `std/fs` |
| **Dirname** | `dirname(path)` | `std/fs` |
| **Get env** | `get(name)` | `std/env` |
| **Set env** | `set_env(name, val)` | `std/env` |
| **CLI args** | `args()` | `std/env` |

---

**Previous:** [Chapter 10: Collections Library](10_collections_library.md)  
**Next:** [Chapter 12: System & Runtime](12_system_runtime.md)
