# nanolang OS Standard Library

This document specifies the OS standard library for nanolang, modeled after Python's `os` module.

## Design Principles

1. **POSIX-compatible**: Works on Unix-like systems (Linux, macOS, BSD)
2. **Error handling**: Functions return error codes (0 = success, non-zero = error)
3. **Type-safe**: Uses nanolang's type system (int, float, bool, string, void)
4. **No arrays yet**: String-based returns for lists (newline-separated)

## File Operations

### file_read
Read entire file contents as a string.

```nano
fn file_read(path: string) -> string
```

**Returns**: File contents as string, or empty string on error
**Example**: `let content: string = (file_read "data.txt")`

### file_write
Write string to file (overwrites existing content).

```nano
fn file_write(path: string, content: string) -> int
```

**Returns**: 0 on success, -1 on error
**Example**: `assert (== (file_write "output.txt" "Hello") 0)`

### file_append
Append string to file.

```nano
fn file_append(path: string, content: string) -> int
```

**Returns**: 0 on success, -1 on error
**Example**: `(file_append "log.txt" "New entry\n")`

### file_remove
Delete a file.

```nano
fn file_remove(path: string) -> int
```

**Returns**: 0 on success, -1 on error
**Example**: `(file_remove "temp.txt")`

### file_rename
Rename or move a file.

```nano
fn file_rename(old_path: string, new_path: string) -> int
```

**Returns**: 0 on success, -1 on error
**Example**: `(file_rename "old.txt" "new.txt")`

### file_exists
Check if file exists.

```nano
fn file_exists(path: string) -> bool
```

**Returns**: true if file exists, false otherwise
**Example**: `if (file_exists "config.txt") { ... }`

### file_size
Get file size in bytes.

```nano
fn file_size(path: string) -> int
```

**Returns**: File size in bytes, or -1 on error
**Example**: `let size: int = (file_size "data.bin")`

## Directory Operations

### dir_create
Create a directory.

```nano
fn dir_create(path: string) -> int
```

**Returns**: 0 on success, -1 on error
**Example**: `(dir_create "output")`

### dir_remove
Remove an empty directory.

```nano
fn dir_remove(path: string) -> int
```

**Returns**: 0 on success, -1 on error
**Example**: `(dir_remove "temp")`

### dir_list
List directory contents (newline-separated string).

```nano
fn dir_list(path: string) -> string
```

**Returns**: Newline-separated list of filenames, or empty string on error
**Example**: `let files: string = (dir_list ".")`

### dir_exists
Check if directory exists.

```nano
fn dir_exists(path: string) -> bool
```

**Returns**: true if directory exists, false otherwise
**Example**: `if (dir_exists "data") { ... }`

### getcwd
Get current working directory.

```nano
fn getcwd() -> string
```

**Returns**: Current directory path
**Example**: `let cwd: string = (getcwd)`

### chdir
Change current working directory.

```nano
fn chdir(path: string) -> int
```

**Returns**: 0 on success, -1 on error
**Example**: `(chdir "/tmp")`

## Path Operations

### path_isfile
Check if path is a file.

```nano
fn path_isfile(path: string) -> bool
```

**Returns**: true if path is a regular file, false otherwise
**Example**: `if (path_isfile "data.txt") { ... }`

### path_isdir
Check if path is a directory.

```nano
fn path_isdir(path: string) -> bool
```

**Returns**: true if path is a directory, false otherwise
**Example**: `if (path_isdir "src") { ... }`

### path_join
Join two path components.

```nano
fn path_join(a: string, b: string) -> string
```

**Returns**: Joined path with appropriate separator
**Example**: `let full: string = (path_join "/home" "user")`
**Result**: `"/home/user"`

### path_basename
Get the base name of a path.

```nano
fn path_basename(path: string) -> string
```

**Returns**: Last component of path
**Example**: `(path_basename "/home/user/file.txt")` → `"file.txt"`

### path_dirname
Get the directory name of a path.

```nano
fn path_dirname(path: string) -> string
```

**Returns**: Directory portion of path
**Example**: `(path_dirname "/home/user/file.txt")` → `"/home/user"`

## Process Operations

### system
Execute a shell command.

```nano
fn system(command: string) -> int
```

**Returns**: Command exit code
**Example**: `let result: int = (system "ls -l")`

### exit
Exit the program with a status code.

```nano
fn exit(code: int) -> void
```

**Returns**: Never returns (terminates program)
**Example**: `(exit 1)`

### getenv
Get environment variable value.

```nano
fn getenv(name: string) -> string
```

**Returns**: Environment variable value, or empty string if not set
**Example**: `let home: string = (getenv "HOME")`

## Error Handling

Since nanolang doesn't have exceptions, we use return codes:
- **0**: Success
- **-1**: Error (check errno in C for details)

For functions returning strings, empty string indicates error.
For functions returning bool, false may indicate "not found" or error depending on function.

## Example: File Processing Program

```nano
fn process_file() -> int {
    # Check if file exists
    if (not (file_exists "input.txt")) {
        print "Error: input.txt not found"
        return 1
    }

    # Read file
    let content: string = (file_read "input.txt")

    # Get file size
    let size: int = (file_size "input.txt")
    print "File size: "
    print size
    print " bytes"

    # Write to output
    let result: int = (file_write "output.txt" content)
    if (!= result 0) {
        print "Error writing file"
        return 1
    }

    return 0
}

shadow process_file {
    # Create test file
    assert (== (file_write "test_input.txt" "test data") 0)
    assert (== (process_file) 0)
    assert (file_exists "test_output.txt")
    # Cleanup
    (file_remove "test_input.txt")
    (file_remove "test_output.txt")
}
```

## Implementation Notes

- All functions are implemented as built-in functions (not user-defined)
- C transpiler generates calls to POSIX functions (open, read, write, stat, etc.)
- Interpreter calls C stdlib functions directly
- Memory management for strings handled by runtime
- Platform-specific: Requires POSIX-compatible OS

## Future Enhancements

When arrays are added:
- `dir_list` could return `array<string>`
- `file_readlines` could return `array<string>`
- `path_split` could return `array<string>`

When error types are added:
- Functions could return `Result<T, Error>`
- Better error messages and handling
