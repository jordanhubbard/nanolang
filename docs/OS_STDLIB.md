# My OS Standard Library

I provide an OS standard library modeled after Python's `os` module. It allows me to interact with the underlying system in a predictable way.

## My Design Principles

1. **POSIX-compatible.** I work on Unix-like systems, including Linux, macOS, and BSD.
2. **Error handling.** My functions return error codes where 0 indicates success and non-zero indicates an error.
3. **Type-safe.** I use my own type system for all operations.
4. **No arrays yet.** I return lists as newline-separated strings for now.

## File Operations

### file_read
I read the entire contents of a file and return it as a string.

```nano
fn file_read(path: string) -> string
```

**Returns**: File contents as string, or empty string on error.
**Example**: `let content: string = (file_read "data.txt")`

### file_write
I write a string to a file. This overwrites any existing content.

```nano
fn file_write(path: string, content: string) -> int
```

**Returns**: 0 on success, -1 on error.
**Example**: `assert (== (file_write "output.txt" "Hello") 0)`

### file_append
I append a string to the end of a file.

```nano
fn file_append(path: string, content: string) -> int
```

**Returns**: 0 on success, -1 on error.
**Example**: `(file_append "log.txt" "New entry\n")`

### file_remove
I delete a file from the system.

```nano
fn file_remove(path: string) -> int
```

**Returns**: 0 on success, -1 on error.
**Example**: `(file_remove "temp.txt")`

### file_rename
I rename or move a file.

```nano
fn file_rename(old_path: string, new_path: string) -> int
```

**Returns**: 0 on success, -1 on error.
**Example**: `(file_rename "old.txt" "new.txt")`

### file_exists
I check if a file exists at the given path.

```nano
fn file_exists(path: string) -> bool
```

**Returns**: true if the file exists, false otherwise.
**Example**: `if (file_exists "config.txt") { ... }`

### file_size
I retrieve the size of a file in bytes.

```nano
fn file_size(path: string) -> int
```

**Returns**: File size in bytes, or -1 on error.
**Example**: `let size: int = (file_size "data.bin")`

## Directory Operations

### dir_create
I create a new directory.

```nano
fn dir_create(path: string) -> int
```

**Returns**: 0 on success, -1 on error.
**Example**: `(dir_create "output")`

### dir_remove
I remove an empty directory.

```nano
fn dir_remove(path: string) -> int
```

**Returns**: 0 on success, -1 on error.
**Example**: `(dir_remove "temp")`

### dir_list
I list the contents of a directory as a newline-separated string.

```nano
fn dir_list(path: string) -> string
```

**Returns**: Newline-separated list of filenames, or empty string on error.
**Example**: `let files: string = (dir_list ".")`

### dir_exists
I check if a directory exists at the given path.

```nano
fn dir_exists(path: string) -> bool
```

**Returns**: true if the directory exists, false otherwise.
**Example**: `if (dir_exists "data") { ... }`

### getcwd
I return my current working directory.

```nano
fn getcwd() -> string
```

**Returns**: Current directory path.
**Example**: `let cwd: string = (getcwd)`

### chdir
I change my current working directory.

```nano
fn chdir(path: string) -> int
```

**Returns**: 0 on success, -1 on error.
**Example**: `(chdir "/tmp")`

## Path Operations

### path_isfile
I check if a path refers to a regular file.

```nano
fn path_isfile(path: string) -> bool
```

**Returns**: true if the path is a regular file, false otherwise.
**Example**: `if (path_isfile "data.txt") { ... }`

### path_isdir
I check if a path refers to a directory.

```nano
fn path_isdir(path: string) -> bool
```

**Returns**: true if the path is a directory, false otherwise.
**Example**: `if (path_isdir "src") { ... }`

### path_join
I join two path components using the appropriate separator for the system.

```nano
fn path_join(a: string, b: string) -> string
```

**Returns**: Joined path string.
**Example**: `let full: string = (path_join "/home" "user")`
**Result**: `"/home/user"`

### path_basename
I extract the base name from a path.

```nano
fn path_basename(path: string) -> string
```

**Returns**: The last component of the path.
**Example**: `(path_basename "/home/user/file.txt")` → `"file.txt"`

### path_dirname
I extract the directory name from a path.

```nano
fn path_dirname(path: string) -> string
```

**Returns**: The directory portion of the path.
**Example**: `(path_dirname "/home/user/file.txt")` → `"/home/user"`

## Process Operations

### system
I execute a shell command on the underlying system.

```nano
fn system(command: string) -> int
```

**Returns**: The exit code of the command.
**Example**: `let result: int = (system "ls -l")`

### exit
I terminate the program with a specific status code.

```nano
fn exit(code: int) -> void
```

**Returns**: I never return from this call.
**Example**: `(exit 1)`

### getenv
I retrieve the value of an environment variable.

```nano
fn getenv(name: string) -> string
```

**Returns**: The value of the variable, or an empty string if it is not set.
**Example**: `let home: string = (getenv "HOME")`

## My Error Handling

I do not have exceptions. I use return codes instead.

- **0**: Success.
- **-1**: Error. You can check errno in C for details.

For functions that return strings, I return an empty string to indicate an error.
For functions that return a boolean, false can mean "not found" or indicate an error depending on the function.

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

## My Implementation Notes

- I implement all these functions as built-ins, not user-defined functions.
- My C transpiler generates calls to POSIX functions like open, read, write, and stat.
- My interpreter calls C standard library functions directly.
- I handle memory management for strings within my runtime.
- I require a POSIX-compatible operating system.

## My Future Enhancements

When I add arrays:
- `dir_list` will return `array<string>`.
- `file_readlines` will return `array<string>`.
- `path_split` will return `array<string>`.

When I add error types:
- My functions will return `Result<T, Error>`.
- I will provide better error messages and handling.
