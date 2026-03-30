# I/O Standard Library

File system and OS interaction functions.

> Auto-generated from source. Do not edit directly.

---

## Functions

- [`file_read(path: string) -> string`](#file_read)
- [`file_write(path: string, content: string) -> int`](#file_write)
- [`file_append(path: string, content: string) -> int`](#file_append)

---

### `file_read(path: string) -> string` { #file_read }

Reads the entire contents of a file and returns it as a string.

**Parameters:**

- `path` — Path to the file to read.

**Returns:** File contents as a string, or empty string on error.

**Example:**

```nano
let content: string = (file_read "data.txt")
(println content)
```

---

### `file_write(path: string, content: string) -> int` { #file_write }

Writes a string to a file, overwriting any existing content.

**Parameters:**

- `path` — Path to the file to write.
- `content` — String content to write.

**Returns:** 0 on success, -1 on error.

**Example:**

```nano
assert (== (file_write "output.txt" "Hello, World!\n") 0)
```

---

### `file_append(path: string, content: string) -> int` { #file_append }

Appends a string to the end of a file, creating it if it does not exist.

**Parameters:**

- `path` — Path to the file to append to.
- `content` — String content to append.

**Returns:** 0 on success, -1 on error.

**Example:**

```nano
(file_append "log.txt" "New log entry\n")
```

---
