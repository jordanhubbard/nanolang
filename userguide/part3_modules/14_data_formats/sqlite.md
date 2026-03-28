# 14.2 SQLite - Embedded Database

**Embedded SQL database for persistent local storage.**

The `sqlite` module wraps SQLite3, giving NanoLang programs a full relational database without any external server. SQLite databases are single files on disk — or can live entirely in memory for testing. The module exposes both a low-level extern API and a clean set of public wrapper functions with result-code constants.

## Installation

Import from the module path:

```nano
from "modules/sqlite/sqlite.nano" import open, close, exec, exec_ok,
    prepare, step, finalize, reset,
    bind_int, bind_double, bind_text, bind_null,
    column_int, column_double, column_text, column_type, column_count, column_name,
    last_insert_rowid, changes, errmsg,
    begin_transaction, commit, rollback,
    has_row, is_done,
    SQLITE_OK, SQLITE_ROW, SQLITE_DONE
```

Import only the symbols you need. The module declares no public structs — database handles and statement handles are plain `int` values.

## Quick Start

```nano
from "modules/sqlite/sqlite.nano" import open, close, exec_ok, prepare, step, finalize,
    bind_text, bind_int, column_text, column_int, has_row

fn quick_demo() -> int {
    # Open an in-memory database
    let db: int = (open ":memory:")

    # Create a table
    (exec_ok db "CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    # Insert using a prepared statement
    let stmt: int = (prepare db "INSERT INTO people (name, age) VALUES (?, ?)")
    (bind_text stmt 1 "Alice")
    (bind_int stmt 2 30)
    (step stmt)
    (finalize stmt)

    # Query
    let q: int = (prepare db "SELECT name, age FROM people")
    let rc: int = (step q)
    let mut count: int = 0
    while (has_row rc) {
        let name: string = (column_text q 0)
        let age: int = (column_int q 1)
        (println (+ name (+ " is " (+ (int_to_string age) " years old"))))
        set count (+ count 1)
        set rc (step q)
    }
    (finalize q)
    (close db)
    return count
}

shadow quick_demo {
    assert (== (quick_demo) 1)
}
```

---

## API Reference

### Result Codes

The module exports these as zero-argument functions so you can use them as named constants:

| Function | Value | Meaning |
|----------|-------|---------|
| `SQLITE_OK()` | `0` | Operation succeeded |
| `SQLITE_ROW()` | `100` | `step` returned a row |
| `SQLITE_DONE()` | `101` | `step` finished, no more rows |
| `SQLITE_INTEGER()` | `1` | Column type: integer |
| `SQLITE_FLOAT()` | `2` | Column type: float |
| `SQLITE_TEXT()` | `3` | Column type: text |
| `SQLITE_BLOB()` | `4` | Column type: blob |
| `SQLITE_NULL()` | `5` | Column type: null |

Use `has_row` and `is_done` helpers instead of comparing against raw numbers:

```nano
let rc: int = (step stmt)
while (has_row rc) {
    # process row
    set rc (step stmt)
}
```

---

### Database Management

#### `fn open(filename: string) -> int`

Open a database file. Returns a non-zero database handle on success. Use `":memory:"` for a transient in-memory database.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | `string` | File path or `":memory:"` |

**Returns:** `int` — database handle (non-zero = success, `0` = failure).

```nano
let db: int = (open "myapp.db")
if (== db 0) {
    (println "Failed to open database")
}
```

#### `fn close(db: int) -> int`

Close the database and release all resources. Returns `0` on success.

```nano
let rc: int = (close db)
```

#### `fn errmsg(db: int) -> string`

Return the human-readable error message for the most recent failure on this database handle. Useful for diagnosing errors after a failed `exec` or `step`.

```nano
let msg: string = (errmsg db)
(println (+ "SQLite error: " msg))
```

#### `fn version() -> string`

Return the SQLite library version string (e.g. `"3.41.2"`).

#### `fn version_number() -> int`

Return the SQLite version as an integer (e.g. `3041002`).

---

### Simple Execution

#### `fn exec(db: int, sql: string) -> int`

Execute a SQL statement that produces no result rows (DDL, INSERT, UPDATE, DELETE without row return). Returns `0` (`SQLITE_OK`) on success, non-zero on error.

```nano
let rc: int = (exec db "CREATE TABLE log (msg TEXT, ts INTEGER)")
if (!= rc 0) {
    (println (+ "Error: " (errmsg db)))
}
```

#### `fn exec_ok(db: int, sql: string) -> bool`

Convenience wrapper around `exec`. Returns `true` if the statement succeeded.

```nano
assert (exec_ok db "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, name TEXT)")
```

---

### Prepared Statements

Prepared statements are the safe way to pass data into SQL. They use `?` placeholders that are bound to values before execution.

#### `fn prepare(db: int, sql: string) -> int`

Compile a SQL statement into a prepared statement handle. Returns a non-zero handle on success, `0` on failure.

```nano
let stmt: int = (prepare db "SELECT id, name FROM users WHERE age > ?")
if (== stmt 0) {
    (println "Failed to prepare statement")
}
```

#### `fn step(stmt: int) -> int`

Execute one step of a prepared statement. For SELECT queries, each call to `step` advances to the next row. Returns `SQLITE_ROW` (100) when a row is available and `SQLITE_DONE` (101) when there are no more rows. Use `has_row` and `is_done` to interpret the result.

```nano
let rc: int = (step stmt)
while (has_row rc) {
    # read columns
    set rc (step stmt)
}
```

#### `fn finalize(stmt: int) -> int`

Destroy a prepared statement and release its resources. Always call `finalize` when done with a statement, including after errors.

```nano
(finalize stmt)
```

#### `fn reset(stmt: int) -> int`

Reset a prepared statement so it can be re-executed with new bindings. Does not clear the bindings — call `bind_*` again to update them.

```nano
(reset stmt)
(bind_int stmt 1 42)
(step stmt)
```

#### `fn has_row(step_result: int) -> bool`

Returns `true` if the result of `step` indicates a row is available (`SQLITE_ROW`).

#### `fn is_done(step_result: int) -> bool`

Returns `true` if the result of `step` indicates no more rows (`SQLITE_DONE`).

---

### Parameter Binding (1-based indices)

Bind values to `?` placeholders in a prepared statement. Indices start at **1** (the first `?` is index 1).

#### `fn bind_int(stmt: int, index: int, value: int) -> int`

Bind an integer value. Returns `0` on success.

#### `fn bind_double(stmt: int, index: int, value: float) -> int`

Bind a floating-point value. Returns `0` on success.

#### `fn bind_text(stmt: int, index: int, value: string) -> int`

Bind a string value. Returns `0` on success.

#### `fn bind_null(stmt: int, index: int) -> int`

Bind a SQL NULL value. Returns `0` on success.

```nano
let stmt: int = (prepare db "INSERT INTO readings (sensor, value, ts) VALUES (?, ?, ?)")
(bind_text stmt 1 "temperature")
(bind_double stmt 2 23.5)
(bind_int stmt 3 1711584000)
(step stmt)
(finalize stmt)
```

---

### Column Reading (0-based indices)

After a successful `step` that returned `SQLITE_ROW`, read column values using their 0-based index.

#### `fn column_int(stmt: int, index: int) -> int`

Read an integer column.

#### `fn column_double(stmt: int, index: int) -> float`

Read a floating-point column.

#### `fn column_text(stmt: int, index: int) -> string`

Read a text column.

#### `fn column_type(stmt: int, index: int) -> int`

Return the storage type of the column in the current row. Compare against `SQLITE_INTEGER()`, `SQLITE_FLOAT()`, `SQLITE_TEXT()`, `SQLITE_NULL()`, or `SQLITE_BLOB()`.

#### `fn column_count(stmt: int) -> int`

Return the number of columns in the result set.

#### `fn column_name(stmt: int, index: int) -> string`

Return the name of column `index` (as declared in the SQL).

```nano
let stmt: int = (prepare db "SELECT id, name, score FROM results")
let rc: int = (step stmt)
while (has_row rc) {
    let id: int = (column_int stmt 0)
    let name: string = (column_text stmt 1)
    let score: float = (column_double stmt 2)
    (println (+ name (+ ": " (int_to_string id))))
    set rc (step stmt)
}
(finalize stmt)
```

---

### Database Information

#### `fn last_insert_rowid(db: int) -> int`

Return the row ID of the last successful INSERT on this connection. Useful when the table has an `INTEGER PRIMARY KEY AUTOINCREMENT` column.

```nano
(exec_ok db "INSERT INTO items (name) VALUES ('widget')")
let new_id: int = (last_insert_rowid db)
```

#### `fn changes(db: int) -> int`

Return the number of rows modified by the most recent INSERT, UPDATE, or DELETE statement.

```nano
(exec_ok db "UPDATE users SET active = 0 WHERE last_login < 1000000")
let deactivated: int = (changes db)
(println (+ (int_to_string deactivated) " users deactivated"))
```

---

### Transactions

SQLite wraps each statement in an implicit transaction by default. For bulk operations, wrapping them in an explicit transaction is dramatically faster.

#### `fn begin_transaction(db: int) -> int`

Begin an explicit transaction. Returns `0` on success.

#### `fn commit(db: int) -> int`

Commit the current transaction. Returns `0` on success.

#### `fn rollback(db: int) -> int`

Roll back the current transaction, undoing all changes since `begin_transaction`. Returns `0` on success.

```nano
(begin_transaction db)
let ok: bool = true
for i in (range 0 1000) {
    let rc: int = (exec db (+ "INSERT INTO log (msg) VALUES ('item " (+ (int_to_string i) "')")))
    if (!= rc 0) {
        set ok false
    } else {
        (print "")
    }
}
if ok {
    (commit db)
} else {
    (rollback db)
}
```

---

## Examples

### Example 1: Simple CRUD Operations

```nano
from "modules/sqlite/sqlite.nano" import open, close, exec_ok, prepare, step, finalize,
    bind_int, bind_text, column_int, column_text, has_row, last_insert_rowid

fn setup_db(path: string) -> int {
    let db: int = (open path)
    (exec_ok db "CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, price INTEGER, stock INTEGER DEFAULT 0)")
    return db
}

fn add_product(db: int, name: string, price: int) -> int {
    let stmt: int = (prepare db "INSERT INTO products (name, price) VALUES (?, ?)")
    (bind_text stmt 1 name)
    (bind_int stmt 2 price)
    (step stmt)
    (finalize stmt)
    return (last_insert_rowid db)
}

fn find_by_name(db: int, name: string) -> int {
    let stmt: int = (prepare db "SELECT id, price FROM products WHERE name = ? LIMIT 1")
    (bind_text stmt 1 name)
    let rc: int = (step stmt)
    let found_id: int = 0
    if (has_row rc) {
        set found_id (column_int stmt 0)
    } else {
        (print "")
    }
    (finalize stmt)
    return found_id
}

shadow setup_db {
    let db: int = (setup_db ":memory:")
    let id: int = (add_product db "Widget" 999)
    assert (> id 0)
    let found: int = (find_by_name db "Widget")
    assert (== found id)
    (close db)
}
```

### Example 2: Iterating Over Query Results

```nano
from "modules/sqlite/sqlite.nano" import open, close, exec_ok, prepare, step, finalize,
    bind_int, column_int, column_text, has_row

fn get_users_older_than(db: int, min_age: int) -> array<string> {
    let stmt: int = (prepare db "SELECT name FROM users WHERE age > ? ORDER BY name")
    (bind_int stmt 1 min_age)
    let mut names: array<string> = []
    let mut rc: int = (step stmt)
    while (has_row rc) {
        let name: string = (column_text stmt 0)
        set names (array_push names name)
        set rc (step stmt)
    }
    (finalize stmt)
    return names
}

shadow get_users_older_than {
    let db: int = (open ":memory:")
    (exec_ok db "CREATE TABLE users (name TEXT, age INTEGER)")
    (exec_ok db "INSERT INTO users VALUES ('Alice', 35)")
    (exec_ok db "INSERT INTO users VALUES ('Bob', 20)")
    (exec_ok db "INSERT INTO users VALUES ('Carol', 40)")
    let results: array<string> = (get_users_older_than db 30)
    assert (== (array_length results) 2)
    (close db)
}
```

### Example 3: Using Transactions for Bulk Inserts

```nano
from "modules/sqlite/sqlite.nano" import open, close, exec_ok, prepare, step, finalize, reset,
    bind_text, begin_transaction, commit, rollback

fn bulk_insert_tags(db: int, tags: array<string>) -> bool {
    (exec_ok db "CREATE TABLE IF NOT EXISTS tags (name TEXT UNIQUE)")
    let rc: int = (begin_transaction db)
    if (!= rc 0) {
        return false
    } else {
        (print "")
    }

    let stmt: int = (prepare db "INSERT OR IGNORE INTO tags (name) VALUES (?)")
    let mut failed: bool = false
    for i in (range 0 (array_length tags)) {
        let tag: string = (at tags i)
        (reset stmt)
        (bind_text stmt 1 tag)
        (step stmt)
    }
    (finalize stmt)

    if failed {
        (rollback db)
        return false
    } else {
        (commit db)
        return true
    }
}

shadow bulk_insert_tags {
    let db: int = (open ":memory:")
    let tags: array<string> = ["nano", "lang", "sqlite", "fast"]
    assert (bulk_insert_tags db tags)
    (close db)
}
```

### Example 4: Dynamic Columns with column_count and column_name

```nano
from "modules/sqlite/sqlite.nano" import open, close, exec_ok, prepare, step, finalize,
    column_count, column_name, column_text, has_row

fn describe_table(db: int, table: string) -> array<string> {
    let sql: string = (+ "SELECT * FROM " (+ table " LIMIT 0"))
    let stmt: int = (prepare db sql)
    (step stmt)
    let n: int = (column_count stmt)
    let mut col_names: array<string> = []
    for i in (range 0 n) {
        set col_names (array_push col_names (column_name stmt i))
    }
    (finalize stmt)
    return col_names
}

shadow describe_table {
    let db: int = (open ":memory:")
    (exec_ok db "CREATE TABLE events (id INTEGER, name TEXT, ts INTEGER)")
    let cols: array<string> = (describe_table db "events")
    assert (== (array_length cols) 3)
    assert (== (at cols 0) "id")
    assert (== (at cols 1) "name")
    (close db)
}
```

### Example 5: Complete User Database

A realistic application module:

```nano
from "modules/sqlite/sqlite.nano" import open, close, exec_ok, prepare, step, finalize, reset,
    bind_text, bind_int, column_int, column_text, has_row, last_insert_rowid, errmsg

struct User {
    id: int,
    username: string,
    email: string
}

fn db_open(path: string) -> int {
    let db: int = (open path)
    (exec_ok db "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT)")
    (exec_ok db "CREATE INDEX IF NOT EXISTS idx_username ON users (username)")
    return db
}

fn db_add_user(db: int, username: string, email: string) -> int {
    let stmt: int = (prepare db "INSERT INTO users (username, email) VALUES (?, ?)")
    (bind_text stmt 1 username)
    (bind_text stmt 2 email)
    (step stmt)
    (finalize stmt)
    return (last_insert_rowid db)
}

fn db_get_user(db: int, username: string) -> User {
    let stmt: int = (prepare db "SELECT id, username, email FROM users WHERE username = ? LIMIT 1")
    (bind_text stmt 1 username)
    let rc: int = (step stmt)
    if (has_row rc) {
        let u: User = User {
            id:       (column_int stmt 0),
            username: (column_text stmt 1),
            email:    (column_text stmt 2)
        }
        (finalize stmt)
        return u
    } else {
        (finalize stmt)
        return User { id: 0, username: "", email: "" }
    }
}

shadow db_open {
    let db: int = (db_open ":memory:")
    let id: int = (db_add_user db "alice" "alice@example.com")
    assert (> id 0)
    let u: User = (db_get_user db "alice")
    assert (== u.id id)
    assert (== u.email "alice@example.com")
    (close db)
}
```

---

## Best Practices

- **Always use prepared statements** for queries with user-supplied data. String concatenation into SQL is vulnerable to SQL injection.
- **Always call `finalize`** when done with a statement, including when an error occurs mid-way. A non-finalized statement holds a read lock on the database.
- **Use transactions for bulk writes.** SQLite without explicit transactions commits after every statement. A loop of 10,000 inserts without a transaction can be 100x slower than the same inserts wrapped in a single transaction.
- **Check return codes.** `exec` and `step` return non-zero on failure. Call `errmsg` to get a descriptive message.
- **Use `":memory:"` in shadow tests** so tests don't create files on disk and run in isolation.
- **Open the database once** and pass the handle around, rather than opening and closing for each operation.
- **Call `reset` before re-using a statement** rather than calling `finalize` and `prepare` again — it is more efficient.

---

**Previous:** [14.1 JSON - Parsing & Generation](json.html)
**Next:** [Chapter 15: Web & Networking](../15_web_networking/index.html)
