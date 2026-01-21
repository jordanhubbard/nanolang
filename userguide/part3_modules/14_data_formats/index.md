# Chapter 14: Data Formats

**JSON parsing and SQLite database operations.**

This chapter covers working with structured data formats: JSON for serialization and SQLite for local database storage.

## 14.1 JSON (JavaScript Object Notation)

The `modules/std/json/json.nano` module provides JSON parsing and generation.

### Parsing JSON

```nano
from "modules/std/json/json.nano" import parse, free, Json

fn parse_json_string() -> Json {
    let json_text: string = "{\"name\": \"Alice\", \"age\": 30}"
    let obj: Json = (parse json_text)
    return obj
}

shadow parse_json_string {
    let obj: Json = (parse_json_string)
    assert (!= obj 0)
    (free obj)
}
```

**Important:** Always call `free(json)` when done to prevent memory leaks.

### Type Checking

```nano
from "modules/std/json/json.nano" import parse, is_object, is_array
from "modules/std/json/json.nano" import is_string, is_number, is_bool, is_null, free

fn check_json_types(text: string) -> bool {
    let json: Json = (parse text)
    
    let result: bool = (cond
        ((is_object json) true)
        ((is_array json) true)
        ((is_string json) true)
        ((is_number json) true)
        ((is_bool json) true)
        ((is_null json) true)
        (else false)
    )
    
    (free json)
    return result
}

shadow check_json_types {
    assert (check_json_types "{}")
    assert (check_json_types "[]")
    assert (check_json_types "\"text\"")
    assert (check_json_types "42")
    assert (check_json_types "true")
    assert (check_json_types "null")
}
```

### Extracting Values

```nano
from "modules/std/json/json.nano" import parse, get, as_string, as_int, as_bool, free

fn extract_user_data(json_text: string) -> bool {
    let obj: Json = (parse json_text)
    
    # Get nested JSON values
    let name_json: Json = (get obj "name")
    let age_json: Json = (get obj "age")
    let active_json: Json = (get obj "active")
    
    # Convert to NanoLang types
    let name: string = (as_string name_json)
    let age: int = (as_int age_json)
    let active: bool = (as_bool active_json)
    
    # Clean up
    (free name_json)
    (free age_json)
    (free active_json)
    (free obj)
    
    return (and (== name "Alice") (== age 30))
}

shadow extract_user_data {
    assert (extract_user_data "{\"name\": \"Alice\", \"age\": 30, \"active\": true}")
}
```

### Convenience Functions

```nano
from "modules/std/json/json.nano" import parse, get_string, get_int, free

fn get_values_easily(json_text: string) -> string {
    let obj: Json = (parse json_text)
    
    # Convenience functions handle type conversion
    let name: string = (get_string obj "name")
    let age: int = (get_int obj "age")
    
    (free obj)
    return name
}

shadow get_values_easily {
    let name: string = (get_values_easily "{\"name\": \"Bob\", \"age\": 25}")
    assert (== name "Bob")
}
```

**Available convenience functions:**
- `get_string(obj, key)` - Returns string or `""`
- `get_int(obj, key)` - Returns int or `0`

### Checking Keys

```nano
from "modules/std/json/json.nano" import parse, object_has, get_string, free

fn safe_get(json_text: string, key: string) -> string {
    let obj: Json = (parse json_text)
    
    if (object_has obj key) {
        let value: string = (get_string obj key)
        (free obj)
        return value
    }
    
    (free obj)
    return ""
}

shadow safe_get {
    assert (== (safe_get "{\"name\": \"Alice\"}" "name") "Alice")
    assert (== (safe_get "{\"name\": \"Alice\"}" "missing") "")
}
```

### Iterating Over Objects

```nano
from "modules/std/json/json.nano" import parse, keys, get_string, free

fn print_all_keys(json_text: string) -> int {
    let obj: Json = (parse json_text)
    let key_array: array<string> = (keys obj)
    let count: int = (array_length key_array)
    
    for i in (range 0 count) {
        let key: string = (at key_array i)
        let value: string = (get_string obj key)
        (println (+ key (+ ": " value)))
    }
    
    (free obj)
    return count
}

shadow print_all_keys {
    let count: int = (print_all_keys "{\"a\": \"1\", \"b\": \"2\"}")
    assert (== count 2)
}
```

### Working with Arrays

```nano
from "modules/std/json/json.nano" import parse, array_size, get_index, as_int, free

fn sum_json_array(json_text: string) -> int {
    let arr: Json = (parse json_text)
    let len: int = (array_size arr)
    let mut sum: int = 0
    
    for i in (range 0 len) {
        let item: Json = (get_index arr i)
        set sum (+ sum (as_int item))
        (free item)
    }
    
    (free arr)
    return sum
}

shadow sum_json_array {
    assert (== (sum_json_array "[1, 2, 3, 4, 5]") 15)
}
```

### Creating JSON

```nano
from "modules/std/json/json.nano" import new_object, new_string, new_int
from "modules/std/json/json.nano" import object_set, stringify, free

fn create_json_object() -> string {
    let obj: Json = (new_object)
    
    # Add fields
    (object_set obj "name" (new_string "Alice"))
    (object_set obj "age" (new_int 30))
    
    # Convert to string
    let json_str: string = (stringify obj)
    
    (free obj)
    return json_str
}

shadow create_json_object {
    let json: string = (create_json_object)
    assert (str_contains json "Alice")
}
```

### Creating Arrays

```nano
from "modules/std/json/json.nano" import new_array, new_int
from "modules/std/json/json.nano" import json_array_push, stringify, free

fn create_json_array() -> string {
    let arr: Json = (new_array)
    
    # Add elements
    (json_array_push arr (new_int 1))
    (json_array_push arr (new_int 2))
    (json_array_push arr (new_int 3))
    
    # Convert to string
    let json_str: string = (stringify arr)
    
    (free arr)
    return json_str
}

shadow create_json_array {
    let json: string = (create_json_array)
    assert (str_contains json "1")
}
```

### Complete Example: Configuration File

```nano
from "modules/std/json/json.nano" import parse, get_string, get_int, object_has, free
from "modules/std/fs.nano" import file_read, file_write

struct Config {
    app_name: string,
    version: string,
    port: int,
    debug: bool
}

fn load_config(path: string) -> Config {
    let json_text: string = (file_read path)
    let obj: Json = (parse json_text)
    
    let config: Config = Config {
        app_name: (get_string obj "app_name"),
        version: (get_string obj "version"),
        port: (get_int obj "port"),
        debug: (object_has obj "debug")
    }
    
    (free obj)
    return config
}

shadow load_config {
    # Would test with actual file
    assert true
}
```

## 14.2 SQLite Database

The `modules/sqlite/sqlite.nano` module provides embedded SQL database functionality.

### Opening a Database

```nano
extern fn nl_sqlite3_open(_filename: string) -> int
extern fn nl_sqlite3_close(_db: int) -> int

fn open_database(path: string) -> int {
    unsafe {
        let db: int = (nl_sqlite3_open path)
        return db
    }
}

shadow open_database {
    let db: int = (open_database ":memory:")
    assert (!= db 0)
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

**Database paths:**
- `":memory:"` - In-memory database (temporary)
- `"data.db"` - File on disk (persistent)

### Executing SQL

```nano
extern fn nl_sqlite3_exec(_db: int, _sql: string) -> int

fn create_table(db: int) -> bool {
    let sql: string = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
    
    unsafe {
        let result: int = (nl_sqlite3_exec db sql)
        return (== result 0)
    }
}

shadow create_table {
    let db: int = (open_database ":memory:")
    assert (create_table db)
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

### Inserting Data

```nano
fn insert_user(db: int, name: string, age: int) -> bool {
    let sql: string = (+ "INSERT INTO users (name, age) VALUES ('" 
                        (+ name (+ "', " 
                        (+ (int_to_string age) ")"))))
    
    unsafe {
        let result: int = (nl_sqlite3_exec db sql)
        return (== result 0)
    }
}

shadow insert_user {
    let db: int = (open_database ":memory:")
    (create_table db)
    assert (insert_user db "Alice" 30)
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

⚠️ **Warning:** The above uses string concatenation. For production, use prepared statements to prevent SQL injection.

### Prepared Statements (Safe)

```nano
extern fn nl_sqlite3_prepare(_db: int, _sql: string) -> int
extern fn nl_sqlite3_bind_text(_stmt: int, _index: int, _value: string) -> int
extern fn nl_sqlite3_bind_int(_stmt: int, _index: int, _value: int) -> int
extern fn nl_sqlite3_step(_stmt: int) -> int
extern fn nl_sqlite3_finalize(_stmt: int) -> int

fn insert_user_safe(db: int, name: string, age: int) -> bool {
    let sql: string = "INSERT INTO users (name, age) VALUES (?, ?)"
    
    unsafe {
        let stmt: int = (nl_sqlite3_prepare db sql)
        if (== stmt 0) {
            return false
        }
        
        (nl_sqlite3_bind_text stmt 1 name)
        (nl_sqlite3_bind_int stmt 2 age)
        
        let result: int = (nl_sqlite3_step stmt)
        (nl_sqlite3_finalize stmt)
        
        return (!= result 0)
    }
}

shadow insert_user_safe {
    let db: int = (open_database ":memory:")
    (create_table db)
    assert (insert_user_safe db "Bob" 25)
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

**Parameter binding:**
- Indices are **1-based** (first `?` is index 1)
- `bind_text` - Bind string
- `bind_int` - Bind integer
- `bind_double` - Bind float
- `bind_null` - Bind NULL

### Querying Data

```nano
extern fn nl_sqlite3_column_count(_stmt: int) -> int
extern fn nl_sqlite3_column_text(_stmt: int, _index: int) -> string
extern fn nl_sqlite3_column_int(_stmt: int, _index: int) -> int

fn count_users(db: int) -> int {
    let sql: string = "SELECT COUNT(*) FROM users"
    
    unsafe {
        let stmt: int = (nl_sqlite3_prepare db sql)
        if (== stmt 0) {
            return 0
        }
        
        (nl_sqlite3_step stmt)
        let count: int = (nl_sqlite3_column_int stmt 0)
        (nl_sqlite3_finalize stmt)
        
        return count
    }
}

shadow count_users {
    let db: int = (open_database ":memory:")
    (create_table db)
    (insert_user_safe db "Alice" 30)
    (insert_user_safe db "Bob" 25)
    
    assert (== (count_users db) 2)
    
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

**Column access:**
- Indices are **0-based** (first column is index 0)
- `column_text` - Get string
- `column_int` - Get integer
- `column_double` - Get float

### Iterating Over Results

```nano
fn list_users(db: int) -> array<string> {
    let sql: string = "SELECT name, age FROM users"
    let mut names: array<string> = (array_new 10 "")
    let mut count: int = 0
    
    unsafe {
        let stmt: int = (nl_sqlite3_prepare db sql)
        if (== stmt 0) {
            return names
        }
        
        # Step through results
        while (!= (nl_sqlite3_step stmt) 0) {
            let name: string = (nl_sqlite3_column_text stmt 0)
            let age: int = (nl_sqlite3_column_int stmt 1)
            
            if (< count 10) {
                (array_set names count name)
                set count (+ count 1)
            }
        }
        
        (nl_sqlite3_finalize stmt)
    }
    
    return names
}

shadow list_users {
    let db: int = (open_database ":memory:")
    (create_table db)
    (insert_user_safe db "Alice" 30)
    (insert_user_safe db "Bob" 25)
    
    let names: array<string> = (list_users db)
    assert (> (array_length names) 0)
    
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

### Transactions

```nano
extern fn nl_sqlite3_begin_transaction(_db: int) -> int
extern fn nl_sqlite3_commit(_db: int) -> int
extern fn nl_sqlite3_rollback(_db: int) -> int

fn batch_insert(db: int, names: array<string>) -> bool {
    unsafe {
        # Start transaction
        (nl_sqlite3_begin_transaction db)
        
        let len: int = (array_length names)
        for i in (range 0 len) {
            let name: string = (at names i)
            if (not (insert_user_safe db name 0)) {
                # Error - rollback
                (nl_sqlite3_rollback db)
                return false
            }
        }
        
        # Success - commit
        (nl_sqlite3_commit db)
        return true
    }
}

shadow batch_insert {
    let db: int = (open_database ":memory:")
    (create_table db)
    
    let names: array<string> = ["Alice", "Bob", "Carol"]
    assert (batch_insert db names)
    
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

### Error Handling

```nano
extern fn nl_sqlite3_errmsg(_db: int) -> string

fn insert_with_error_check(db: int, name: string) -> bool {
    unsafe {
        if (not (insert_user_safe db name 25)) {
            let error: string = (nl_sqlite3_errmsg db)
            (println (+ "Error: " error))
            return false
        }
        return true
    }
}

shadow insert_with_error_check {
    let db: int = (open_database ":memory:")
    (create_table db)
    assert (insert_with_error_check db "Alice")
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

### Complete Example: User Database

```nano
struct User {
    id: int,
    name: string,
    email: string,
    age: int
}

fn setup_users_db(db: int) -> bool {
    let sql: string = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT, age INTEGER)"
    
    unsafe {
        return (== (nl_sqlite3_exec db sql) 0)
    }
}

fn add_user(db: int, name: string, email: string, age: int) -> int {
    let sql: string = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)"
    
    unsafe {
        let stmt: int = (nl_sqlite3_prepare db sql)
        if (== stmt 0) {
            return 0
        }
        
        (nl_sqlite3_bind_text stmt 1 name)
        (nl_sqlite3_bind_text stmt 2 email)
        (nl_sqlite3_bind_int stmt 3 age)
        
        (nl_sqlite3_step stmt)
        (nl_sqlite3_finalize stmt)
        
        # Get last inserted row ID
        return (nl_sqlite3_last_insert_rowid db)
    }
}

fn find_user_by_email(db: int, email: string) -> int {
    let sql: string = "SELECT id, name, age FROM users WHERE email = ?"
    
    unsafe {
        let stmt: int = (nl_sqlite3_prepare db sql)
        if (== stmt 0) {
            return 0
        }
        
        (nl_sqlite3_bind_text stmt 1 email)
        
        if (!= (nl_sqlite3_step stmt) 0) {
            let user_id: int = (nl_sqlite3_column_int stmt 0)
            (nl_sqlite3_finalize stmt)
            return user_id
        }
        
        (nl_sqlite3_finalize stmt)
        return 0
    }
}

shadow find_user_by_email {
    let db: int = (open_database ":memory:")
    (setup_users_db db)
    
    let user_id: int = (add_user db "Alice" "alice@example.com" 30)
    assert (> user_id 0)
    
    let found_id: int = (find_user_by_email db "alice@example.com")
    assert (== found_id user_id)
    
    unsafe {
        (nl_sqlite3_close db)
    }
}
```

## Summary

In this chapter, you learned:
- ✅ JSON parsing: `parse`, type checking, value extraction
- ✅ JSON creation: `new_object`, `new_array`, `stringify`
- ✅ SQLite basics: open, close, exec
- ✅ Prepared statements: bind parameters, prevent injection
- ✅ Querying: iterate results, extract columns
- ✅ Transactions: begin, commit, rollback
- ✅ Error handling: `errmsg`

### Quick Reference

| Operation | JSON | SQLite |
|-----------|------|--------|
| **Open/Create** | `parse(text)` | `nl_sqlite3_open(path)` |
| **Close/Free** | `free(json)` | `nl_sqlite3_close(db)` |
| **Execute** | - | `nl_sqlite3_exec(db, sql)` |
| **Prepare** | - | `nl_sqlite3_prepare(db, sql)` |
| **Read value** | `get(obj, key)`, `as_int` | `column_int(stmt, idx)` |
| **Write value** | `object_set(obj, key, val)` | `bind_int(stmt, idx, val)` |
| **Iterate** | `keys`, `array_size` | `step(stmt)` |
| **Convert** | `stringify(json)` | `column_text(stmt, idx)` |

---

**Previous:** [Chapter 13: Text Processing](../13_text_processing/index.md)  
**Next:** [Chapter 15: Web & Networking](../15_web_networking/index.md)
