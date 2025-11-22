# SQLite3 Module for nanolang

Embedded SQL database engine for local data storage and querying.

## Installation

**macOS:**
```bash
brew install sqlite
```

**Ubuntu/Debian:**
```bash
sudo apt install libsqlite3-dev
```

## Usage

```nano
import "modules/sqlite/sqlite.nano"

fn main() -> int {
    # Open database (creates if doesn't exist)
    let db: int = (nl_sqlite3_open "mydata.db")
    
    # Create table
    (nl_sqlite3_exec db "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    
    # Insert data with prepared statement
    let stmt: int = (nl_sqlite3_prepare db "INSERT INTO users (name) VALUES (?)")
    (nl_sqlite3_bind_text stmt 1 "Alice")
    (nl_sqlite3_step stmt)
    (nl_sqlite3_finalize stmt)
    
    # Query data
    set stmt (nl_sqlite3_prepare db "SELECT id, name FROM users")
    let mut result: int = (nl_sqlite3_step stmt)
    while (== result 100) {
        let id: int = (nl_sqlite3_column_int stmt 0)
        let name: string = (nl_sqlite3_column_text stmt 1)
        (print id)
        (print ": ")
        (println name)
        set result (nl_sqlite3_step stmt)
    }
    (nl_sqlite3_finalize stmt)
    
    # Close database
    (nl_sqlite3_close db)
    return 0
}

shadow main {
    # Skip - uses extern functions
}
```

## Features

- **Full SQL support**: CREATE, INSERT, UPDATE, DELETE, SELECT
- **Prepared statements**: Parameterized queries for security
- **Transactions**: BEGIN, COMMIT, ROLLBACK
- **Type safety**: Typed binding and column access
- **Zero configuration**: No server setup required

## Example

See `examples/sqlite_example.nano` for comprehensive usage examples.

## API Reference

### Database Management
- `nl_sqlite3_open(filename: string) -> int` - Open/create database
- `nl_sqlite3_close(db: int) -> int` - Close database
- `nl_sqlite3_exec(db: int, sql: string) -> int` - Execute SQL directly

### Prepared Statements
- `nl_sqlite3_prepare(db: int, sql: string) -> int` - Prepare statement
- `nl_sqlite3_finalize(stmt: int) -> int` - Free statement
- `nl_sqlite3_step(stmt: int) -> int` - Execute/fetch next row (returns 100 for row, 101 for done)
- `nl_sqlite3_reset(stmt: int) -> int` - Reset for re-execution

### Binding Parameters (1-based index)
- `nl_sqlite3_bind_int(stmt: int, index: int, value: int) -> int`
- `nl_sqlite3_bind_double(stmt: int, index: int, value: float) -> int`
- `nl_sqlite3_bind_text(stmt: int, index: int, value: string) -> int`
- `nl_sqlite3_bind_null(stmt: int, index: int) -> int`

### Reading Columns (0-based index)
- `nl_sqlite3_column_count(stmt: int) -> int` - Number of columns
- `nl_sqlite3_column_int(stmt: int, index: int) -> int` - Get integer
- `nl_sqlite3_column_double(stmt: int, index: int) -> float` - Get float
- `nl_sqlite3_column_text(stmt: int, index: int) -> string` - Get string
- `nl_sqlite3_column_name(stmt: int, index: int) -> string` - Get column name

### Transactions
- `nl_sqlite3_begin_transaction(db: int) -> int`
- `nl_sqlite3_commit(db: int) -> int`
- `nl_sqlite3_rollback(db: int) -> int`

## Notes

**Step Return Values:**
- `100` = SQLITE_ROW (data available)
- `101` = SQLITE_DONE (no more rows)
- Other values = error codes

**Index Conventions:**
- Binding parameters: 1-based (first parameter is 1)
- Reading columns: 0-based (first column is 0)
