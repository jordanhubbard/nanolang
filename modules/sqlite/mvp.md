# sqlite module MVP

<!--nl-snippet {"name":"module_sqlite_mvp","check":false}-->
```nano
from "modules/sqlite/sqlite.nano" import nl_sqlite3_version

fn main() -> int {
    let mut version: string = ""
    unsafe { set version (nl_sqlite3_version) }
    assert (> (str_length version) 0)
    return 0
}

shadow main { assert true }
```
