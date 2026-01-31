# uv module MVP

<!--nl-snippet {"name":"module_uv_mvp","check":false}-->
```nano
from "modules/uv/uv.nano" import nl_uv_version

fn main() -> int {
    let mut ver: int = 0
    unsafe { set ver (nl_uv_version) }
    assert (>= ver 0)
    return 0
}

shadow main { assert true }
```
