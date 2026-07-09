# math_ext module MVP

<!--nl-snippet {"name":"module_math_ext_mvp","check":false}-->
```nano
from "modules/math_ext/math_ext.nano" import asin

fn main() -> int {
    let mut v: float = 0.0
    unsafe { set v (asin 0.0) }
    assert (== v 0.0)
    return 0
}

shadow main { assert true }
```
