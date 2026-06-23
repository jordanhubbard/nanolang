# pybridge_matplotlib module MVP

<!--nl-snippet {"name":"module_pybridge_matplotlib_mvp","check":false}-->
```nano
from "modules/pybridge_matplotlib/pybridge_matplotlib.nano" import mpl_init, mpl_shutdown

fn main() -> int {
    if false {
        unsafe { (mpl_init 0) }
        unsafe { (mpl_shutdown) }
    }
    return 0
}

shadow main { assert true }
```
