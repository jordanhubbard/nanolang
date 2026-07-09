# glew module MVP

<!--nl-snippet {"name":"module_glew_mvp","check":false}-->
```nano
from "modules/glew/glew.nano" import glewGetErrorString

fn main() -> int {
    let mut msg: string = ""
    unsafe { set msg (glewGetErrorString 0) }
    assert (>= (str_length msg) 0)
    return 0
}

shadow main { assert true }
```
