# pt2_module module MVP

<!--nl-snippet {"name":"module_pt2_module_mvp","check":false}-->
```nano
from "modules/pt2_module/pt2_module.nano" import pt2_module_load, pt2_module_get_name

fn main() -> int {
    if false {
        let mut name: string = ""
        unsafe { (pt2_module_load "song.mod") }
        unsafe { set name (pt2_module_get_name) }
        assert (>= (str_length name) 0)
    }
    return 0
}

shadow main { assert true }
```
