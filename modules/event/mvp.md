# event module MVP

<!--nl-snippet {"name":"module_event_mvp","check":false}-->
```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_free

fn main() -> int {
    let mut base: int = 0
    unsafe { set base (nl_event_base_new) }
    if (!= base 0) {
        unsafe { (nl_event_base_free base) }
    }
    return 0
}

shadow main { assert true }
```
