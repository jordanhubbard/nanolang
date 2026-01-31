# sdl_helpers module MVP

<!--nl-snippet {"name":"module_sdl_helpers_mvp","check":false}-->
```nano
from "modules/sdl_helpers/sdl_helpers.nano" import nl_sdl_poll_event_quit

fn main() -> int {
    if false {
        let mut quit: int = 0
        unsafe { set quit (nl_sdl_poll_event_quit) }
        assert (>= quit 0)
    }
    return 0
}

shadow main { assert true }
```
