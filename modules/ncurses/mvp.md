# ncurses module MVP

<!--nl-snippet {"name":"module_ncurses_mvp","check":false}-->
```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, endwin_wrapper

fn main() -> int {
    if false {
        unsafe { (initscr_wrapper) }
        unsafe { (endwin_wrapper) }
    }
    return 0
}

shadow main { assert true }
```
