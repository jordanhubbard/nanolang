# sdl_ttf module MVP

<!--nl-snippet {"name":"module_sdl_ttf_mvp","check":false}-->
```nano
from "modules/sdl_ttf/sdl_ttf.nano" import TTF_Init, TTF_Quit

fn main() -> int {
    unsafe { (TTF_Init) }
    unsafe { (TTF_Quit) }
    return 0
}

shadow main { assert true }
```
