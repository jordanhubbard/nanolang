# sdl_mixer module MVP

<!--nl-snippet {"name":"module_sdl_mixer_mvp","check":false}-->
```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_Init, Mix_Quit

fn main() -> int {
    unsafe { (Mix_Init 0) }
    unsafe { (Mix_Quit) }
    return 0
}

shadow main { assert true }
```
