# sdl_image module MVP

<!--nl-snippet {"name":"module_sdl_image_mvp","check":false}-->
```nano
import "modules/sdl/sdl.nano"
from "modules/sdl_image/sdl_image.nano" import IMG_Init, IMG_Quit

fn main() -> int {
    unsafe { (IMG_Init 0) }
    unsafe { (IMG_Quit) }
    return 0
}

shadow main { assert true }
```
