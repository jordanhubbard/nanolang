# glut module MVP

<!--nl-snippet {"name":"module_glut_mvp","check":false}-->
```nano
from "modules/glut/glut.nano" import glutInit, glutCreateWindow

fn main() -> int {
    if false {
        unsafe { (glutInit 0 0) }
        unsafe { (glutCreateWindow "demo") }
    }
    return 0
}

shadow main { assert true }
```
