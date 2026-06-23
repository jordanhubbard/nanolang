# bullet module MVP

<!--nl-snippet {"name":"module_bullet_mvp","check":false}-->
```nano
from "modules/bullet/bullet.nano" import nl_bullet_init, nl_bullet_set_gravity, nl_bullet_step, nl_bullet_cleanup

fn main() -> int {
    let mut code: int = 0
    unsafe { set code (nl_bullet_init) }
    assert (>= code -1)
    unsafe { (nl_bullet_set_gravity 0.0 -9.8 0.0) }
    unsafe { (nl_bullet_step 0.016) }
    unsafe { (nl_bullet_cleanup) }
    return 0
}

shadow main { assert true }
```
