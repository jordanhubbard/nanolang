# ui_widgets module MVP

<!--nl-snippet {"name":"module_ui_widgets_mvp","check":false}-->
```nano
from "modules/ui_widgets/ui_widgets.nano" import nl_ui_set_scale

fn main() -> int {
    unsafe { (nl_ui_set_scale 1.0) }
    return 0
}

shadow main { assert true }
```
