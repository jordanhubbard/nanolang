# nano_highlight module MVP

<!--nl-snippet {"name":"module_nano_highlight_mvp","check":false}-->
```nano
from "modules/nano_highlight/nano_highlight.nano" import highlight_html

fn main() -> int {
    let html: string = (highlight_html "(println 1)")
    assert (> (str_length html) 0)
    return 0
}

shadow main { assert true }
```
