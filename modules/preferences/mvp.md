# preferences module MVP

<!--nl-snippet {"name":"module_preferences_mvp","check":false}-->
```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_home, nl_prefs_get_path

fn main() -> int {
    let mut home: string = ""
    let mut cfg: string = ""
    unsafe { set home (nl_prefs_get_home) }
    unsafe { set cfg (nl_prefs_get_path "nanolang") }
    assert (>= (str_length home) 0)
    assert (>= (str_length cfg) 0)
    return 0
}

shadow main { assert true }
```
