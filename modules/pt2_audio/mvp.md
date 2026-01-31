# pt2_audio module MVP

<!--nl-snippet {"name":"module_pt2_audio_mvp","check":false}-->
```nano
from "modules/pt2_audio/pt2_audio.nano" import pt2_audio_init, pt2_audio_stop

fn main() -> int {
    let mut code: int = 0
    unsafe { set code (pt2_audio_init 44100) }
    assert (>= code -1)
    unsafe { (pt2_audio_stop) }
    return 0
}

shadow main { assert true }
```
