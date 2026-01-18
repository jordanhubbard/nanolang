# pt2_state module MVP

<!--nl-snippet {"name":"module_pt2_state_mvp","check":false}-->
```nano
from "modules/pt2_state/pt2_state.nano" import pt2_init_state, pt2_is_playing, pt2_stop_playback

fn main() -> int {
    let mut playing: int = 0
    unsafe { (pt2_init_state) }
    unsafe { set playing (pt2_is_playing) }
    assert (>= playing 0)
    unsafe { (pt2_stop_playback) }
    return 0
}

shadow main { assert true }
```
