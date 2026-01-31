# audio_viz module MVP

<!--nl-snippet {"name":"module_audio_viz_mvp","check":false}-->
```nano
from "modules/audio_viz/audio_viz.nano" import nl_audio_viz_init, nl_audio_viz_get_waveform_size, nl_audio_viz_shutdown

fn main() -> int {
    let mut size: int = 0
    unsafe { (nl_audio_viz_init 0 2) }
    unsafe { set size (nl_audio_viz_get_waveform_size) }
    assert (>= size 0)
    unsafe { (nl_audio_viz_shutdown) }
    return 0
}

shadow main { assert true }
```
