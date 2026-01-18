# audio_helpers module MVP

<!--nl-snippet {"name":"module_audio_helpers_mvp","check":false}-->
```nano
from "modules/audio_helpers/audio_helpers.nano" import audio_convert_raw_to_wav

fn main() -> int {
    if false {
        unsafe { (audio_convert_raw_to_wav "/tmp/nl_audio.raw" "/tmp/nl_audio.wav" 44100) }
    }
    return 0
}

shadow main { assert true }
```
