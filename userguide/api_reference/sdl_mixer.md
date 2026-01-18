# sdl_mixer API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn Mix_Init(_flags: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_flags` | `int` |

**Returns:** `int`


#### `extern fn Mix_Quit() -> void`

**Returns:** `void`


#### `extern fn Mix_OpenAudio(_frequency: int, _format: int, _channels: int, _chunksize: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_frequency` | `int` |
| `_format` | `int` |
| `_channels` | `int` |
| `_chunksize` | `int` |

**Returns:** `int`


#### `extern fn Mix_CloseAudio() -> void`

**Returns:** `void`


#### `extern fn Mix_AllocateChannels(_numchans: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_numchans` | `int` |

**Returns:** `int`


#### `extern fn Mix_LoadWAV(_file: string) -> Mix_Chunk`

**Parameters:**
| Name | Type |
|------|------|
| `_file` | `string` |

**Returns:** `Mix_Chunk`


#### `extern fn Mix_FreeChunk(_chunk: Mix_Chunk) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_chunk` | `Mix_Chunk` |

**Returns:** `int`


#### `extern fn Mix_PlayChannel(_channel: int, _chunk: Mix_Chunk, _loops: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |
| `_chunk` | `Mix_Chunk` |
| `_loops` | `int` |

**Returns:** `int`


#### `extern fn Mix_PlayChannelTimed(_channel: int, _chunk: Mix_Chunk, _loops: int, _ticks: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |
| `_chunk` | `Mix_Chunk` |
| `_loops` | `int` |
| `_ticks` | `int` |

**Returns:** `int`


#### `extern fn Mix_FadeInChannel(_channel: int, _chunk: Mix_Chunk, _loops: int, _ms: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |
| `_chunk` | `Mix_Chunk` |
| `_loops` | `int` |
| `_ms` | `int` |

**Returns:** `int`


#### `extern fn Mix_HaltChannel(_channel: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |

**Returns:** `int`


#### `extern fn Mix_FadeOutChannel(_channel: int, _ms: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |
| `_ms` | `int` |

**Returns:** `int`


#### `extern fn Mix_Volume(_channel: int, _volume: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |
| `_volume` | `int` |

**Returns:** `int`


#### `extern fn Mix_VolumeChunk(_chunk: Mix_Chunk, _volume: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_chunk` | `Mix_Chunk` |
| `_volume` | `int` |

**Returns:** `int`


#### `extern fn Mix_LoadMUS(_file: string) -> Mix_Music`

**Parameters:**
| Name | Type |
|------|------|
| `_file` | `string` |

**Returns:** `Mix_Music`


#### `extern fn Mix_FreeMusic(_music: Mix_Music) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_music` | `Mix_Music` |

**Returns:** `void`


#### `extern fn Mix_PlayMusic(_music: Mix_Music, _loops: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_music` | `Mix_Music` |
| `_loops` | `int` |

**Returns:** `int`


#### `extern fn Mix_FadeInMusic(_music: Mix_Music, _loops: int, _ms: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_music` | `Mix_Music` |
| `_loops` | `int` |
| `_ms` | `int` |

**Returns:** `int`


#### `extern fn Mix_FadeInMusicPos(_music: Mix_Music, _loops: int, _ms: int, _position: float) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_music` | `Mix_Music` |
| `_loops` | `int` |
| `_ms` | `int` |
| `_position` | `float` |

**Returns:** `int`


#### `extern fn Mix_HaltMusic() -> int`

**Returns:** `int`


#### `extern fn Mix_FadeOutMusic(_ms: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_ms` | `int` |

**Returns:** `int`


#### `extern fn Mix_RewindMusic() -> int`

**Returns:** `int`


#### `extern fn Mix_PauseMusic() -> void`

**Returns:** `void`


#### `extern fn Mix_ResumeMusic() -> void`

**Returns:** `void`


#### `extern fn Mix_VolumeMusic(_volume: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_volume` | `int` |

**Returns:** `int`


#### `extern fn Mix_Playing(_channel: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |

**Returns:** `int`


#### `extern fn Mix_Paused(_channel: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_channel` | `int` |

**Returns:** `int`


#### `extern fn Mix_PlayingMusic() -> int`

**Returns:** `int`


#### `extern fn Mix_PausedMusic() -> int`

**Returns:** `int`


#### `extern fn Mix_SetPostMix(_callback: unknown, _arg: void) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_callback` | `unknown` |
| `_arg` | `void` |

**Returns:** `void`


#### `extern fn Mix_GetNumChannels() -> int`

**Returns:** `int`


#### `extern fn Mix_GetError() -> string`

**Returns:** `string`


#### `extern fn Mix_ClearError() -> int`

**Returns:** `int`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

- `opaque type Mix_Chunk`
- `opaque type Mix_Music`

### Constants

| Name | Type | Value |
|------|------|-------|
| `MIX_INIT_FLAC` | `int` | `1` |
| `MIX_INIT_MOD` | `int` | `2` |
| `MIX_INIT_MP3` | `int` | `8` |
| `MIX_INIT_OGG` | `int` | `16` |
| `MIX_INIT_MID` | `int` | `32` |
| `MIX_INIT_OPUS` | `int` | `64` |
| `MIX_DEFAULT_FORMAT` | `int` | `32784` |
| `MIX_MAX_VOLUME` | `int` | `128` |

