# ProTracker Clone Integration Plan

## Goal

Complete `sdl_tracker_shell.nano` integration with `~/Src/pt2-clone` ProTracker implementation.

## Current State

**Existing**:
- `modules/pt2_audio/` - Audio playback engine
- `modules/pt2_module/` - MOD file loader
- `modules/pt2_state/` - State management
- `examples/sdl_tracker_shell.nano` - Basic UI framework (incomplete)

**External**: `~/Src/pt2-clone` - Full C ProTracker implementation

## Integration Architecture

### Phase 1: MOD File Playback (MVP) - 2 days

**Goal**: Load and play MOD files

```nano
from "modules/pt2_module/pt2_module.nano" import load_mod, ModFile
from "modules/pt2_audio/pt2_audio.nano" import audio_init, audio_play_mod

fn main() -> int {
    (audio_init)
    
    let mod_file: ModFile = (load_mod "music.mod")
    if (== mod_file 0) {
        (println "Failed to load MOD file")
        return 1
    }
    
    (audio_play_mod mod_file)
    
    # Play until ESC or quit
    while true {
        if (poll_quit) {
            break
        }
        (sleep 100)
    }
    
    return 0
}
```

### Phase 2: Pattern Viewer - 2 days

**Goal**: Display patterns in SDL window

```nano
fn render_pattern(renderer: SDL_Renderer, pattern: Pattern, row: int) {
    # Render 64 rows of pattern data
    let mut r: int = 0
    while (< r 64) {
        let channel1: Note = (get_note pattern r 0)
        let channel2: Note = (get_note pattern r 1)
        let channel3: Note = (get_note pattern r 2)
        let channel4: Note = (get_note pattern r 3)
        
        # Render each channel's note
        (render_note renderer channel1 (+ 100 (* r 10)) 100)
        (render_note renderer channel2 (+ 100 (* r 10)) 150)
        (render_note renderer channel3 (+ 100 (* r 10)) 200)
        (render_note renderer channel4 (+ 100 (* r 10)) 250)
        
        set r (+ r 1)
    }
}
```

### Phase 3: Sample Editor - 3 days

**Goal**: Visualize and edit samples

- Waveform display
- Sample metadata (length, loop points, volume)
- Basic editing (trim, normalize)

### Phase 4: Instrument Panel - 2 days

**Goal**: Display and manage instruments

- Instrument list (1-31)
- Sample parameters
- Volume/finetune controls

### Phase 5: Real-time Playback Sync - 2 days

**Goal**: Sync UI with playback position

- Highlight current row
- Follow playback in pattern
- Show current pattern/position

### Phase 6: Keyboard Input - 2 days

**Goal**: ProTracker-style keyboard controls

- Number keys: Select pattern
- Space: Play/pause
- Arrow keys: Navigate
- F-keys: Functions

## Data Structures

### MOD File Format

```nano
struct ModFile {
    title: string,
    num_patterns: int,
    pattern_table: array<int>,
    patterns: array<Pattern>,
    samples: array<Sample>
}

struct Pattern {
    rows: array<PatternRow>  # 64 rows
}

struct PatternRow {
    channels: array<Note>  # 4 channels
}

struct Note {
    sample: int,      # 0-31
    period: int,      # Note pitch
    effect: int,      # Effect type (0-15)
    effect_param: int # Effect parameter
}

struct Sample {
    name: string,
    length: int,
    finetune: int,
    volume: int,
    loop_start: int,
    loop_length: int,
    data: array<int>  # Sample data
}
```

## pt2-clone Integration

**Files to integrate from ~/Src/pt2-clone**:
- `src/pt2_audio.c` → `modules/pt2_audio/pt2_audio.c`
- `src/pt2_tables.c` → `modules/pt2_module/tables.c`
- `src/pt2_replayer.c` → `modules/pt2_audio/replayer.c`

**Integration approach**:
1. Wrap pt2-clone C functions with NanoLang FFI
2. Create NanoLang structs matching C structs
3. Test with real MOD files from Amiga scene

## Test MOD Files

Use classic Amiga MOD files:
- `examples/music/4mat-enigma.mod` (4-channel, classic)
- `examples/music/kefrens-desert_dream.mod` (demo scene)
- `examples/music/purple_motion-second_reality.mod` (complex)

## UI Design

```
╔════════════════════════════════════════════════════════════════╗
║ NanoTracker - Pattern 00                          BPM: 125     ║
╠════════════════════════════════════════════════════════════════╣
║ 00│ C-3 01 000 │ --- 00 000 │ E-3 03 000 │ --- 00 000 │      ║
║ 01│ --- 00 000 │ D-3 02 000 │ --- 00 000 │ G-3 04 000 │      ║
║ 02│ C-3 01 000 │ --- 00 000 │ E-3 03 000 │ --- 00 000 │      ║
║...                                                              ║
║ 63│ C-3 01 C00 │ --- 00 000 │ --- 00 000 │ --- 00 000 │      ║
╠════════════════════════════════════════════════════════════════╣
║ [Space] Play/Pause  [←→] Navigate  [1-9] Pattern  [F1] Help  ║
╚════════════════════════════════════════════════════════════════╝

Samples: [01] Piano   [02] Bass    [03] Strings  [04] Drums
```

## Implementation Timeline

- Phase 1: MOD Playback - 2 days
- Phase 2: Pattern Viewer - 2 days
- Phase 3: Sample Editor - 3 days
- Phase 4: Instrument Panel - 2 days
- Phase 5: Playback Sync - 2 days
- Phase 6: Keyboard Input - 2 days

**Total**: 13 days

**MVP** (Phases 1-2): 4 days

## Success Criteria

✅ Load and play real MOD files
✅ Display patterns in ProTracker format
✅ Navigate patterns with keyboard
✅ Show sample waveforms
✅ Sync UI with playback
✅ Match ProTracker UI aesthetics

## Status

**Planning**: ✅ Complete
**Implementation**: ⏸️  Ready to start (4 days for MVP)
**Priority**: P1 (showcase application)

This provides comprehensive roadmap for ProTracker integration.
