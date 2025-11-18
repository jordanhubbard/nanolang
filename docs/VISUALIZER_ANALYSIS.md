# MOD Player Visualizer - Complexity Analysis

## Goal
Add a simple visualizer to the MOD player example

## Option A: Port pt2-clone Visualizer

### Scope Analysis
**pt2_scopes.c:** ~1,600 lines
- Waveform rendering for each channel
- Sample scope visualization
- Complex rendering pipeline
- Tightly coupled to Paula emulation

**Dependencies:**
- pt2_audio.h - audio state access
- pt2_helpers.h - utility functions
- pt2_palette.h - color schemes
- pt2_config.h - configuration
- pt2_structs.h - module structures
- Direct framebuffer access

**Estimated Effort:**
- Extract scope logic: 3-5 days
- Rewrite for SDL2 textures: 2-3 days
- Integrate with nanolang: 2-3 days  
- **Total: 7-11 days**

**Pros:**
- Professional-quality visualization
- Authentic ProTracker look
- Feature-complete

**Cons:**
- Complex codebase (1,600 lines)
- Requires pt2-clone internals knowledge
- Overkill for example/demo

## Option B: Simple SDL2 Visualizer

### Approach
Build minimalist visualizer with SDL2 primitives:

1. **Waveform Display** (Simple)
   - Read audio buffer samples
   - Draw as lines with SDL_RenderDrawLines
   - ~50-100 lines

2. **VU Meters** (Very Simple)
   - Calculate RMS or peak values
   - Draw rectangles with SDL_RenderFillRect
   - ~30-50 lines

3. **Spectrum Analyzer** (Medium)
   - Basic FFT or simple frequency bins
   - Draw bars with SDL_RenderFillRect
   - ~100-150 lines

4. **Pattern Display** (Simple)
   - Show current row/pattern info as text
   - Use TTF_RenderText functions
   - ~50 lines

**Total Lines:** 230-350 lines
**Estimated Effort:** 2-3 days

**Pros:**
- Clean, simple code
- Demonstrates SDL2 graphics
- Good learning example
- Easy to understand/modify
- No pt2-clone dependencies

**Cons:**
- Not authentic ProTracker look
- Basic visualizations only
- No advanced features

## Recommended Option: **Option B - Simple SDL2 Visualizer**

### Rationale
1. **Fits Example Purpose:** Shows SDL2 graphics capabilities
2. **Pragmatic:** 2-3 days vs 7-11 days
3. **Maintainable:** 300 lines vs 1,600 lines
4. **Educational:** Clear, simple code

### Implementation Plan

**Phase 1: Audio Data Access (Day 1)**
- Get audio buffer from SDL_mixer
- Calculate waveform samples
- Calculate VU meter values

**Phase 2: Basic Visualizations (Day 2)**
- Waveform display (lines)
- VU meters (rectangles)
- Pattern info (text)

**Phase 3: Polish (Day 3)**
- Colors and styling
- Smooth animations
- Performance optimization

## Minimal Example
```nano
fn draw_waveform(renderer: int, samples: array<int>, width: int, height: int) -> void {
    (SDL_SetRenderDrawColor renderer 0 255 0 255)
    
    let scale: float = (cast_float height) / 256.0
    let i: int = 0
    while (< i (- width 1)) {
        let sample1: int = (array_get samples i)
        let sample2: int = (array_get samples (+ i 1))
        let y1: int = (cast_int (* (cast_float sample1) scale))
        let y2: int = (cast_int (* (cast_float sample2) scale))
        (SDL_RenderDrawLine renderer i y1 (+ i 1) y2)
        set i = (+ i 1)
    }
}
```

**Estimated Lines:** ~250 lines total
**Estimated Time:** 2-3 days

## Decision
Proceed with **Option B** - simple SDL2 visualizer.
