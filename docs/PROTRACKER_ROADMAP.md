# ProTracker Clone - Porting Roadmap

## Overview
Porting pt2-clone (~47K lines) to nanolang. This is a **7-8 month project** with clear milestones.

## Current Status
- **Date:** November 17, 2024
- **Lines Ported:** 1,465 / ~47,000 (3%)
- **Modules Complete:** 5 basic modules + infrastructure
- **Build Status:** ✅ Compiles (92KB binary)
- **Playback Status:** ❌ Loads MODs but no audio yet

### What Works
✅ FFI pointer casting system  
✅ Multi-file compilation  
✅ MOD file loading (pattern data parsing)  
✅ UI framework (temporarily displays colored square)  
✅ Pattern data structures  
✅ Basic replayer state management  

### What's Missing
❌ Audio output (no sound yet)  
❌ Sample playback  
❌ Effect processing  
❌ Visual scopes/meters  
❌ Pattern editor  
❌ Sample editor  

---

## pt2-clone Components (47K lines total)

### 🔴 CRITICAL: Core Playback (5,023 lines)
| Component | Lines | Status | Priority |
|-----------|-------|--------|----------|
| Basic structure | 1,465 | ✅ Complete | Done |
| pt2_replayer.c | 1,911 | ⚠️ Partial | **M1** |
| pt2_paula.c | 424 | ❌ Not started | **M1** |
| pt2_sampler.c | 2,688 | ❌ Not started | **M1** |

### 🟡 Audio & Effects (3,000+ lines)
| Component | Lines | Status | Priority |
|-----------|-------|--------|----------|
| pt2_audio.c | 17K | ❌ Not started | M1 |
| pt2_blep.c | 9K | ❌ Not started | M2 |
| pt2_downsample2x.c | 7K | ❌ Not started | M2 |

### 🟢 Editor & UI (6,000+ lines)
| Component | Lines | Status | Priority |
|-----------|-------|--------|----------|
| Basic UI | 136 | ✅ Complete | Done |
| pt2_edit.c | 26K | ❌ Not started | M4 |
| pt2_keyboard.c | 13K | ❌ Not started | M4 |
| pt2_mouse.c | 19K | ❌ Not started | M4 |
| pt2_visuals.c | 31K | ❌ Not started | M3 |

### 🔵 File Operations (3,000+ lines)
| Component | Lines | Status | Priority |
|-----------|-------|--------|----------|
| Basic MOD loader | 151 | ✅ Complete | Done |
| pt2_module_loader.c | 25K | ❌ Not started | M2 |
| pt2_module_saver.c | 15K | ❌ Not started | M4 |
| pt2_diskop.c | 23K | ❌ Not started | M6 |

### ⚪ Tools & Features (5,000+ lines)
| Component | Lines | Status | Priority |
|-----------|-------|--------|----------|
| pt2_sampler_editor.c | 44K | ❌ Not started | M5 |
| pt2_scopes.c | 15K | ❌ Not started | M3 |
| pt2_spectrum.c | 6K | ❌ Not started | M3 |
| pt2_chordmaker.c | 12K | ❌ Not started | M6 |
| pt2_config.c | 20K | ❌ Not started | M6 |

---

## Milestone Plan

### ✅ Milestone 0: Infrastructure (COMPLETE)
**Duration:** Completed  
**Lines:** ~1,500

**Achievements:**
- ✅ FFI pointer casting for C libraries
- ✅ Multi-file project compilation
- ✅ MOD file loading infrastructure
- ✅ Mutable state via extern C (pt2_state.c)
- ✅ Basic UI framework
- ✅ Pattern data structures
- ✅ Period/note conversion tables

---

### 🎯 Milestone 1: Basic Playback (IN PROGRESS)
**Goal:** Play a MOD file with audible sound  
**Duration:** 2 weeks  
**Lines to Add:** ~1,000

**Tasks:**
1. **Audio Output Integration** (300 lines)
   - SDL audio callback setup
   - 4-channel mixer
   - Buffer management
   
2. **Sample Playback** (400 lines)
   - Load sample data from MOD
   - Trigger samples on notes
   - Volume control
   - Sample interpolation
   
3. **Basic Effects** (300 lines)
   - 0x0C: Set Volume
   - 0x0F: Set Speed/BPM
   - 0x00: Arpeggio (basic)

**Success Criteria:**
- ✅ Load gabba-studies-12.mod
- ✅ Hear recognizable music
- ✅ Volume control works
- ✅ Tempo changes work

**Files to Create/Modify:**
- `audio.nano` (NEW - 300 lines) - SDL audio integration
- `sampler.nano` (NEW - 400 lines) - Sample playback
- `replayer.nano` (enhance - +300 lines) - Effect processing

---

### Milestone 2: Full Effects (1 month)
**Goal:** All ProTracker effects working  
**Duration:** 1 month  
**Lines to Add:** ~1,500

**Tasks:**
1. Port all 36 effects from pt2_replayer.c
2. Pattern jumping (0x0B Position Jump, 0x0D Pattern Break)
3. Portamento (0x01, 0x02, 0x03)
4. Vibrato (0x04) and Tremolo (0x07)
5. Fine control effects (0xE_ series)

**Success Criteria:**
- ✅ All standard MOD files play correctly
- ✅ Effects sound identical to pt2-clone
- ✅ Pass effect test MODs

---

### Milestone 3: Visual Feedback (1 month)
**Goal:** See what you hear  
**Duration:** 1 month  
**Lines to Add:** ~1,000

**Tasks:**
1. Oscilloscopes (4 channels)
2. VU meters
3. Pattern position display
4. Sample waveform display
5. Spectrum analyzer (optional)

**Success Criteria:**
- ✅ Scopes show correct waveforms
- ✅ Pattern scrolls during playback
- ✅ VU meters respond to volume

---

### Milestone 4: Pattern Editor (2 months)
**Goal:** Edit patterns and save MODs  
**Duration:** 2 months  
**Lines to Add:** ~2,500

**Tasks:**
1. Keyboard input for notes
2. Pattern editing (insert, delete, copy, paste)
3. Module saving
4. Undo/redo system
5. Transpose, quantize tools

**Success Criteria:**
- ✅ Create new patterns from scratch
- ✅ Edit existing songs
- ✅ Save modifications to disk
- ✅ Undo/redo works

---

### Milestone 5: Sample Editor (2 months)
**Goal:** Edit and manipulate samples  
**Duration:** 2 months  
**Lines to Add:** ~2,000

**Tasks:**
1. Sample waveform editing
2. Cut, copy, paste operations
3. Resample, normalize
4. Sample generation tools
5. Loop point editing

**Success Criteria:**
- ✅ Edit sample data
- ✅ Generate new samples
- ✅ Loop points work correctly

---

### Milestone 6: Polish & Features (1 month)
**Goal:** Feature parity with pt2-clone  
**Duration:** 1 month  
**Lines to Add:** ~1,000

**Tasks:**
1. Configuration system
2. Disk operations (load/save)
3. Help system
4. Keyboard shortcuts
5. Bug fixes and optimizations

**Success Criteria:**
- ✅ Full feature set
- ✅ Stable and bug-free
- ✅ Ready for users

---

## Timeline

```
Milestone 0: Infrastructure          [████████████] COMPLETE
Milestone 1: Basic Playback          [██░░░░░░░░░░] 2 weeks  ← YOU ARE HERE
Milestone 2: Full Effects            [░░░░░░░░░░░░] 1 month
Milestone 3: Visual Feedback         [░░░░░░░░░░░░] 1 month
Milestone 4: Pattern Editor          [░░░░░░░░░░░░] 2 months
Milestone 5: Sample Editor           [░░░░░░░░░░░░] 2 months
Milestone 6: Polish                  [░░░░░░░░░░░░] 1 month
─────────────────────────────────────────────────────────────
Total:                                             7-8 months
```

---

## Immediate Next Steps

### This Session Focus: Start Milestone 1

1. **Create audio.nano** - SDL audio callback
   - Initialize SDL audio
   - Set up 4-channel mixing
   - Buffer management
   
2. **Create sampler.nano** - Sample playback
   - Load sample data from MOD
   - Trigger samples with period/volume
   - Mix to audio buffer
   
3. **Enhance replayer.nano** - Process effects
   - Add effect 0x0C (volume)
   - Add effect 0x0F (speed/BPM)
   - Call sampler functions

4. **Test with gabba-studies-12.mod**
   - Hear actual music!
   - Verify tempo is correct
   - Check for glitches

---

## Technical Strategy

### Simplified Approach for Milestone 1

Instead of porting full Paula emulation (424 lines + BLEP synthesis), we'll use **SDL_mixer's built-in mixing**:

**Advantages:**
- ✅ Much simpler (~300 lines vs 2,000+)
- ✅ Hardware-accelerated
- ✅ Cross-platform
- ✅ Good enough for Milestone 1

**Later (Milestone 2):**
- Port Paula emulation for authenticity
- BLEP synthesis for anti-aliasing
- Filters (LED, lowpass, highpass)

### Code Organization

```
examples/protracker-clone/
├── main.nano          (236 lines) - Entry point, UI loop
├── types.nano         (248 lines) - Period tables, constants
├── pattern.nano       (149 lines) - Pattern data structures
├── mod_loader.nano    (151 lines) - MOD file parsing
├── replayer.nano      (305 lines) - Playback engine
├── audio.nano         (NEW)       - SDL audio integration
├── sampler.nano       (NEW)       - Sample playback
├── ui.nano            (136 lines) - UI rendering
└── effects.nano       (NEW)       - Effect processing

modules/pt2_state/
├── pt2_state.c        (199 lines) - Mutable state in C
└── pt2_state.nano     (43 lines)  - State FFI bindings
```

---

## Success Metrics

### Milestone 1 Success:
- ✅ Can hear music from gabba-studies-12.mod
- ✅ Recognizable melody
- ✅ Tempo approximately correct
- ✅ No crashes or glitches

### Full Clone Success (Milestone 6):
- ✅ 100% MOD compatibility
- ✅ All effects working perfectly
- ✅ Full editing capabilities
- ✅ Sample editor functional
- ✅ Stable and performant
- ✅ Authentic Amiga sound

---

## Notes

- pt2-clone is 47K lines - this is a **serious undertaking**
- Focus on **incremental progress** - each milestone delivers value
- Use **pragmatic shortcuts** early (SDL_mixer vs Paula emulation)
- **Test continuously** with real MOD files
- **Document everything** - future you will thank you

---

**Last Updated:** November 17, 2024  
**Status:** Milestone 1 in progress  
**Next Session:** Create audio.nano and sampler.nano
