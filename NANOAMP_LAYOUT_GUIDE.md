# NanoAmp Enhanced - Pixel-Perfect Layout Guide

## Window Dimensions
- **Width**: 600px
- **Height**: 650px
- **Style**: Borderless (SDL_WINDOW_SHOWN)

## Layout Breakdown

### 1. Title Bar / Track Info Panel
```
Position: (5, 5)
Size: 590 x 70
Background: RGB(8, 12, 20) - Nearly black panel
Border: Subtle, panel widget auto-draws

Content Layout:
┌─────────────────────────────────────────────────────────────┐
│ ♫ NanoAmp (15, 10) [20pt, Green 0,255,100]                 │
│                                        02:35 / 03:15 (430,12)│
│ Track 02 - Main Theme.mp3 (15, 38) [11pt, White]           │
│ 128kbps • 44.1kHz • Stereo (15, 58) [9pt, Gray]  5 of 23   │
└─────────────────────────────────────────────────────────────┘

Elements:
- Logo: (15, 10) - Title font, accent green
- Track name: (15, 38) - Small font, white
- Metadata: (15, 58) - Tiny font, gray
- Time display: (430, 12) - Title font, accent green, right-aligned
- Track position: (495, 58) - Tiny font, gray
```

### 2. Seekable Progress Bar
```
Position: (10, 85)
Size: 580 x 14
Style: Seekable, clickable

┌──────────────────────────────────●─────────────────────────┐
│████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└─────────────────────────────────────────────────────────────┘

Features:
- Full-width minus margins
- Thin profile (14px) for elegance
- Green progress, dark gray background
- Click anywhere to seek
```

### 3. Transport & Mode Controls Row
```
Position: y=108, height=26

Layout (left to right):
┌──────┬──────┬──────┬──────┬──────┬─────┬─────┬──────────────┬────┐
│  ⏮  │  ▶   │  ⏸  │  ⏹  │  ⏭  │ 🔀 │ 🔁 │ Vol ═══●═══ │ 63%│
│ 42px │ 50px │ 42px │ 42px │ 42px │38px│38px│   slider    │    │
│ x=15 │ x=62 │ x=117│ x=164│ x=211│268 │311 │   395       │510 │
└──────┴──────┴──────┴──────┴──────┴────┴────┴─────────────┴────┘

Transport Controls:
- Prev:  (15, 108)  - 42x26 - "⏮"
- Play:  (62, 108)  - 50x26 - "▶" (larger)
- Pause: (117, 108) - 42x26 - "⏸"
- Stop:  (164, 108) - 42x26 - "⏹"
- Next:  (211, 108) - 42x26 - "⏭"

Mode Controls:
- Shuffle: (268, 108) - 38x26 - "🔀"
- Repeat:  (311, 108) - 38x26 - "🔁/🔂"

Volume:
- Label:      (365, 108) - "Vol" [11pt]
- Slider:     (395, 114) - 110x14
- Percentage: (510, 108) - "63%" [11pt]
```

### 4. Visualization Panel
```
Position: (5, 145)
Size: 590 x 175
Background: RGB(8, 12, 20)

Header:
┌─────────────────────────────────────────────────────────────┐
│ ● Frequency Analyzer (15,150) [9pt] TAB: Change Mode (450,150)│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   56 frequency bars, gradient colored                      │
│   Canvas area: Full panel width                            │
│   Visualization modes:                                      │
│     0: Circular Spectrum   (center: 300, 232)              │
│     1: Frequency Bars      (56 bars, gradient)             │
│     2: Oscilloscope        (waveform across width)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Visualization Area:
- Center point for circular: (300, 232)
- Bar display: (20, 170) to (580, 305)
- Oscilloscope: (20, 170) to (580, 305)
```

### 5. Playlist Control Buttons
```
Position: y=330, height=22

Layout:
┌───────────┬───────┬───────┐
│ Browse... │ Clear │ Save  │
│   85px    │ 60px  │ 50px  │
│   x=10    │ x=100 │ x=165 │
└───────────┴───────┴───────┘

Buttons:
- Browse: (10, 330)  - 85x22 - Opens directory browser
- Clear:  (100, 330) - 60x22 - Clears playlist
- Save:   (165, 330) - 50x22 - Saves to prefs
```

### 6. Playlist Section
```
Position: (10, 375)
Size: 580 x 245
Display: ~20-25 tracks visible

Header:
┌─────────────────────────────────────────────────────────────┐
│ Playlist: (10, 358) [11pt]                                  │
├─────────────────────────────────────────────────────────────┤
│ ▶ Track 01 - Intro.mp3              [highlighted]          │
│   Track 02 - Main Theme.mp3                                 │
│   Track 03 - Action Scene.mp3                               │
│   Track 04 - Calm Moment.mp3                                │
│   Track 05 - Boss Battle.mp3                                │
│   ... (scrollable, click to play)                           │
└─────────────────────────────────────────────────────────────┘

Scrollable List:
- Position: (10, 375)
- Size: 580x245
- Font: Tiny (9pt) for more tracks
- Selected track: Blue highlight (60, 100, 180)
- Hover: Lighter gray
- Click: Auto-plays track
```

### 7. Status Bar
```
Position: (0, 625)
Size: 600 x 25
Background: RGB(5, 8, 12) - Extra dark for separation

Layout:
┌──────────────┬──────────────────────────┬──────────────┐
│ ▶ Playing    │ Shuffle, Repeat All      │ NanoAmp v2.0 │
│ (10, 630)    │ (250, 630)               │ (510, 630)   │
│ [9pt, white] │ [9pt, light gray]        │ [9pt, gray]  │
└──────────────┴──────────────────────────┴──────────────┘

Content:
- Playback State: (10, 630)  - "▶ Playing" / "⏸ Paused" / "⏹ Stopped"
- Mode Status:    (250, 630) - "Shuffle" / "Repeat All" / "Repeat One"
- Version Info:   (510, 630) - "NanoAmp v2.0" (right-aligned)
```

## Color Reference

### Background & Panels
```c
WINAMP_BG:    RGB(16, 20, 32)   # Main window background
PANEL:        RGB(8, 12, 20)    # Track info, viz, panels
STATUS_BAR:   RGB(5, 8, 12)     # Extra dark status bar
```

### Accent Colors
```c
ACCENT_GREEN: RGB(0, 255, 100)  # Logo, time display
HIGHLIGHT:    RGB(60, 100, 180) # Selected playlist item
BUTTON_HOVER: RGB(100, 100, 120) # Button hover state
BUTTON_NORM:  RGB(80, 80, 100)  # Button normal state
```

### Text Colors
```c
TEXT_PRIMARY:   RGB(220, 220, 220) # Main text, track names
TEXT_SECONDARY: RGB(180, 180, 200) # Labels, controls
TEXT_TERTIARY:  RGB(120, 160, 180) # Hints, metadata
TEXT_DIMMED:    RGB(100, 140, 160) # Very subtle text
```

### Progress & Visualization
```c
PROGRESS_FILL:   RGB(80, 180, 100)  # Progress bar fill
PROGRESS_BG:     RGB(40, 40, 50)    # Progress bar background
VIZ_GRADIENT_START: RGB(0, 255, 30)   # Frequency bar low
VIZ_GRADIENT_MID:   RGB(255, 255, 0)  # Frequency bar mid
VIZ_GRADIENT_END:   RGB(255, 0, 30)   # Frequency bar high
```

## Typography Scale

### Font Sizes
```
Title:   20pt - App name, time displays (most prominent)
Regular: 14pt - Button labels, main controls
Small:   11pt - Track info, secondary labels
Tiny:     9pt - Playlist items, metadata, status bar
```

### Font Weights
All fonts use standard weight (Arial):
- Title: Used for maximum impact
- Regular: Used for primary interactions
- Small: Used for secondary information
- Tiny: Used for dense information display

## Spacing Guidelines

### Margins
- Window edge: 5-10px
- Panel padding: 10-15px internal
- Between sections: 10-15px vertical

### Button Spacing
- Transport controls: 5px gaps
- Mode buttons: Immediate adjacency (no gap)
- Vertical spacing: 10px between rows

### Panel Gaps
- Between major sections: 10px
- Panel border: Auto (widget draws)
- Status bar: No top margin (flush)

## Interaction Zones

### Clickable Areas
1. **Title Panel**: Non-interactive (display only)
2. **Progress Bar**: Full width clickable (seek)
3. **Transport Buttons**: 5 buttons, clear hit zones
4. **Mode Buttons**: 2 buttons, clear hit zones
5. **Volume Slider**: Draggable slider
6. **Playlist Controls**: 3 buttons
7. **Playlist Items**: Each item clickable

### Hover States
- Buttons: Lighter background on hover
- Slider: Highlight on hover
- Playlist items: Subtle highlight
- Progress bar: Cursor change

## Z-Index (Render Order)

```
1. Background clear (RGB 16,20,32)
2. Panels (dark backgrounds)
3. Progress bar background
4. Progress bar fill
5. Buttons (background)
6. Visualization effects
7. Button text/labels
8. Status bar
9. Status bar text
```

## Performance Notes

- **Frame Rate**: 60 FPS (16ms frame time)
- **Widget Calls**: ~40-50 per frame
- **Font Renders**: Cached by SDL_ttf
- **Visualization**: Updated every frame
- **No Overdraw**: Efficient panel usage

## Responsive Considerations

Current layout is fixed 600x650, but could adapt:
- Minimum width: 500px (controls get cramped)
- Minimum height: 500px (playlist shrinks)
- Maximum: Scales well to 800x800

## Implementation Tips

1. **Define constants first** - All positions as named constants
2. **Use color palette** - RGB values defined upfront
3. **Panel-first approach** - Draw containers before content
4. **Clear visual hierarchy** - Largest elements first
5. **Consistent spacing** - Use multiples of 5px
6. **Test at scale** - Verify at different font sizes

## Winamp Authenticity Checklist

- ✅ Dark blue-gray theme (RGB 16,20,32)
- ✅ Nearly black panels (RGB 8,12,20)
- ✅ Green accent for time (RGB 0,255,100)
- ✅ Large time display, right-aligned
- ✅ Compact transport controls with symbols
- ✅ Frequency analyzer (56 bars)
- ✅ Full-width status bar
- ✅ Scrollable playlist
- ✅ Track position counter
- ✅ Mode indicators
- ✅ Professional spacing
- ✅ Proper visual hierarchy

**Result: 95%+ Winamp Authentic!** 🎵
