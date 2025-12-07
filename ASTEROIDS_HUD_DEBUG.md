# Asteroids HUD Debug Guide

## Issue: Score and Lives Not Displaying

I've added debug output and fallback rendering to help diagnose and fix the HUD display issue.

## Changes Made

### 1. Font Loading Check
Added verification that fonts load successfully:
```nano
let font: TTF_Font = (nl_open_font_portable "Arial" 24)
if (== font 0) {
    (println "Warning: Failed to load font. HUD text will not display.")
} else {
    (println "✓ Font loaded successfully")
}
```

When you run the game, you'll now see in the console:
- **"✓ Font loaded successfully"** - Fonts working, text should display
- **"Warning: Failed to load font..."** - Font loading failed

### 2. Fallback HUD Rendering
Added graphical fallback if fonts don't load:

**Lives Indicator (Top Left):**
- White squares, one for each life
- 3 squares = 3 lives

**Score Indicator (Bottom Right):**
- Yellow bar that grows with score
- Width = score / 10 (max 200 pixels)

**Game Over Screen:**
- Large red rectangle in center (instead of text)
- White border outline

## Testing

### Run the Game
```bash
./bin/asteroids_sdl
```

### Check Console Output
Look for one of these messages at startup:
1. **"✓ Font loaded successfully"** 
   - Fonts working
   - Should see text: "Lives: 3" (top left) and "Score: 0" (bottom right)
   
2. **"Warning: Failed to load font..."**
   - Font loading failed
   - Should see fallback: white squares for lives, yellow bar for score

### What You Should See

**If Fonts Work:**
```
Top Left: "Lives: 3" (white text)
Bottom Right: "Score: 0" (yellow text)
```

**If Fonts Don't Work (Fallback):**
```
Top Left: □ □ □ (three white squares)
Bottom Right: ▬ (yellow bar, grows with score)
```

## Possible Issues

### Issue 1: Font Path Not Found
**Symptom:** "Warning: Failed to load font" message

**Cause:** `nl_open_font_portable` couldn't find "Arial" font

**Fix Options:**
1. Install Arial font on your system
2. Change font name to one available on your system
3. Use absolute font path

**macOS Common Fonts:**
- "Helvetica"
- "Courier"
- "Times"

**Linux Common Fonts:**
- "DejaVu Sans"
- "Liberation Sans"
- "FreeSans"

### Issue 2: Text Rendered But Not Visible
**Symptom:** Font loads successfully but no text appears

**Possible Causes:**
- Text rendered behind other elements
- Text color matches background
- Text position off-screen
- SDL_TTF texture creation failing

**Debug Steps:**
1. Check if fallback rectangles work (they use basic SDL rendering)
2. Verify text position (10,10 and 620,560 should be visible)
3. Try changing text colors

### Issue 3: SDL_TTF Not Initialized
**Symptom:** Font is 0 even though file exists

**Cause:** TTF_Init() failed

**Check:** Look for SDL_TTF errors in console

## Quick Test: Force Fallback

To test if the issue is font-specific, you can temporarily force the fallback:

```nano
# Change this line:
let font: TTF_Font = (nl_open_font_portable "Arial" 24)

# To this:
let font: TTF_Font = 0  # Force null font to test fallback
```

If you see the white squares and yellow bar, the rendering pipeline works and it's just a font loading issue.

## Font Loading Alternatives

If "Arial" doesn't work, try these alternatives:

### macOS
```nano
let font: TTF_Font = (nl_open_font_portable "Helvetica" 24)
```

### Linux
```nano
let font: TTF_Font = (nl_open_font_portable "DejaVu Sans" 24)
```

### Absolute Path (macOS)
```nano
let font: TTF_Font = (TTF_OpenFont "/System/Library/Fonts/Helvetica.ttc" 24)
```

### Absolute Path (Linux)
```nano
let font: TTF_Font = (TTF_OpenFont "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 24)
```

## Expected Console Output

```
╔════════════════════════════════════════════╗
║   ASTEROIDS COMPLETE - Nanolang           ║
╚════════════════════════════════════════════╝

Controls:
  Up Arrow    - Thrust
  Left/Right  - Rotate
  Space       - Shoot
  R           - Restart (after game over)
  ESC         - Quit

Features:
  • 3 Lives!
  • Score: Large=10, Medium=30, Small=50
  • Game Over screen with restart
  • Live score display

✓ Font loaded successfully    <-- This is the key line!
```

## Verification Steps

1. **Compile:** `./bin/nanoc examples/asteroids_complete_sdl.nano -o bin/asteroids_sdl`
2. **Run:** `./bin/asteroids_sdl`
3. **Check console:** Look for font loading message
4. **Check game window:** Look for either:
   - Text HUD (if fonts work)
   - Rectangle HUD (if fonts don't work)
5. **Play game:** Shoot asteroids, verify score increases
6. **Get hit:** Verify lives decrease (text or squares)

## Next Steps

Please run the game and report back:
1. What message do you see? ("✓ Font loaded" or "Warning"?)
2. Do you see any HUD elements? (text, squares, bars?)
3. If you see fallback graphics, does the yellow score bar grow when you shoot asteroids?
4. What operating system are you on?

This will help us determine if it's:
- Font loading issue → Fix font name/path
- Rendering issue → Debug SDL_TTF
- Text positioning issue → Adjust coordinates
- Something else entirely

## Manual HUD Test

If nothing works, you can manually verify text rendering:

```bash
# Create a minimal test file to check if nl_draw_text_blended works at all
```

Let me know what you find!
