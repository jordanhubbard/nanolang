# Asteroids - Full Arcade Game

## Overview

Complete implementation of the classic Asteroids arcade game with authentic gameplay mechanics, lives system, game over screen, and HUD display.

## Features

### Core Gameplay
- **Multi-size asteroids** (Large/Medium/Small) with speed scaling
- **Breaking mechanics** - Large splits into 2 Medium, Medium splits into 3 Small
- **Thrust cone visualization** - Orange flame when accelerating
- **Particle explosions** - Scaled by asteroid size
- **Wrapping screen edges** - True arcade feel

### Game Systems
- **3 Lives** - Ship respawns after 2-second delay
- **Game Over Screen** - Big red text with final score
- **Restart System** - Press R to play again
- **Progressive Scoring** - Large=10, Medium=30, Small=50

### HUD Display
- **Lives Counter** - Top left, white text
- **Score Display** - Lower right, yellow text
- Real-time updates during gameplay

## Controls

| Key         | Action       |
|-------------|--------------|
| Up Arrow    | Thrust       |
| Left Arrow  | Rotate Left  |
| Right Arrow | Rotate Right |
| Space       | Shoot        |
| R           | Restart      |
| ESC         | Quit         |

## Building and Running

```bash
# Build
make -C examples asteroids-sdl

# Or directly
./bin/nanoc examples/asteroids_complete_sdl.nano -o bin/asteroids_sdl

# Run
./bin/asteroids_sdl
```

## Gameplay

1. **Start** - 3 lives, score 0, three large asteroids spawn
2. **Shoot Asteroids** - Large breaks into 2 Medium, Medium into 3 Small
3. **Avoid Collisions** - Lose 1 life when hit, 2-second respawn delay
4. **Score Points** - Small asteroids worth most (50 points each)
5. **Game Over** - After 3 deaths, press R to restart or ESC to quit

## Scoring Strategy

Breaking down a single large asteroid completely:
- Large asteroid: **10 points**
- 2 Medium asteroids: **60 points** (2 × 30)
- 6 Small asteroids: **300 points** (6 × 50)
- **Total: 370 points** per large asteroid!

The key is to break them all down to small size for maximum score.

## Technical Details

### Source Files
- **examples/asteroids_complete_sdl.nano** (687 lines) - Full implementation
- **examples/asteroids_enhanced_sdl.nano** - Enhanced version (breaking, no lives)
- **examples/asteroids_sdl.nano** - Basic version (learning example)

### Makefile
The Makefile builds the complete version by default:
```bash
make asteroids-sdl  # Builds the full game
```

### Dependencies
- SDL2 (graphics, input, timing)
- SDL2_ttf (text rendering)
- Modules: sdl, sdl_helpers, sdl_ttf, sdl_ttf_helpers

### Game State Machine
- **STATE_PLAYING** - Normal gameplay
- **STATE_DEAD** - Respawn timer countdown
- **STATE_GAME_OVER** - Waiting for restart

## Code Statistics

- 687 lines of nanolang code
- 5 shadow tests (all passing)
- 3 game states
- Parallel arrays for game objects
- Cross-platform font loading
- Real-time HUD rendering

## Features Comparison

| Feature              | Basic | Enhanced | **Complete** |
|----------------------|-------|----------|--------------|
| Asteroid sizes       | 1     | 3        | 3            |
| Breaking mechanics   | ✗     | ✓        | ✓            |
| Thrust visualization | ✗     | ✓        | ✓            |
| Lives system         | ✗     | ✗        | **✓**        |
| Game over screen     | ✗     | ✗        | **✓**        |
| Restart              | ✗     | ✗        | **✓**        |
| HUD display          | ✗     | ✗        | **✓**        |
| Score tracking       | ✗     | ✗        | **✓**        |

**The Complete version is the recommended version to play!**

## Implementation Highlights

### SDL_TTF Integration
Uses the `sdl_ttf_helpers` module for easy font handling:
```nano
let font: TTF_Font = (nl_open_font_portable "Arial" 24)
(nl_draw_text_blended renderer font text x y r g b a)
```

### Lives Management
```nano
# On collision
set lives (- lives 1)
set game_state STATE_DEAD
set respawn_timer RESPAWN_DELAY

# Check for respawn or game over
if (<= respawn_timer 0.0) {
    if (> lives 0) {
        # Respawn
    } else {
        set game_state STATE_GAME_OVER
    }
}
```

### Restart System
```nano
if (== key 21) {  # R key
    if (== game_state STATE_GAME_OVER) {
        # Reset entire game state
        set lives STARTING_LIVES
        set score 0
        # Clear and respawn asteroids
    }
}
```

## Why Three Versions?

While the Makefile builds the complete version by default, the source files for all three versions are available for educational purposes:

1. **asteroids_sdl.nano** - Shows parallel arrays pattern, good for learning
2. **asteroids_enhanced_sdl.nano** - Demonstrates breaking mechanics and particles
3. **asteroids_complete_sdl.nano** - Full game with all features ⭐

Users should play the complete version, but developers can study the simpler versions to understand the progression.

## Quick Start

```bash
# One command to build and run:
make -C examples asteroids-sdl && ./bin/asteroids_sdl
```

## Success!

All requirements have been successfully implemented:
- ✅ 3 lives system
- ✅ Game over screen with restart
- ✅ Updated scoring (10/30/50)
- ✅ Score display in lower right
- ✅ Lives display in top left
- ✅ Breaking asteroids mechanics
- ✅ Thrust cone visualization
- ✅ Professional arcade experience

**Enjoy the full Asteroids arcade game in nanolang!** 🚀🎮
