# Nanolang Examples Roadmap

## ✅ Phase 1 Complete: Cleanup (Dec 2024)

**Removed 6 duplicate files (~98KB):**
- ❌ asteroids_enhanced_sdl.nano, asteroids_sdl.nano → ✅ Keep asteroids_complete_sdl.nano
- ❌ raytracer_classic.nano, raytracer_demo.nano → ✅ Keep raytracer_simple.nano
- ❌ mod_player_sdl.nano → ✅ Keep mod_player.nano
- ❌ sqlite_example.nano → ✅ Keep sqlite_simple.nano

**Result:** Cleaner examples directory, no naming confusion

---

## 🎯 Phase 2: Add Iconic Examples (Next)

### Priority 1: Instant Recognition Games (10-15 hours)

#### 1. Pong (30-60 min) 🏓
**File:** `examples/sdl_pong.nano`
**Why:** First video game ever, universally recognized
**Features:**
- Two paddles (player vs player or vs AI)
- Ball physics with angle reflection
- Score display
- Simple but satisfying

**Code estimate:** ~200 lines

#### 2. Tetris (2-3 hours) 🟦
**File:** `examples/sdl_tetris.nano`
**Why:** Global phenomenon, most recognizable puzzle game
**Features:**
- 7 tetromino shapes (I, O, T, S, Z, J, L)
- Rotation, line clearing
- Increasing speed
- Score and level display
- Next piece preview

**Code estimate:** ~400-500 lines
**Challenge:** Collision detection, rotation logic

#### 3. Breakout (1-2 hours) 🧱
**File:** `examples/sdl_breakout.nano`
**Why:** Atari classic, satisfying brick-breaking physics
**Features:**
- Paddle control
- Ball physics with angle variation
- Multiple brick types/colors
- Power-ups (optional)
- Lives and score

**Code estimate:** ~300 lines

#### 4. Space Invaders (2-3 hours) 👾
**File:** `examples/sdl_space_invaders.nano`
**Why:** Arcade legend, defined the shoot-em-up genre
**Features:**
- Rows of descending aliens
- Player ship with laser
- Shields that erode
- UFO bonus targets
- Increasing difficulty

**Code estimate:** ~400 lines

#### 5. Matrix Rain (1 hour) 🖥️
**File:** `examples/ncurses_matrix_rain.nano`
**Why:** Iconic visual from The Matrix (1999)
**Features:**
- Falling green characters
- Variable speeds per column
- Fading trails
- Mesmerizing effect

**Code estimate:** ~150 lines
**Easy win:** Simple but impressive visual

---

### Priority 2: Visual Demos (5-8 hours)

#### 6. Mandelbrot Set (2 hours) 🌀
**File:** `examples/sdl_mandelbrot.nano`
**Why:** Beautiful fractal, zoom exploration is addictive
**Features:**
- Color-mapped iterations
- Mouse click to zoom
- Pan with arrow keys
- Coordinates display

**Code estimate:** ~200 lines
**Math showcase:** Complex numbers, iteration

#### 7. Fire Effect (1 hour) 🔥
**File:** `examples/sdl_fire.nano`
**Why:** Classic demoscene effect, hypnotic
**Features:**
- Pixel-based fire simulation
- Color palette animation
- Heat propagation algorithm
- Runs at 60fps

**Code estimate:** ~150 lines
**Easy win:** Simple algorithm, impressive result

#### 8. Starfield (1 hour) ⭐
**File:** `examples/sdl_starfield.nano`
**Why:** Every 80s/90s game intro, space travel feel
**Features:**
- 3D perspective projection
- Multiple star layers/speeds
- Smooth animation
- Color depth variation

**Code estimate:** ~100 lines
**Easy win:** Simple 3D math

#### 9. Maze Generator (2 hours) 🌀
**File:** `examples/ncurses_maze_generator.nano`
**Why:** Algorithmic interest, animated creation
**Features:**
- Recursive backtracker algorithm
- Animated generation (watch it grow)
- Path solving (optional)
- Different algorithms (DFS, Prim's)

**Code estimate:** ~200 lines

---

### Priority 3: Practical Utilities (3-5 hours)

#### 10. Todo List CLI (2 hours) ✔️
**File:** `examples/sqlite_todo.nano`
**Why:** Practical database example, useful tool
**Features:**
- Add/remove/list tasks
- Mark complete/incomplete
- Priority levels
- Due dates (optional)
- Persistent SQLite storage

**Code estimate:** ~300 lines
**Practical:** Real CRUD operations

#### 11. Weather Fetcher (30 min) 🌤️
**File:** `examples/curl_weather.nano`
**Why:** REST API demo, real-world data
**Features:**
- Fetch from weather API (OpenWeatherMap)
- Parse JSON response
- Display temperature, conditions
- City lookup

**Code estimate:** ~100 lines
**Requires:** curl module working

#### 12. JSON Parser Demo (1 hour) 📄
**File:** `examples/json_parser.nano`
**Why:** Modern web development essential
**Features:**
- Parse JSON from API
- Extract nested fields
- Pretty-print output
- Error handling

**Code estimate:** ~150 lines

---

## 📋 Phase 3: Build Integration (1 hour)

### Update Makefile

Add to `examples/Makefile`:

```make
# Additional SDL examples
ICONIC_GAMES = \
    $(BIN_DIR)/sdl_pong \
    $(BIN_DIR)/sdl_tetris \
    $(BIN_DIR)/sdl_breakout \
    $(BIN_DIR)/sdl_space_invaders \
    $(BIN_DIR)/sdl_mandelbrot \
    $(BIN_DIR)/sdl_fire \
    $(BIN_DIR)/sdl_starfield

NCURSES_ADDITIONS = \
    $(BIN_DIR)/ncurses_matrix_rain \
    $(BIN_DIR)/ncurses_maze_generator

UTILITY_EXAMPLES = \
    $(BIN_DIR)/sqlite_todo \
    $(BIN_DIR)/curl_weather

# Update SDL_EXAMPLES to include ICONIC_GAMES
SDL_EXAMPLES = \
    $(EXISTING_SDL_EXAMPLES) \
    $(ICONIC_GAMES)

# Build rules for each...
```

### Update Help Text

```make
@echo "Classic games:"
@echo "  ./bin/sdl_pong               - Two-player table tennis (1972) 🏓"
@echo "  ./bin/sdl_tetris             - Falling blocks puzzle (1985) 🟦"
@echo "  ./bin/sdl_breakout           - Brick-breaking arcade (1976) 🧱"
@echo "  ./bin/sdl_space_invaders     - Alien shoot-em-up (1978) 👾"
@echo ""
@echo "Visual demos:"
@echo "  ./bin/sdl_mandelbrot         - Fractal explorer 🌀"
@echo "  ./bin/sdl_fire               - Fire effect (demoscene) 🔥"
@echo "  ./bin/sdl_starfield          - 3D star field ⭐"
@echo "  ./bin/ncurses_matrix_rain    - Matrix digital rain 🖥️"
@echo ""
@echo "Utilities:"
@echo "  ./bin/sqlite_todo            - CLI todo list ✔️"
@echo "  ./bin/curl_weather           - Weather fetcher 🌤️"
```

---

## 🎯 Expected Final State

### SDL Examples (15 total)
**Current (7):**
1. sdl_checkers - Board game with AI ✅
2. sdl_boids - Flocking simulation ✅
3. sdl_particles - Explosion effects ✅
4. sdl_asteroids - Space shooter ✅
5. sdl_raytracer - Real-time ray tracing ✅
6. sdl_terrain_explorer - 3D terrain ✅
7. sdl_mod_player - MOD music player ✅

**New (8):**
8. sdl_pong - Two-player classic 🏓
9. sdl_tetris - Falling blocks 🟦
10. sdl_breakout - Brick breaker 🧱
11. sdl_space_invaders - Alien shooter 👾
12. sdl_mandelbrot - Fractal explorer 🌀
13. sdl_fire - Fire effect 🔥
14. sdl_starfield - 3D stars ⭐
15. sdl_flappy_bird - (Optional) Viral mobile game

### NCurses Examples (5 total)
**Current (2):**
1. ncurses_snake - Interactive snake ✅
2. ncurses_game_of_life - Cellular automaton ✅

**New (3):**
3. ncurses_matrix_rain - Matrix effect 🖥️
4. ncurses_maze_generator - Maze creation 🌀
5. ncurses_tetris - Terminal Tetris 🟦

### Utility Examples (5 total)
1. sqlite_simple - Database basics ✅
2. sqlite_todo - Todo list app ✔️
3. curl_weather - Weather API 🌤️
4. mod_player - Music player ✅
5. vector2d_demo - 2D math (needs build)

---

## 📊 Implementation Priority

### Week 1: Quick Wins (Immediate Gratification)
1. ✅ Cleanup duplicates (DONE)
2. 🎯 Pong (30 min) - Easiest, most iconic
3. 🎯 Matrix Rain (1 hour) - Cool visual, easy code
4. 🎯 Fire Effect (1 hour) - Demoscene classic
5. 🎯 Starfield (1 hour) - Simple 3D

**Total: ~3-4 hours, 4 new examples**

### Week 2: Major Games
1. 🎯 Breakout (1-2 hours)
2. 🎯 Tetris (2-3 hours)
3. 🎯 Space Invaders (2-3 hours)

**Total: ~5-8 hours, 3 new examples**

### Week 3: Advanced & Utilities
1. 🎯 Mandelbrot (2 hours)
2. 🎯 Maze Generator (2 hours)
3. 🎯 SQLite Todo (2 hours)
4. 🎯 Curl Weather (30 min)

**Total: ~6-7 hours, 4 new examples**

---

## 🎉 Success Metrics

**After completion:**
- ✅ 20+ well-named examples (no duplicates)
- ✅ Every major module has an example
- ✅ Mix of iconic games, visual demos, utilities
- ✅ Instant gratification for new users
- ✅ Strong pop-culture recognition
- ✅ Easy to find and run (`./bin/sdl_pong`)
- ✅ Consistent naming (`module_function` pattern)

**User Experience:**
- New users see Pong/Tetris and immediately understand the language capability
- Demoscene enthusiasts recognize fire/starfield effects
- Matrix fans love the digital rain
- Developers appreciate practical SQLite/curl examples
- Everyone can build and run examples in seconds

---

## 📝 Notes

### Code Quality Standards
- All examples must have close button support (SDL)
- Clean, readable code with comments
- Shadow tests for non-interactive parts
- No hardcoded paths
- Proper error handling

### Naming Convention
- `{module}_{name}.nano` for module examples
- `{concept}.nano` for language demos
- No version suffixes (simple/complete/v2)
- Descriptive, self-explanatory names

### Build System
- All examples in Makefile with proper dependencies
- Clear help text with year/cultural reference
- Organized by category (games, demos, utilities)
- Fast parallel builds

---

**Last Updated:** December 8, 2024  
**Status:** Phase 1 Complete ✅, Phase 2 Ready to Start 🎯
