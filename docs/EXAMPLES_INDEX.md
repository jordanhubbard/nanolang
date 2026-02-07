# NanoLang Examples Index
## Complete Guide to All 79 Examples

**Last Updated**: 2026-01-23
**Total Examples**: 79  
**Organization**: By difficulty and topic

---

## Quick Start - Learning Paths

### 🌱 Beginner Path (Start Here!)
Follow this sequence to learn NanoLang basics:
1. `nl_hello.nano` - Hello World, basic syntax
2. `nl_calculator.nano` - Functions and prefix/infix notation
3. `nl_operators.nano` - Arithmetic operators
4. `nl_comparisons.nano` - Comparison operators
5. `nl_types.nano` - Type system basics
6. `nl_mutable.nano` - Immutable vs mutable
7. `nl_array_complete.nano` - Arrays and collections
8. `nl_struct.nano` - User-defined types
9. `nl_factorial.nano` - Recursion
10. `nl_fibonacci.nano` - Iteration and recursion
11. `namespace_demo.nano` - Module system and visibility ⭐ NEW

**Estimated Time**: 4-7 hours

---

### 🎮 Graphics & Games Path
For visual applications and game development:
1. `sdl_drawing_primitives.nano` - SDL basics
2. `sdl_mouse_click.nano` - Input handling
3. `sdl_particles.nano` - Particle systems
4. `sdl_pong.nano` - Complete game
5. `sdl_asteroids.nano` - Advanced game ⭐ SHOWCASE
6. `sdl_terrain_explorer.nano` - 3D graphics ⭐ SHOWCASE
7. `opengl_cube.nano` - OpenGL 3D
8. `sdl_raytracer.nano` - Ray tracing

**Estimated Time**: 8-12 hours

---

### 🔌 C FFI & Integration Path
For calling C libraries and external integrations:
1. `nl_extern_math.nano` - C math functions
2. `nl_extern_string.nano` - C string functions
3. `curl_example.nano` - HTTP requests with libcurl
4. `sqlite_simple.nano` - Database operations
5. `uv_example.nano` - Async I/O with libuv
6. `event_example.nano` - Event loops with libevent

**Estimated Time**: 6-8 hours

---

### 🧠 Advanced Features Path
For advanced language features and metaprogramming:
1. `nl_generics_demo.nano` - Generic types
2. `nl_first_class_functions.nano` - Higher-order functions
3. `nl_filter_map_fold.nano` - Functional programming
4. `stdlib_ast_demo.nano` - AST manipulation ⭐ SHOWCASE
5. `nl_demo_selfhosting.nano` - Self-hosting demo
6. `nl_tracing.nano` - Execution tracing

**Estimated Time**: 8-10 hours

---

## Complete Examples Catalog

### 1. LANGUAGE BASICS (15 examples)

#### Core Syntax
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_hello.nano` | ⭐ Beginner | Hello World, print | 5 min |
| `nl_calculator.nano` | ⭐ Beginner | Functions, arithmetic | 10 min |
| `nl_operators.nano` | ⭐ Beginner | +, -, *, /, % | 10 min |
| `nl_comparisons.nano` | ⭐ Beginner | ==, !=, <, >, etc | 10 min |
| `nl_logical.nano` | ⭐ Beginner | and, or, not | 10 min |
| `nl_floats.nano` | ⭐ Beginner | Float arithmetic | 10 min |

#### Types & Variables
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_types.nano` | ⭐ Beginner | Type system | 15 min |
| `nl_mutable.nano` | ⭐ Beginner | let vs let mut | 15 min |

#### Control Flow
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_factorial.nano` | ⭐⭐ Intermediate | Recursion | 15 min |
| `nl_fibonacci.nano` | ⭐⭐ Intermediate | Recursion, memoization | 20 min |
| `nl_primes.nano` | ⭐⭐ Intermediate | Algorithms | 20 min |
| `nl_pi_chudnovsky.nano` | ⭐⭐⭐ Advanced | Machin's formula, timing, verification | 30 min |

#### Language Features
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_random_sentence.nano` | ⭐ Beginner | RNG, strings | 15 min |

---

### 2. DATA STRUCTURES (12 examples)

#### Arrays
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_array_complete.nano` | ⭐⭐ Intermediate | Arrays (comprehensive) | 30 min |
| `nl_array_bounds.nano` | ⭐⭐ Intermediate | Bounds checking | 15 min |
| `vector2d_demo.nano` | ⭐⭐ Intermediate | 2D vectors | 20 min |

#### User-Defined Types
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_struct.nano` | ⭐⭐ Intermediate | Structs | 20 min |
| `nl_enum.nano` | ⭐⭐ Intermediate | Enumerations | 20 min |
| `nl_union_types.nano` | ⭐⭐⭐ Advanced | Tagged unions | 30 min |

#### Generics
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_generics_demo.nano` | ⭐⭐⭐ Advanced | Generic List<T> | 40 min |

---

### 3. FUNCTIONS & FUNCTIONAL PROGRAMMING (4 examples)

| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_first_class_functions.nano` | ⭐⭐⭐ Advanced | Functions as values | 30 min |
| `nl_function_factories_v2.nano` | ⭐⭐⭐ Advanced | Closures, factories | 30 min |
| `nl_filter_map_fold.nano` | ⭐⭐⭐ Advanced | FP patterns | 40 min |
| `nl_function_variables.nano` | ⭐⭐ Intermediate | Function references | 20 min |

---

### 4. STRINGS (2 examples)

| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_string_operations.nano` | ⭐⭐ Intermediate | String manipulation | 25 min |
| `nl_extern_string.nano` | ⭐⭐⭐ Advanced | C FFI strings | 25 min |

---

### 5. MATH (5 examples)

| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_advanced_math.nano` | ⭐⭐ Intermediate | Trig, logarithms | 25 min |
| `nl_extern_math.nano` | ⭐⭐ Intermediate | C math FFI | 20 min |
| `nl_extern_char.nano` | ⭐⭐ Intermediate | C char FFI | 15 min |
| `nl_matrix_operations.nano` | ⭐⭐⭐⭐⭐ Expert | Linear algebra ⭐ SHOWCASE | 60 min |
| `nl_pi_calculator.nano` | ⭐⭐ Intermediate | Pi calculation | 20 min |

---

### 6. STANDARD LIBRARY (2 examples)

| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `stdlib_ast_demo.nano` | ⭐⭐⭐⭐⭐ Expert | AST manipulation ⭐ SHOWCASE | 60 min |
| `nl_tracing.nano` | ⭐⭐ Intermediate | Execution tracing | 25 min |

---

### 7. EXTERNAL LIBRARIES (7 examples)

#### HTTP & Networking
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `curl_example.nano` | ⭐⭐⭐ Advanced | HTTP with libcurl | 30 min |
| `uv_example.nano` | ⭐⭐⭐⭐ Expert | Async I/O (libuv) | 40 min |
| `event_example.nano` | ⭐⭐⭐⭐ Expert | Event loops (libevent) | 40 min |

#### Database
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sqlite_simple.nano` | ⭐⭐⭐ Advanced | SQLite database | 30 min |

#### Machine Learning
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `onnx_classifier.nano` | ⭐⭐⭐⭐ Expert | ML inference (ONNX) | 45 min |
| `onnx_inference.nano` | ⭐⭐⭐ Advanced | ONNX basics | 30 min |
| `onnx_simple.nano` | ⭐⭐ Intermediate | Simple ONNX | 20 min |

---

### 8. GAMES (8 examples)

#### Cellular Automata
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_game_of_life.nano` | ⭐⭐⭐ Advanced | Conway's Life | 40 min |
| `nl_falling_sand.nano` | ⭐⭐⭐ Advanced | Particle physics | 40 min |

#### Classic Games
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_snake.nano` | ⭐⭐ Intermediate | Snake game | 30 min |
| `nl_tictactoe.nano` | ⭐⭐ Intermediate | Tic-tac-toe | 30 min |
| `nl_maze.nano` | ⭐⭐⭐ Advanced | Maze generation | 35 min |

#### AI & Simulation
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_boids.nano` | ⭐⭐⭐⭐ Expert | Flocking AI | 50 min |

#### Ncurses Games
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `ncurses_snake.nano` | ⭐⭐⭐ Advanced | Terminal snake | 35 min |
| `ncurses_game_of_life.nano` | ⭐⭐⭐ Advanced | Terminal Life | 35 min |

---

### 9. SDL GRAPHICS & GAMES (22 examples)

#### SDL Basics
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_drawing_primitives.nano` | ⭐⭐ Intermediate | Drawing basics | 25 min |
| `sdl_texture_demo.nano` | ⭐⭐ Intermediate | Textures | 25 min |
| `sdl_mouse_click.nano` | ⭐⭐ Intermediate | Mouse input | 20 min |

#### SDL Audio
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_audio_wav.nano` | ⭐⭐⭐ Advanced | WAV playback | 30 min |
| `sdl_audio_player.nano` | ⭐⭐⭐⭐ Expert | Full audio player | 60 min |
| `sdl_nanoamp.nano` | ⭐⭐⭐⭐⭐ Expert | Music visualizer ⭐ SHOWCASE | 90 min |
| `sdl_mod_visualizer.nano` | ⭐⭐⭐⭐ Expert | MOD player | 50 min |

#### SDL UI
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_ui_widgets_extended.nano` | ⭐⭐⭐⭐ Expert | Complete UI suite | 60 min |

#### SDL Visual Effects
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_fire.nano` | ⭐⭐⭐ Advanced | Fire effect | 35 min |
| `sdl_particles.nano` | ⭐⭐⭐⭐⭐ Expert | Particle system ⭐ SHOWCASE | 50 min |
| `sdl_starfield.nano` | ⭐⭐⭐ Advanced | Starfield | 30 min |
| `ncurses_matrix_rain.nano` | ⭐⭐⭐ Advanced | Matrix rain | 30 min |

#### SDL Physics & Simulation
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_falling_sand.nano` | ⭐⭐⭐⭐ Expert | Falling sand physics | 50 min |
| `sdl_boids.nano` | ⭐⭐⭐⭐⭐ Expert | Flocking AI ⭐ SHOWCASE | 60 min |

#### SDL Games
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_pong.nano` | ⭐⭐⭐ Advanced | Pong game | 40 min |
| `sdl_checkers.nano` | ⭐⭐⭐⭐ Expert | Checkers + AI | 60 min |
| `sdl_asteroids.nano` | ⭐⭐⭐⭐⭐ Expert | Complete game ⭐ SHOWCASE | 90 min |

#### SDL Advanced Graphics
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_raytracer.nano` | ⭐⭐⭐⭐⭐ Expert | Ray tracing | 90 min |
| `sdl_terrain_explorer.nano` | ⭐⭐⭐⭐⭐ Expert | 3D terrain ⭐ SHOWCASE | 90 min |

#### SDL Integration
| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `sdl_nanoviz.nano` | ⭐⭐⭐⭐⭐ Expert | 3D music visualizer | 90 min |
| `sdl_example_launcher.nano` | ⭐⭐ Intermediate | Example browser/launcher (SDL UI) | 20 min |

---

### 10. OPENGL (2 examples)

| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `opengl_cube.nano` | ⭐⭐⭐⭐ Expert | 3D cube | 45 min |
| `opengl_teapot.nano` | ⭐⭐⭐⭐ Expert | Utah teapot | 45 min |

---

### 11. ADVANCED FEATURES (1 example)

| Example | Difficulty | Topics | Time |
|---------|------------|--------|------|
| `nl_demo_selfhosting.nano` | ⭐⭐⭐⭐⭐ Expert | Self-hosting demo | 60 min |

---

## ⭐ Showcase Applications

These 6 examples represent the best of NanoLang - production-quality applications demonstrating multiple features:

1. **SDL Asteroids** (`sdl_asteroids.nano`) - Complete arcade game
   - Topics: Game loop, physics, collision, entities, state management
   - Time: 90+ minutes
   - Why showcase: Complete, polished, production-ready

2. **SDL Terrain Explorer** (`sdl_terrain_explorer.nano`) - 3D graphics
   - Topics: 3D math, LOD rendering, Perlin noise, camera controls
   - Time: 90+ minutes
   - Why showcase: Advanced graphics, performance optimization

3. **SDL Boids** (`sdl_boids.nano`) - Flocking AI simulation
   - Topics: AI, spatial hashing, emergent behavior, 1000+ entities
   - Time: 60+ minutes
   - Why showcase: Sophisticated algorithms, excellent performance

4. **SDL NanoAmp** (`sdl_nanoamp.nano`) - Music visualizer
   - Topics: Audio, FFT, DSP, real-time visualization
   - Time: 90+ minutes
   - Why showcase: Audio processing, beautiful visualization

5. **Matrix Operations** (`nl_matrix_operations.nano`) - Linear algebra
   - Topics: Generics, performance, algorithms, comprehensive tests
   - Time: 60+ minutes
   - Why showcase: Production library quality

6. **Stdlib AST Demo** (`stdlib_ast_demo.nano`) - Metaprogramming
   - Topics: AST manipulation, compiler internals, code generation
   - Time: 60+ minutes
   - Why showcase: Unique NanoLang feature, advanced

See `docs/SHOWCASE_APPLICATIONS.md` for detailed analysis.

---

## Difficulty Ratings Explained

- ⭐ **Beginner** (0-15 min): Basic syntax, no prerequisites
- ⭐⭐ **Intermediate** (15-30 min): Requires basic knowledge
- ⭐⭐⭐ **Advanced** (30-50 min): Complex concepts, multiple features
- ⭐⭐⭐⭐ **Expert** (50-90 min): Production-quality, sophisticated
- ⭐⭐⭐⭐⭐ **Showcase** (90+ min): Best-in-class, comprehensive

---

## Topic Index

Find examples by topic:

**Language Core**: hello, calculator, operators, types, mutable, factorial, fibonacci

**Data Structures**: arrays, structs, enums, unions, generics

**Functions**: first-class, factories (no captured closures), map/reduce

**Strings**: operations, extern string

**Math**: advanced math, matrix operations, pi calculator, extern math

**I/O & OS**: file operations, paths

**FFI & External**: curl, sqlite, uv, event, onnx

**Games**: game of life, snake, maze, tic-tac-toe, boids

**SDL Graphics**: primitives, textures, particles, effects

**SDL Games**: pong, checkers, asteroids

**SDL Audio**: wav player, audio player, visualizers

**OpenGL**: 3D cube, teapot

**Advanced**: generics, AST, self-hosting, tracing

---

## Prerequisites Map

```
LEVEL 1 (No Prerequisites):
└─ nl_hello.nano

LEVEL 2 (Hello World only):
├─ nl_calculator.nano
├─ nl_operators.nano
└─ nl_types.nano

LEVEL 3 (Basics + Types):
├─ nl_mutable.nano
├─ nl_array_complete.nano
└─ nl_struct.nano

LEVEL 4 (Data Structures):
├─ nl_generics_demo.nano
├─ nl_first_class_functions.nano
└─ SDL basics (primitives, mouse)

LEVEL 5 (Advanced Features):
├─ SDL games (pong, asteroids)
├─ stdlib_ast_demo.nano
└─ Matrix operations

SHOWCASE LEVEL:
└─ All showcase applications
```

---

## Total Time Estimates

- **Beginner Path**: 4-7 hours
- **Graphics Path**: 8-12 hours
- **FFI Path**: 6-8 hours
- **Advanced Path**: 8-10 hours
- **All Examples**: 60-80 hours
- **Showcase Only**: 8-10 hours

---

## Next Steps

1. **Start with Beginner Path** - Learn fundamentals
2. **Choose your interest** - Graphics, FFI, or Advanced
3. **Build something** - Apply what you learned
4. **Study Showcases** - See production-quality code

For detailed analysis, see:
- `docs/EXAMPLES_OVERLAP_AUDIT.md` - Redundancy analysis
- `docs/EXAMPLES_INSTRUCTIONAL_REVIEW.md` - Teaching focus
- `docs/REALWORLD_EXAMPLES_EVALUATION.md` - Production quality
- `docs/SHOWCASE_APPLICATIONS.md` - Flagship applications

---

**Happy Learning!** 🚀
