# Examples Consolidation Audit

## Overview
Total examples: ~108 .nano files
Goal: Reduce duplication, merge similar examples, improve discoverability

## Category 1: BASIC LANGUAGE FEATURES (Merge into fewer examples)

### Arrays (5 files → 1 comprehensive example)
- `nl_arrays.nano` - Basic array operations
- `nl_arrays_simple.nano` - Simplified arrays
- `nl_array_complete.nano` - Complete array features
- `nl_array_bounds.nano` - Bounds checking
- `nl_arrays_test.nano` - Array tests
**→ MERGE INTO**: `nl_arrays_complete.nano` (comprehensive array showcase)

### Loops (3 files → 1 example)
- `nl_loops.nano` - Basic loops
- `nl_loops_working.nano` - Working loops
- `nl_for_loop_patterns.nano` - For loop patterns
**→ MERGE INTO**: `nl_loops_complete.nano`

### Math Operations (4 files → 1 example)
- `nl_math.nano` - Math functions
- `nl_math_utils.nano` - Math utilities
- `nl_advanced_math.nano` - Advanced math
- `nl_extern_math.nano` - External C math functions
**→ MERGE INTO**: `nl_math_complete.nano`

### String Operations (4 files → 1 example)
- `nl_strings.nano` - Basic strings
- `nl_string_operations.nano` - String operations
- `nl_string_ops.nano` - String ops (duplicate?)
- `nl_extern_string.nano` - External C string functions
**→ MERGE INTO**: `nl_strings_complete.nano`

### Functions (5 files → 1 example)
- `nl_first_class_functions.nano` - First-class functions
- `nl_function_factories.nano` - Function factories
- `nl_function_factories_v2.nano` - Function factories v2
- `nl_function_return_values.nano` - Return values
- `nl_function_variables.nano` - Function variables
**→ MERGE INTO**: `nl_functions_complete.nano`

### Generics/Lists (7 files → 2 examples: basics + advanced)
- `nl_generic_lists.nano` - Generic lists
- `nl_generic_list_basics.nano` - List basics
- `nl_generic_list_point.nano` - List with points
- `nl_list_int.nano` - Integer lists
- `nl_generic_stack.nano` - Generic stack
- `nl_generic_queue.nano` - Generic queue
- `nl_generics_demo.nano` - Generics demo
**→ MERGE INTO**: 
  - `nl_generics_basics.nano` (lists, basic usage)
  - `nl_generics_structures.nano` (stack, queue, advanced)

### Simple Math Programs (4 files → 1 example)
- `nl_factorial.nano` - Factorial
- `nl_fibonacci.nano` - Fibonacci
- `nl_primes.nano` - Prime numbers
- `nl_pi_simple.nano` - Simple pi calculation
- `nl_pi_calculator.nano` - Pi calculator
**→ MERGE INTO**: `nl_algorithms.nano` (comprehensive algorithms showcase)

### Basic Language Demos (Keep separate but minimal)
- `nl_hello.nano` - Keep as minimal example
- `nl_operators.nano` - Keep
- `nl_comparisons.nano` - Keep
- `nl_logical.nano` - Keep
- `nl_enum.nano` - Keep
- `nl_struct.nano` - Keep
- `nl_types.nano` - Keep
- `nl_floats.nano` - Merge into nl_types
- `nl_mutable.nano` - Keep
- `nl_union_types.nano` - Keep
- `nl_calculator.nano` - Keep (good simple example)

## Category 2: GAMES (Consolidate versions)

### Snake (3 files → 1 comprehensive example)
- `nl_snake.nano` - Text-based snake
- `ncurses_snake.nano` - NCurses snake
- (Both could stay as separate terminal vs ncurses demos)
**→ KEEP**: Both (different UI approaches)

### Game of Life (3 files → 2 examples)
- `nl_game_of_life.nano` - Pure computation
- `ncurses_game_of_life.nano` - NCurses version
**→ KEEP**: Both (different UI approaches)

### Tic-Tac-Toe (2 files → 1 example)
- `nl_tictactoe.nano` - Full version
- `nl_tictactoe_simple.nano` - Simplified version
**→ MERGE INTO**: `nl_tictactoe.nano` (keep full version only)

### Falling Sand (2 files → 2 examples)
- `nl_falling_sand.nano` - Text-based
- `sdl_falling_sand.nano` - SDL graphical version
**→ KEEP**: Both (very different implementations)

### Boids (2 files → 2 examples)
- `nl_boids.nano` - Pure computation
- `sdl_boids.nano` - SDL visual
**→ KEEP**: Both (computational vs visual demo)

### Other Games (Keep all - unique)
- `sdl_pong.nano` - Classic Pong
- `sdl_asteroids.nano` - Space shooter
- `sdl_checkers.nano` - Checkers with AI
- `nl_maze.nano` - Maze generator/solver

## Category 3: GRAPHICS DEMOS (Consolidate UI widgets)

### SDL UI Widgets (4 files → 1 comprehensive example)
- `sdl_ui_widgets.nano` - Basic widgets
- `sdl_ui_widgets_extended.nano` - Extended widgets
- `sdl_ui_widgets_fixed.nano` - Fixed version
- `sdl_ui_demo.nano` - Demo with file browser
**→ MERGE INTO**: `sdl_ui_showcase.nano` (comprehensive UI demo with file browser)

### Audio/Music Players (3 files → 2 examples)
- `sdl_nanoamp.nano` - NanoAmp with directory browser
- `sdl_nanoamp_enhanced.nano` - Enhanced Winamp tribute
- `nanoamp_simple.nano` - Simple version
**→ MERGE INTO**: 
  - `sdl_audio_simple.nano` (basic audio playback)
  - `sdl_nanoamp.nano` (full-featured player, keep enhanced version)

### Audio Examples (3 files → 1 example)
- `sdl_audio_player.nano` - Audio player
- `sdl_audio_wav.nano` - WAV playback test
- `sdl_mod_visualizer.nano` - MOD player with visualizer
**→ KEEP**: `sdl_mod_visualizer.nano` and `sdl_audio_simple.nano`

### Graphics Demos (Keep all - unique visuals)
- `sdl_fire.nano` - Fire effect
- `sdl_starfield.nano` - Starfield effect
- `sdl_particles.nano` - Particle system
- `sdl_raytracer.nano` - Ray tracer
- `sdl_terrain_explorer.nano` - Terrain generation
- `sdl_nanoviz.nano` - 3D music visualizer
- `sdl_drawing_primitives.nano` - Drawing primitives
- `sdl_texture_demo.nano` - Texture demo
- `sdl_mouse_click.nano` - Mouse input demo
- `ncurses_matrix_rain.nano` - Matrix effect

### OpenGL (2 files → Keep both)
- `opengl_cube.nano` - Rotating cube
- `opengl_teapot.nano` - Utah teapot with shaders
**→ KEEP**: Both (different complexity levels)

## Category 4: MODULES/LIBRARIES (Keep all - different modules)

### ONNX (3 files → 1 example)
- `onnx_simple.nano` - Simple ONNX test
- `onnx_inference.nano` - Inference example
- `onnx_classifier.nano` - Image classifier
**→ MERGE INTO**: `onnx_classifier.nano` (keep most comprehensive)

### External Libraries (Keep separate)
- `sqlite_simple.nano` - SQLite example
- `curl_example.nano` - HTTP requests
- `event_example.nano` - libevent
- `uv_example.nano` - libuv
- `sdl_import_test.nano` - DELETE (just a test)

## Category 5: ADVANCED FEATURES (Keep all)

- `nl_language_features.nano` - Language features showcase
- `nl_demo_selfhosting.nano` - Self-hosting demo
- `nl_filter_map_fold.nano` - Functional programming
- `nl_tuple_coordinates.nano` - Tuple usage
- `nl_matrix_operations.nano` - Matrix operations
- `nl_random_sentence.nano` - Random sentence generator
- `nl_tracing.nano` - Tracing functionality
- `nl_os_basic.nano` - OS functions
- `nl_stdlib.nano` - Standard library
- `nl_new_features.nano` - New features demo
- `stdlib_ast_demo.nano` - AST demo
- `vector2d_demo.nano` - 2D vectors

## CONSOLIDATION SUMMARY

### DELETE (11 files)
1. `nl_arrays_simple.nano` (merge into nl_arrays_complete)
2. `nl_array_bounds.nano` (merge into nl_arrays_complete)
3. `nl_arrays_test.nano` (merge into nl_arrays_complete)
4. `nl_loops_working.nano` (merge into nl_loops)
5. `nl_math_utils.nano` (merge into nl_math)
6. `nl_string_ops.nano` (duplicate of nl_string_operations)
7. `nl_function_factories_v2.nano` (merge into nl_function_factories)
8. `nl_tictactoe_simple.nano` (keep full version only)
9. `sdl_ui_widgets.nano` (superseded by extended)
10. `sdl_ui_widgets_fixed.nano` (superseded by demo)
11. `sdl_import_test.nano` (test file, not example)
12. `nanoamp_simple.nano` (superseded by enhanced)
13. `sdl_audio_wav.nano` (merge into audio_simple)
14. `onnx_simple.nano` (merge into classifier)
15. `onnx_inference.nano` (merge into classifier)

### MERGE (Create 8 new comprehensive examples)
1. **nl_arrays_complete.nano** ← nl_arrays + nl_array_complete + nl_arrays_simple + nl_array_bounds
2. **nl_loops_complete.nano** ← nl_loops + nl_loops_working + nl_for_loop_patterns  
3. **nl_math_complete.nano** ← nl_math + nl_advanced_math + nl_extern_math
4. **nl_strings_complete.nano** ← nl_strings + nl_string_operations + nl_extern_string
5. **nl_functions_complete.nano** ← All 5 function examples
6. **nl_generics_basics.nano** ← Basic list examples
7. **nl_generics_structures.nano** ← Stack, queue, advanced
8. **nl_algorithms.nano** ← factorial, fibonacci, primes, pi

9. **sdl_ui_showcase.nano** ← sdl_ui_widgets_extended + sdl_ui_demo
10. **sdl_audio_simple.nano** ← sdl_audio_player + sdl_audio_wav
11. **onnx_complete.nano** ← All 3 ONNX examples

### RESULT
- **Before**: 108 examples
- **After**: ~75 examples (-33 files, ~30% reduction)
- **Benefit**: Easier to discover, less maintenance, more comprehensive examples
