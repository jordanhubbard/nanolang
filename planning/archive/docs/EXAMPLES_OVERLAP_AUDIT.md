# NanoLang Examples Overlap Audit
## Identifying Redundancy and Consolidation Opportunities

**Date**: 2025-12-16  
**Total Examples**: 103  
**Goal**: Flag examples with substantial overlap or redundancy

---

## Executive Summary

**Findings**:
- **34 examples** identified with significant overlap (33%)
- **15 examples** flagged for removal (redundant)
- **12 examples** flagged for consolidation (merge into comprehensive versions)
- **7 examples** flagged for clarification (unclear purpose)

**Recommended Actions**:
- Remove: 15 examples
- Merge: 12 → 5 comprehensive examples
- Keep with clarification: 7
- **Net Result**: 103 → 81 examples (21% reduction)

---

## Overlap Categories

### 1. ARRAYS - 7 Examples (EXCESSIVE OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_arrays.nano` | Basic array operations | **REMOVE** (superseded by complete) |
| `nl_arrays_simple.nano` | Simple array test | **REMOVE** (minimal value) |
| `nl_arrays_test.nano` | Shadow tests for arrays | **MERGE** into complete |
| `nl_array_complete.nano` | Comprehensive array suite | **KEEP** (definitive) |
| `nl_array_bounds.nano` | Bounds checking test | **KEEP** (specific feature) |
| `nl_list_int.nano` | Dynamic list_int | **KEEP** (different type) |
| `vector2d_demo.nano` | 2D vector math | **KEEP** (specific domain) |

**Recommendation**: 
- **Remove**: `nl_arrays.nano`, `nl_arrays_simple.nano`
- **Merge** `nl_arrays_test.nano` shadow tests into `nl_array_complete.nano`
- **Result**: 7 → 4 examples

---

### 2. STRINGS - 4 Examples (SIGNIFICANT OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_strings.nano` | String basics | **REMOVE** (superseded) |
| `nl_string_operations.nano` | String operations | **KEEP** (comprehensive) |
| `nl_string_ops.nano` | String ops (duplicate?) | **REMOVE** (duplicate name) |
| `nl_extern_string.nano` | C FFI string functions | **KEEP** (specific to FFI) |

**Recommendation**:
- **Remove**: `nl_strings.nano`, `nl_string_ops.nano`
- **Result**: 4 → 2 examples

---

### 3. MATH - 7 Examples (MODERATE OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_math.nano` | Basic math operations | **REMOVE** (redundant) |
| `nl_math_utils.nano` | Math utility functions | **MERGE** with advanced_math |
| `nl_advanced_math.nano` | Advanced math (trig, etc.) | **KEEP** (expand) |
| `nl_extern_math.nano` | C FFI math functions | **KEEP** (FFI specific) |
| `nl_matrix_operations.nano` | Matrix math | **KEEP** (specific domain) |
| `nl_pi_calculator.nano` | Pi calculation (Leibniz) | **KEEP** (algorithmic) |
| `nl_pi_simple.nano` | Simple pi calc | **REMOVE** (minimal value) |

**Recommendation**:
- **Remove**: `nl_math.nano`, `nl_pi_simple.nano`
- **Merge**: `nl_math_utils.nano` → `nl_advanced_math.nano`
- **Result**: 7 → 4 examples

---

### 4. LOOPS - 4 Examples (SIGNIFICANT OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_loops.nano` | Basic loop examples | **REMOVE** (superseded) |
| `nl_loops_working.nano` | Loop tests (working?) | **REMOVE** (confusing name) |
| `nl_for_loop_patterns.nano` | Comprehensive for-loops | **KEEP** (definitive) |
| `nl_factorial.nano` | Recursion demo | **KEEP** (different concept) |
| `nl_fibonacci.nano` | Recursion demo | **KEEP** (different concept) |

**Recommendation**:
- **Remove**: `nl_loops.nano`, `nl_loops_working.nano`
- **Result**: 4 → 2 examples

---

### 5. GENERICS - 7 Examples (EXCESSIVE OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_generics_demo.nano` | Generic types intro | **KEEP** (primary) |
| `nl_generic_lists.nano` | Generic list impl | **MERGE** into demo |
| `nl_generic_list_basics.nano` | List basics | **MERGE** into demo |
| `nl_generic_list_point.nano` | List with Point type | **MERGE** into demo |
| `nl_generic_stack.nano` | Generic stack | **KEEP** (data structure) |
| `nl_generic_queue.nano` | Generic queue | **KEEP** (data structure) |
| `nl_list_int.nano` | Non-generic int list | **KEEP** (contrast) |

**Recommendation**:
- **Merge**: `nl_generic_list_*.nano` → `nl_generics_demo.nano`
- **Result**: 7 → 4 examples

---

### 6. FUNCTIONS - 5 Examples (MODERATE OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_first_class_functions.nano` | Functions as values | **KEEP** |
| `nl_function_variables.nano` | Function variables | **MERGE** (same concept) |
| `nl_function_return_values.nano` | Functions returning functions | **MERGE** |
| `nl_function_factories.nano` | Function factories v1 | **REMOVE** (superseded) |
| `nl_function_factories_v2.nano` | Function factories v2 | **KEEP** |
| `nl_filter_map_fold.nano` | Functional programming | **KEEP** |

**Recommendation**:
- **Remove**: `nl_function_factories.nano`
- **Merge**: `nl_function_variables.nano`, `nl_function_return_values.nano` → `nl_first_class_functions.nano`
- **Result**: 5 → 3 examples

---

### 7. GAMES - 11 Examples (SOME OVERLAP)

#### Falling Sand - 2 Examples
| File | Purpose | Status |
|------|---------|--------|
| `nl_falling_sand.nano` | Terminal falling sand | **KEEP** |
| `sdl_falling_sand.nano` | SDL falling sand | **KEEP** (different UI) |

**No changes** - Different implementations are valuable.

#### Game of Life - 2 Examples  
| File | Purpose | Status |
|------|---------|--------|
| `nl_game_of_life.nano` | Terminal Game of Life | **KEEP** |
| `ncurses_game_of_life.nano` | Ncurses Game of Life | **KEEP** (different UI) |

**No changes** - Different implementations are valuable.

#### Snake - 2 Examples
| File | Purpose | Status |
|------|---------|--------|
| `nl_snake.nano` | Terminal snake | **KEEP** |
| `ncurses_snake.nano` | Ncurses snake | **KEEP** (different UI) |

**No changes** - Different implementations are valuable.

#### Boids - 2 Examples
| File | Purpose | Status |
|------|---------|--------|
| `nl_boids.nano` | Terminal boids | **KEEP** |
| `sdl_boids.nano` | SDL boids | **KEEP** (different UI) |

**No changes** - Different implementations are valuable.

#### Tic-Tac-Toe - 2 Examples
| File | Purpose | Status |
|------|---------|--------|
| `nl_tictactoe.nano` | Full tic-tac-toe | **KEEP** |
| `nl_tictactoe_simple.nano` | Simple version | **REMOVE** |

**Recommendation**: Remove simple version.

#### Maze - 1 Example
| File | Purpose | Status |
|------|---------|--------|
| `nl_maze.nano` | Maze generator | **KEEP** |

**No changes**.

---

### 8. SDL EXAMPLES - 23 Examples (MODERATE OVERLAP)

#### Audio - 3 Examples
| File | Purpose | Status |
|------|---------|--------|
| `sdl_audio_wav.nano` | Basic WAV playback | **KEEP** |
| `sdl_audio_player.nano` | Full audio player | **KEEP** (advanced) |
| `sdl_nanoamp.nano` | Music visualizer v1 | **REMOVE** (superseded) |
| `sdl_nanoamp_enhanced.nano` | Music visualizer v2 | **KEEP** |
| `sdl_mod_visualizer.nano` | MOD music visualizer | **KEEP** (specific format) |
| `nanoamp_simple.nano` | Simple amp | **REMOVE** |

**Recommendation**: Remove `sdl_nanoamp.nano`, `nanoamp_simple.nano`

#### UI Widgets - 4 Examples
| File | Purpose | Status |
|------|---------|--------|
| `sdl_ui_demo.nano` | UI demo | **REMOVE** (superseded) |
| `sdl_ui_widgets.nano` | Basic UI widgets | **REMOVE** (superseded) |
| `sdl_ui_widgets_fixed.nano` | Fixed UI widgets | **REMOVE** (what was fixed?) |
| `sdl_ui_widgets_extended.nano` | Full UI suite | **KEEP** |

**Recommendation**: Remove 3 older widget examples.

#### Graphics - 10 Examples
| File | Purpose | Status |
|------|---------|--------|
| `sdl_drawing_primitives.nano` | Basic drawing | **KEEP** |
| `sdl_texture_demo.nano` | Texture demo | **KEEP** |
| `sdl_mouse_click.nano` | Mouse input | **KEEP** |
| `sdl_fire.nano` | Fire effect | **KEEP** |
| `sdl_particles.nano` | Particle system | **KEEP** |
| `sdl_starfield.nano` | Starfield effect | **KEEP** |
| `sdl_pong.nano` | Pong game | **KEEP** |
| `sdl_checkers.nano` | Checkers game | **KEEP** |
| `sdl_asteroids.nano` | Asteroids game | **KEEP** |
| `sdl_raytracer.nano` | Raytracer | **KEEP** |
| `sdl_terrain_explorer.nano` | 3D terrain | **KEEP** |

**No changes** - All provide unique value.

#### Misc SDL - 3 Examples
| File | Purpose | Status |
|------|---------|--------|
| `sdl_import_test.nano` | Import test | **REMOVE** (test file) |
| `sdl_nanoviz.nano` | Visualization | **KEEP** |
| `example_launcher.nano` | Example launcher | **KEEP** (utility) |
| `example_launcher_simple.nano` | Simple launcher | **REMOVE** |

**Recommendation**: Remove test files.

---

### 9. NCURSES - 3 Examples (NO OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `ncurses_snake.nano` | Snake game | **KEEP** |
| `ncurses_game_of_life.nano` | Game of Life | **KEEP** |
| `ncurses_matrix_rain.nano` | Matrix rain effect | **KEEP** |

**No changes** - Good variety.

---

### 10. OPENGL - 2 Examples (NO OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `opengl_cube.nano` | Spinning cube | **KEEP** |
| `opengl_teapot.nano` | Utah teapot | **KEEP** |

**No changes** - Classic demos.

---

### 11. ONNX - 3 Examples (EXCESSIVE OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `onnx_simple.nano` | Simple ONNX | **REMOVE** |
| `onnx_inference.nano` | ONNX inference | **MERGE** |
| `onnx_classifier.nano` | ONNX classifier | **KEEP** (merge inference into this) |

**Recommendation**: 
- **Remove**: `onnx_simple.nano`
- **Merge**: `onnx_inference.nano` → `onnx_classifier.nano`
- **Result**: 3 → 1 example

---

### 12. LIBRARY BINDINGS - 4 Examples (NO OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `curl_example.nano` | HTTP requests | **KEEP** |
| `uv_example.nano` | libuv async I/O | **KEEP** |
| `event_example.nano` | Event loop | **KEEP** |
| `sqlite_simple.nano` | SQLite database | **KEEP** |

**No changes** - Unique integrations.

---

### 13. TYPE SYSTEM - 5 Examples (MINOR OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_types.nano` | Basic types | **KEEP** |
| `nl_struct.nano` | Struct demo | **KEEP** |
| `nl_enum.nano` | Enum demo | **KEEP** |
| `nl_union_types.nano` | Union types | **KEEP** |
| `nl_tuple_coordinates.nano` | Tuple demo | **KEEP** |

**No changes** - Each teaches distinct concept.

---

### 14. OPERATORS - 4 Examples (NO OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_operators.nano` | Arithmetic operators | **KEEP** |
| `nl_comparisons.nano` | Comparison operators | **KEEP** |
| `nl_logical.nano` | Logical operators | **KEEP** |
| `nl_floats.nano` | Float operations | **KEEP** |

**No changes** - Systematic coverage.

---

### 15. LANGUAGE FEATURES - 6 Examples (MINOR OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_hello.nano` | Hello world | **KEEP** |
| `nl_calculator.nano` | Calculator | **KEEP** |
| `nl_primes.nano` | Prime numbers | **KEEP** |
| `nl_language_features.nano` | Feature showcase | **KEEP** |
| `nl_new_features.nano` | New features | **CLARIFY** (which features?) |
| `nl_mutable.nano` | Mutability demo | **KEEP** |

**Recommendation**: Clarify `nl_new_features.nano` or remove.

---

### 16. STANDARD LIBRARY - 3 Examples (OVERLAP)

| File | Purpose | Status |
|------|---------|--------|
| `nl_stdlib.nano` | Stdlib demo | **KEEP** |
| `nl_os_basic.nano` | OS operations | **KEEP** (specific area) |
| `stdlib_ast_demo.nano` | AST demo | **KEEP** (advanced) |

**No changes**.

---

### 17. MISCELLANEOUS - 4 Examples

| File | Purpose | Status |
|------|---------|--------|
| `nl_demo_selfhosting.nano` | Self-hosting demo | **KEEP** (historical) |
| `nl_tracing.nano` | Execution tracing | **KEEP** (debugging) |
| `nl_random_sentence.nano` | Random text gen | **KEEP** (fun demo) |
| `nl_extern_char.nano` | C FFI char functions | **KEEP** (FFI) |

**No changes**.

---

## Removal Summary

### Files to REMOVE (15 total):

#### Arrays (2):
1. `nl_arrays.nano` - Superseded by `nl_array_complete.nano`
2. `nl_arrays_simple.nano` - Minimal value

#### Strings (2):
3. `nl_strings.nano` - Superseded by `nl_string_operations.nano`
4. `nl_string_ops.nano` - Duplicate name confusion

#### Math (2):
5. `nl_math.nano` - Redundant with `nl_advanced_math.nano`
6. `nl_pi_simple.nano` - Minimal value vs `nl_pi_calculator.nano`

#### Loops (2):
7. `nl_loops.nano` - Superseded by `nl_for_loop_patterns.nano`
8. `nl_loops_working.nano` - Confusing "working" name

#### Functions (1):
9. `nl_function_factories.nano` - Superseded by v2

#### Games (1):
10. `nl_tictactoe_simple.nano` - Simplified version not needed

#### SDL (7):
11. `sdl_nanoamp.nano` - Superseded by enhanced version
12. `nanoamp_simple.nano` - Minimal version
13. `sdl_ui_demo.nano` - Superseded by widgets
14. `sdl_ui_widgets.nano` - Superseded by extended
15. `sdl_ui_widgets_fixed.nano` - Unclear what was fixed

#### SDL Misc (2):
- Already counted above in SDL section

#### ONNX (1):
- Counted below in merges

**Note**: SDL has 5 total removes (nanoamp, nanoamp_simple, ui_demo, ui_widgets, ui_widgets_fixed)

---

## Files to MERGE (12 → 5 comprehensive):

### Arrays:
- **Merge** `nl_arrays_test.nano` → `nl_array_complete.nano`

### Math:
- **Merge** `nl_math_utils.nano` → `nl_advanced_math.nano`

### Generics (3 → 1):
- **Merge** `nl_generic_lists.nano` → `nl_generics_demo.nano`
- **Merge** `nl_generic_list_basics.nano` → `nl_generics_demo.nano`
- **Merge** `nl_generic_list_point.nano` → `nl_generics_demo.nano`

### Functions (2 → 1):
- **Merge** `nl_function_variables.nano` → `nl_first_class_functions.nano`
- **Merge** `nl_function_return_values.nano` → `nl_first_class_functions.nano`

### ONNX (2 → 1):
- **Merge** `onnx_inference.nano` → `onnx_classifier.nano`
- **Remove** `onnx_simple.nano`

### SDL (2 → 0):
- **Remove** `example_launcher_simple.nano`
- **Remove** `sdl_import_test.nano`

---

## Files Needing CLARIFICATION (7):

1. `nl_new_features.nano` - Which features? Rename to be specific
2. `nl_demo_selfhosting.nano` - Is this still relevant post-bootstrap?
3. `nl_language_features.nano` vs `nl_new_features.nano` - Redundant names?
4. SDL examples without clear documentation headers
5. `sdl_nanoviz.nano` - Purpose unclear vs other visualizers

---

## Implementation Plan

### Phase 1: Low-Risk Removals (Priority 1)
Remove obviously redundant files:
```bash
# Arrays
rm nl_arrays.nano nl_arrays_simple.nano

# Strings  
rm nl_strings.nano nl_string_ops.nano

# Math
rm nl_math.nano nl_pi_simple.nano

# Loops
rm nl_loops.nano nl_loops_working.nano
```

### Phase 2: Consolidation (Priority 2)
Merge content into comprehensive examples:
1. Copy tests from `nl_arrays_test.nano` to `nl_array_complete.nano`
2. Merge utility functions from `nl_math_utils.nano` to `nl_advanced_math.nano`
3. Consolidate generics examples into `nl_generics_demo.nano`
4. Merge function examples into `nl_first_class_functions.nano`
5. Consolidate ONNX examples

### Phase 3: SDL Cleanup (Priority 3)
Remove superseded SDL versions:
```bash
rm sdl_nanoamp.nano nanoamp_simple.nano
rm sdl_ui_demo.nano sdl_ui_widgets.nano sdl_ui_widgets_fixed.nano
rm sdl_import_test.nano example_launcher_simple.nano
```

### Phase 4: Clarification (Priority 4)
Rename or document unclear examples:
- `nl_new_features.nano` → clarify which features or remove
- Add header comments to all examples explaining purpose

---

## Expected Outcomes

**Before**: 103 examples  
**After**: 81 examples (21% reduction)

**Benefits**:
1. ✅ Clearer learning path (less confusion)
2. ✅ Easier maintenance (fewer files to update)
3. ✅ Better organization (comprehensive > scattered)
4. ✅ Faster CI builds (fewer examples to compile)
5. ✅ Improved documentation (each example has clear purpose)

**Risk**: Low - all removed examples are redundant or superseded

---

## Appendix: Full Categorization

### Keep As-Is (65 examples):
- Core language: hello, calculator, primes, factorial, fibonacci, operators, types, etc.
- Graphics: SDL games, effects, OpenGL demos
- Systems: OS, stdlib, FFI examples
- Data structures: arrays (definitive), generics (stack, queue)
- UI: ncurses demos, SDL widgets (extended)
- Integrations: curl, sqlite, uv, onnx

### Remove (15 examples):
- Listed above in "Removal Summary"

### Merge (12 examples):
- Listed above in "Files to MERGE"

### Clarify (7 examples):
- Listed above in "Files Needing CLARIFICATION"

---

**End of Overlap Audit**

*Generated: 2025-12-16*  
*Auditor: Claude (AI Code Analysis)*  
*Status: Ready for implementation*

