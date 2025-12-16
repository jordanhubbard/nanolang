# NanoLang Examples Index

> **Total Examples**: 85  
> **Last Updated**: 2025-12-16  
> **Purpose**: Complete catalog with difficulty ratings, prerequisites, and learning paths

---

## 📚 Quick Navigation

- [By Difficulty](#by-difficulty)
- [By Category](#by-category)
- [Learning Paths](#learning-paths)
- [Complete Alphabetical Index](#complete-alphabetical-index)

---

## By Difficulty

### 🟢 Beginner (1-2 weeks of programming experience)

Essential basics for getting started with NanoLang.

| Example | Topics | Description |
|---------|--------|-------------|
| `nl_hello.nano` | Output | Classic "Hello, World!" program |
| `nl_calculator.nano` | Arithmetic, functions | Basic calculator with operators |
| `nl_factorial.nano` | Recursion | Compute factorials |
| `nl_types.nano` | Type system | Primitive types demonstration |
| `nl_variables.nano` | Variables | Let bindings and scope |
| `nl_comparisons.nano` | Boolean logic | Comparison operators |
| `nl_conditionals.nano` | Control flow | If/else statements |
| `nl_new_features.nano` | Unary operators | Negation and constants |

**Prerequisites**: None  
**Next Steps**: [Core Language Path](#core-language-path)

### 🟡 Intermediate (Comfortable with basics)

Building on fundamentals with more complex features.

| Example | Topics | Description |
|---------|--------|-------------|
| `nl_loops.nano` | Iteration | While and for loops |
| `nl_arrays.nano` | Data structures | Array operations |
| `nl_strings.nano` | String manipulation | String functions |
| `nl_struct.nano` | User types | Struct definition and usage |
| `nl_first_class_functions.nano` | Functions | Higher-order functions |
| `nl_mutability.nano` | Memory model | Mutable vs immutable |
| `nl_shadow_testing.nano` | Testing | Shadow test patterns |
| `sqlite_simple.nano` | FFI, Database | SQLite integration |

**Prerequisites**: Beginner examples  
**Next Steps**: [FFI Path](#ffi-path) or [Data Structures Path](#data-structures-path)

### 🔴 Advanced (Deep language knowledge)

Complex features requiring understanding of multiple concepts.

| Example | Topics | Description |
|---------|--------|-------------|
| `nl_generics_demo.nano` | Generics, Monomorphization | Comprehensive List<T> demo |
| `nl_generic_stack.nano` | Generics, Data structures | Generic stack implementation |
| `nl_generic_queue.nano` | Generics, Data structures | Generic queue implementation |
| `stdlib_ast_demo.nano` | Metaprogramming | AST manipulation |
| `nl_demo_selfhosting.nano` | Compiler | Self-hosting demonstration |
| `nl_tracing.nano` | Debugging | Execution tracing |

**Prerequisites**: Intermediate examples + generics understanding  
**Next Steps**: [Metaprogramming Path](#metaprogramming-path)

---

## By Category

### 🎯 Core Language (21 examples)

Fundamental language features and syntax.

**Beginner:**
- `nl_hello.nano` - Hello world
- `nl_calculator.nano` - Basic arithmetic
- `nl_types.nano` - Type system
- `nl_variables.nano` - Variables and scope
- `nl_comparisons.nano` - Comparison operators
- `nl_conditionals.nano` - If/else statements
- `nl_factorial.nano` - Recursion basics
- `nl_new_features.nano` - Unary operators

**Intermediate:**
- `nl_loops.nano` - Iteration
- `nl_strings.nano` - String manipulation
- `nl_mutability.nano` - Mutable state
- `nl_shadow_testing.nano` - Testing patterns
- `nl_struct.nano` - User-defined types
- `nl_enum.nano` - Enumerations
- `nl_first_class_functions.nano` - Higher-order functions

**Advanced:**
- `nl_advanced_math.nano` - Complex math operations
- `nl_opaque_types.nano` - Opaque type definitions
- `nl_lifetimes.nano` - Memory lifetime management

### 🔌 Foreign Function Interface (14 examples)

Interoperability with C libraries and system APIs.

**Beginner:**
- `nl_extern_math.nano` - Calling C math functions
- `nl_extern_string.nano` - String FFI
- `nl_extern_char.nano` - Character functions

**Intermediate:**
- `curl_example.nano` - HTTP requests with libcurl
- `sqlite_simple.nano` - Database operations
- `uv_example.nano` - Async I/O with libuv
- `event_example.nano` - Event loop integration

**Advanced:**
- `nl_extern_arrays.nano` - Array marshalling
- `nl_extern_malloc.nano` - Manual memory management
- `nl_extern_structs.nano` - Struct FFI patterns

### 🎮 Graphics & UI (25 examples)

SDL-based graphics, games, and user interfaces.

**Beginner:**
- `sdl_primitives.nano` - Basic shapes
- `sdl_mouse.nano` - Mouse input
- `sdl_keyboard.nano` - Keyboard input
- `sdl_animation.nano` - Simple animation

**Intermediate:**
- `sdl_particles.nano` - Particle systems
- `sdl_pong.nano` - Classic Pong game
- `sdl_asteroids.nano` - Asteroids game ⭐
- `sdl_breakout.nano` - Breakout clone
- `sdl_ui_widgets_extended.nano` - Comprehensive UI widgets
- `sdl_nanoviz.nano` - Data visualization

**Advanced:**
- `sdl_nanoamp_enhanced.nano` - Audio player with UI ⭐
- `sdl_terrain.nano` - Terrain generation ⭐
- `sdl_boids.nano` - Flocking simulation ⭐

**Terminal UI:**
- `ncurses_game_of_life.nano` - Conway's Game of Life
- `ncurses_matrix_rain.nano` - Matrix-style animation
- `ncurses_snake.nano` - Snake game

⭐ = Showcase applications (see [Showcase Applications](#showcase-applications))

### 📊 Data Structures & Algorithms (12 examples)

Collections, sorting, searching, and algorithms.

**Beginner:**
- `nl_arrays.nano` - Array basics
- `nl_array_bounds.nano` - Bounds checking

**Intermediate:**
- `nl_array_complete.nano` - Comprehensive array operations
- `nl_sorting.nano` - Sorting algorithms
- `nl_searching.nano` - Search algorithms
- `nl_matrix_ops.nano` - Matrix operations ⭐

**Advanced:**
- `nl_generics_demo.nano` - Generic List<T> ⭐
- `nl_generic_stack.nano` - Generic stack
- `nl_generic_queue.nano` - Generic queue
- `nl_boids.nano` - Boids algorithm ⭐

### 🧪 Testing & Quality (6 examples)

Testing patterns, debugging, and code quality.

- `nl_shadow_testing.nano` - Shadow test patterns
- `nl_tracing.nano` - Execution tracing
- `test_all_features.nano` - Feature coverage tests
- `test_driver.nano` - Test runner

### 🔧 Metaprogramming & Compiler (5 examples)

AST manipulation, code generation, and compiler internals.

**Advanced Only:**
- `stdlib_ast_demo.nano` - AST manipulation ⭐
- `nl_demo_selfhosting.nano` - Self-hosting demo
- `nl_macros.nano` - Macro system (experimental)
- `nl_codegen.nano` - Code generation patterns

### 🎲 Real-World Applications (8 examples)

Production-quality examples solving real problems.

- `sdl_asteroids.nano` - Full game with collision detection ⭐
- `sdl_terrain.nano` - Procedural terrain generation ⭐
- `sdl_nanoamp_enhanced.nano` - Audio player ⭐
- `nl_boids.nano` - Flocking simulation ⭐
- `nl_matrix_ops.nano` - Linear algebra ⭐
- `stdlib_ast_demo.nano` - Metaprogramming ⭐
- `sqlite_simple.nano` - Database CRUD (best practices) ✅
- `curl_example.nano` - HTTP client

✅ = Production security best practices  
⭐ = Showcase applications

---

## Learning Paths

### 🚀 Path 1: Core Language (2-3 weeks)

**Goal**: Master NanoLang fundamentals

```
Week 1: Basics
  nl_hello.nano → nl_calculator.nano → nl_types.nano → nl_variables.nano
  → nl_comparisons.nano → nl_conditionals.nano → nl_factorial.nano

Week 2: Data & Control Flow
  nl_loops.nano → nl_arrays.nano → nl_strings.nano → nl_struct.nano

Week 3: Advanced Features
  nl_mutability.nano → nl_first_class_functions.nano → nl_shadow_testing.nano
```

**Prerequisites**: None  
**Outcome**: Comfortable writing NanoLang programs  
**Next**: Choose FFI, Graphics, or Data Structures path

### 🔌 Path 2: FFI & System Integration (1-2 weeks)

**Goal**: Integrate with C libraries and system APIs

```
Day 1-2: FFI Basics
  nl_extern_math.nano → nl_extern_string.nano → nl_extern_char.nano

Day 3-5: Libraries
  curl_example.nano → sqlite_simple.nano

Day 6-7: Advanced Integration
  uv_example.nano → event_example.nano → nl_extern_structs.nano
```

**Prerequisites**: Core Language Path  
**Outcome**: Call C libraries, use databases, make HTTP requests  
**Next**: Build real applications

### 🎮 Path 3: Graphics & Game Development (3-4 weeks)

**Goal**: Build interactive graphical applications

```
Week 1: SDL Basics
  sdl_primitives.nano → sdl_mouse.nano → sdl_keyboard.nano
  → sdl_animation.nano

Week 2: Intermediate Graphics
  sdl_particles.nano → sdl_ui_widgets_extended.nano
  → sdl_nanoviz.nano

Week 3: Game Development
  sdl_pong.nano → sdl_breakout.nano → sdl_asteroids.nano

Week 4: Advanced Projects
  sdl_terrain.nano → sdl_boids.nano → sdl_nanoamp_enhanced.nano
```

**Prerequisites**: Core Language Path  
**Outcome**: Build games and graphical applications  
**Project Ideas**: 
- Space shooter game
- Simulation with visualization
- Audio application with UI

### 📊 Path 4: Data Structures & Algorithms (2-3 weeks)

**Goal**: Master efficient data manipulation

```
Week 1: Arrays & Basic Structures
  nl_arrays.nano → nl_array_complete.nano → nl_sorting.nano
  → nl_searching.nano

Week 2: Generics
  nl_generics_demo.nano → nl_generic_stack.nano → nl_generic_queue.nano

Week 3: Advanced Algorithms
  nl_matrix_ops.nano → nl_boids.nano
```

**Prerequisites**: Core Language Path  
**Outcome**: Implement efficient data structures and algorithms  
**Next**: Metaprogramming path or build applications

### 🔬 Path 5: Metaprogramming & Compiler (1-2 weeks)

**Goal**: Understand and manipulate code at compile time

```
Week 1: AST Basics
  stdlib_ast_demo.nano → nl_demo_selfhosting.nano

Week 2: Advanced Metaprogramming
  nl_macros.nano → nl_codegen.nano → nl_tracing.nano
```

**Prerequisites**: All other paths  
**Outcome**: Write code that generates code, build DSLs  
**Advanced Projects**:
- Custom test framework
- Code generator
- Domain-specific language

---

## Showcase Applications

These 6 examples represent production-quality applications:

### 🎯 Top Tier (Reference Quality)

1. **sdl_asteroids.nano** (A+) - Full game
   - Collision detection, game loop, input handling
   - 450 lines of clean, documented code
   - **Learning Value**: Complete game architecture

2. **sdl_terrain.nano** (A) - Procedural generation
   - Perlin noise, rendering, camera controls
   - Sophisticated algorithm implementation
   - **Learning Value**: Graphics + algorithms

3. **nl_matrix_ops.nano** (A-) - Linear algebra
   - Matrix multiplication, transpose, determinant
   - Demonstrates computational patterns
   - **Learning Value**: Algorithm design

### 🌟 Production Examples

4. **sdl_nanoamp_enhanced.nano** (B+) - Audio player
   - Real UI, file handling, audio playback
   - **Learning Value**: Multimedia + UI integration

5. **nl_generics_demo.nano** (A) - Type system showcase
   - Monomorphization, type safety
   - **Learning Value**: Advanced type systems

6. **stdlib_ast_demo.nano** (B+) - Metaprogramming
   - AST manipulation, code generation
   - **Learning Value**: Compiler internals

**Refinement Roadmap**: See `docs/SHOWCASE_APPLICATIONS.md`

---

## Complete Alphabetical Index

| Example | Category | Difficulty | Topics |
|---------|----------|------------|--------|
| `curl_example.nano` | FFI | 🟡 | HTTP, libcurl, networking |
| `event_example.nano` | FFI | 🟡 | Event loops, async patterns |
| `example_launcher.nano` | Utilities | 🟢 | Project structure |
| `example_launcher_simple.nano` | Utilities | 🟢 | Simple launcher |
| `ncurses_game_of_life.nano` | Graphics | 🟡 | Terminal UI, simulation |
| `ncurses_matrix_rain.nano` | Graphics | 🟢 | Terminal animation |
| `ncurses_snake.nano` | Graphics | 🟡 | Terminal game |
| `nl_advanced_math.nano` | Core | 🔴 | Complex math |
| `nl_array_bounds.nano` | Data Structures | 🟢 | Bounds checking |
| `nl_array_complete.nano` | Data Structures | 🟡 | Comprehensive arrays |
| `nl_arrays_test.nano` | Testing | 🟡 | Array testing |
| `nl_boids.nano` | Algorithms | 🔴 | Flocking simulation |
| `nl_calculator.nano` | Core | 🟢 | Basic arithmetic |
| `nl_comparisons.nano` | Core | 🟢 | Boolean logic |
| `nl_demo_selfhosting.nano` | Metaprogramming | 🔴 | Self-hosting |
| `nl_enum.nano` | Core | 🟡 | Enumerations |
| `nl_extern_char.nano` | FFI | 🟢 | Character FFI |
| `nl_extern_math.nano` | FFI | 🟢 | Math FFI |
| `nl_extern_string.nano` | FFI | 🟢 | String FFI |
| `nl_factorial.nano` | Core | 🟢 | Recursion |
| `nl_first_class_functions.nano` | Core | 🟡 | Higher-order functions |
| `nl_generics_demo.nano` | Advanced | 🔴 | Generics, List<T> |
| `nl_generic_queue.nano` | Data Structures | 🔴 | Generic queue |
| `nl_generic_stack.nano` | Data Structures | 🔴 | Generic stack |
| `nl_hello.nano` | Core | 🟢 | Hello world |
| `nl_loops.nano` | Core | 🟡 | Iteration |
| `nl_matrix_ops.nano` | Algorithms | 🔴 | Linear algebra |
| `nl_mutability.nano` | Core | 🟡 | Mutable state |
| `nl_new_features.nano` | Core | 🟢 | Unary operators |
| `nl_opaque_types.nano` | Core | 🔴 | Type system |
| `nl_shadow_testing.nano` | Testing | 🟡 | Test patterns |
| `nl_strings.nano` | Core | 🟡 | String manipulation |
| `nl_struct.nano` | Core | 🟡 | User types |
| `nl_tracing.nano` | Debugging | 🔴 | Execution tracing |
| `nl_types.nano` | Core | 🟢 | Type system |
| `nl_variables.nano` | Core | 🟢 | Variables |
| `sdl_animation.nano` | Graphics | 🟢 | Basic animation |
| `sdl_asteroids.nano` | Graphics | 🔴 | Full game ⭐ |
| `sdl_boids.nano` | Graphics | 🔴 | Flocking ⭐ |
| `sdl_breakout.nano` | Graphics | 🟡 | Game clone |
| `sdl_keyboard.nano` | Graphics | 🟢 | Input handling |
| `sdl_mouse.nano` | Graphics | 🟢 | Mouse input |
| `sdl_nanoamp_enhanced.nano` | Graphics | 🔴 | Audio player ⭐ |
| `sdl_nanoviz.nano` | Graphics | 🟡 | Visualization |
| `sdl_particles.nano` | Graphics | 🟡 | Particle systems |
| `sdl_pong.nano` | Graphics | 🟡 | Classic game |
| `sdl_primitives.nano` | Graphics | 🟢 | Basic shapes |
| `sdl_terrain.nano` | Graphics | 🔴 | Terrain gen ⭐ |
| `sdl_ui_widgets_extended.nano` | Graphics | 🟡 | UI widgets |
| `sqlite_simple.nano` | FFI | 🟡 | Database ✅ |
| `stdlib_ast_demo.nano` | Metaprogramming | 🔴 | AST demo ⭐ |
| `uv_example.nano` | FFI | 🟡 | Async I/O |

**Legend:**
- 🟢 Beginner
- 🟡 Intermediate  
- 🔴 Advanced
- ⭐ Showcase application
- ✅ Production security best practices

---

## Using This Index

### For New Users

1. Start with [Core Language Path](#path-1-core-language-2-3-weeks)
2. Complete all 🟢 Beginner examples
3. Choose a specialization path (FFI, Graphics, Data Structures)
4. Build a project combining learned concepts

### For Teachers/Curriculum Designers

- Each path includes time estimates
- Examples are ordered by prerequisite dependencies
- Categories align with common CS curricula
- Showcase applications work as final projects

### For Contributors

- All examples should fit into a category
- New examples should specify difficulty and prerequisites
- Update this index when adding/removing examples
- Follow instructional template (see `EXAMPLES_INSTRUCTIONAL_REVIEW.md`)

---

## Maintenance

**How to Update This Index:**

1. Count examples: `ls examples/*.nano | wc -l`
2. Update total count at top
3. When adding examples:
   - Assign difficulty (🟢🟡🔴)
   - Place in correct category
   - Add to alphabetical index
   - Update related learning path
4. When removing examples:
   - Remove from all sections
   - Update count
   - Check learning paths for broken references

**Related Documentation:**
- `EXAMPLES_OVERLAP_AUDIT.md` - Redundancy analysis
- `EXAMPLES_INSTRUCTIONAL_REVIEW.md` - Teaching focus
- `REALWORLD_EXAMPLES_EVALUATION.md` - Production quality
- `SHOWCASE_APPLICATIONS.md` - Flagship examples

---

**Index Version**: 1.0  
**Examples Count**: 85  
**Last Audit**: 2025-12-16

