# nanolang Examples

This directory contains **150+ example programs** demonstrating nanolang's features, from simple "Hello World" to complex real-time graphics.

## 📚 Learning Path (Start Here!)

**New to NanoLang?** Follow this curated path through the examples:

### Level 1: Absolute Beginner (Start Here!)

Work through these 5 examples in order:

1. **[nl_hello.nano](language/nl_hello.nano)** ⭐ **START HERE!**
   - Your first NanoLang program
   - Demonstrates: `main` function, `println`, shadow tests
   - Time: 5 minutes

2. **[nl_calculator.nano](language/nl_calculator.nano)**
   - Basic arithmetic and functions
   - Demonstrates: prefix and infix notation, multiple functions, parameters
   - Time: 10 minutes

3. **[nl_mutable.nano](language/nl_mutable.nano)**
   - Variable declarations and mutability
   - Demonstrates: `let`, `mut`, type annotations, reassignment
   - Time: 10 minutes

4. **[nl_control_if_while.nano](language/nl_control_if_while.nano)**
   - Conditional logic
   - Demonstrates: `if/else`, comparison operators, boolean logic, `while`
   - Time: 15 minutes

5. **[nl_control_for.nano](language/nl_control_for.nano)**
   - Iteration fundamentals
   - Demonstrates: `for` loops, `range`
   - Time: 15 minutes

**After these 5:** You understand NanoLang basics! ✅

### Level 2: Beginner (Core Concepts)

Once comfortable with Level 1, explore these:

6. **[nl_functions_basic.nano](language/nl_functions_basic.nano)**
   - Function definitions and calls
   - Return values, multiple parameters

7. **[nl_array_complete.nano](language/nl_array_complete.nano)**
   - Array literals and operations
   - Indexing with `at`, `array_length`

8. **[nl_struct.nano](language/nl_struct.nano)**
   - Custom data types
   - Field access, struct literals

9. **[nl_string_operations.nano](language/nl_string_operations.nano)**
   - String manipulation
   - Concatenation, substring, length

10. **[nl_factorial.nano](language/nl_factorial.nano)**
    - Recursion basics
    - Base case and recursive case

**After these 10:** You can write useful programs! ✅

### Level 3: Intermediate (Advanced Features)

Ready for more? Try these next:

11. **[nl_types_union_construct.nano](language/nl_types_union_construct.nano)**
    - Union types and pattern matching
    - Result<T,E> for error handling

12. **[nl_generics_demo.nano](language/nl_generics_demo.nano)**
    - Generic types (List<T>)
    - Type parameters and monomorphization

13. **[nl_hashmap.nano](language/nl_hashmap.nano)**
    - Hash maps for key-value storage
    - HashMap<K,V> operations

14. **[namespace_demo.nano](advanced/namespace_demo.nano)**
    - Code organization with modules
    - Module declarations and visibility

15. **[nl_extern_string.nano](language/nl_extern_string.nano)**
    - Calling C functions
    - FFI basics

**After these 15:** You're proficient in NanoLang! ✅

### Level 4: Advanced (Real Projects)

Build something real:

16. **Terminal Games:**
    - [ncurses_snake.nano](terminal/ncurses_snake.nano) - Classic snake game
    - [ncurses_game_of_life.nano](terminal/ncurses_game_of_life.nano) - Conway's Life

17. **Graphics & Games:**
    - [sdl_checkers.nano](games/sdl_checkers.nano) - Checkers with AI
    - [sdl_asteroids.nano](games/sdl_asteroids.nano) - Full Asteroids clone

18. **Systems Programming:**
    - [performance_optimization.nano](https://github.com/jordanhubbard/nanolang/blob/main/examples/advanced/performance_optimization.nano)
    - [namespace_demo.nano](advanced/namespace_demo.nano)

**After these:** You're ready to build anything! 🚀

## 🔍 Find Examples by Topic

Looking for something specific?

| Topic | Examples | Difficulty |
|-------|----------|------------|
| **Hello World** | nl_hello.nano | 🟢 Beginner |
| **Functions** | nl_functions.nano, nl_factorial.nano, nl_fibonacci.nano | 🟢 Beginner |
| **Data Structures** | nl_arrays.nano, nl_struct.nano, nl_hashmap.nano | 🟡 Intermediate |
| **Generics** | nl_generics.nano, nl_list_operations.nano | 🟡 Intermediate |
| **Pattern Matching** | nl_control_match.nano, nl_types_union_construct.nano | 🟡 Intermediate |
| **Modules & FFI** | nl_modules.nano, nl_extern_ffi.nano | 🟡 Intermediate |
| **Games** | ncurses_snake.nano, sdl_asteroids.nano, sdl_checkers.nano | 🔴 Advanced |
| **Graphics** | sdl_boids.nano, sdl_particles.nano, opengl_triangle.nano | 🔴 Advanced |
| **Performance** | performance_optimization.nano | 🔴 Advanced |

## Quick Start Commands

```bash
# Build all compiled examples
cd examples && make

# Run with interpreter (instant feedback)
../bin/nano hello.nano

# Compile to native binary
../bin/nanoc hello.nano -o hello && ./hello
```

## All Examples by Category

### 🔍 Debugging & Validation (NEW!)

**Demonstrates feedback mechanisms for LLM-driven code generation**

- **logging_levels_demo.nano** - Structured logging with 6 levels
  - TRACE, DEBUG, INFO, WARN, ERROR, FATAL
  - Threshold filtering (default: INFO)
  - Demonstrates log levels in action
  - Shows category-free convenience functions
  
- **logging_categories_demo.nano** - Category-based logging
  - Multi-tier application simulation
  - User registration workflow
  - Categories: validation, database, email, registration
  - Shows how categories help trace execution flow
  
- **coverage_demo.nano** - Runtime instrumentation
  - Coverage tracking with coverage_record()
  - Execution tracing with trace_record()
  - Instrumented fibonacci, classify_number, sum_array
  - Coverage and trace reports
  
- **property_test_sorting.nano** - Property-based testing for algorithms
  - Bubble sort implementation
  - 4 universal properties: length preservation, sorted output, permutation, idempotence
  - 100 random test cases per property
  
- **property_test_math.nano** - Property-based testing for math
  - 15+ mathematical properties
  - Commutativity, identity, inverse, symmetry
  - Triangle inequality for abs()
  - Distributivity demonstration

**See also:**
- `stdlib/log.nano` - Logging API
- `stdlib/coverage.nano` - Coverage/tracing API
- `docs/DEBUGGING_GUIDE.md` - Complete debugging reference
- `docs/PROPERTY_TESTING_GUIDE.md` - Property testing workflow
- `docs/SELF_VALIDATING_CODE_GENERATION.md` - LLM agent self-correction tutorial

---

### 🎮 Graphics & Games (SDL/OpenGL)

**Compiled Examples** - Built with `make`:

- **checkers_sdl** - Full checkers game with AI opponent
  - Board rendering, piece movement, king promotion
  - Minimax AI with heuristic evaluation
  - Click-to-move interface
  
- **boids_sdl** - Flocking simulation (Craig Reynolds' Boids algorithm)
  - 50 boids with cohesion, separation, alignment
  - Real-time physics simulation
  - Screen wrapping boundaries
  
- **particles_sdl** - Particle explosion effect
  - Gravity simulation
  - Mouse-controlled particle spawning
  - Color gradients based on velocity
  
- **falling_sand_sdl** - Cellular automata sandbox
  - Sand, water, stone, wood, fire, smoke
  - Material interactions and physics
  - Paintbrush with different materials
  
- **terrain_explorer_sdl** - 3D terrain with mouse camera
  - Perlin noise terrain generation
  - Mouse-controlled camera movement
  - Height-based coloring
  
- **raytracer_simple** - Real-time ray tracer
  - Mouse-controlled light positioning
  - Ray-sphere intersection
  - Blinn-Phong lighting model
  - 4 spheres with ground plane
  
- **opengl_cube** - Rotating textured cube (requires GLFW + GLEW)
  - 3D transformations
  - Texture mapping
  - Modern OpenGL with shaders
  
- **opengl_teapot** - Rotating teapot with texture cycling
  - Complex 3D mesh (Utah teapot)
  - Procedural texture generation
  - Animation and camera control

### 📚 Learning Examples (Interpreter-Friendly)

**Basic Concepts:**

- **hello.nano** - Hello World
  - Simplest possible program
  - Function definition and string printing
  - Shadow tests demonstration

- **calculator.nano** - Basic arithmetic
  - Prefix and infix notation for operations
  - Multiple function definitions
  - Integer arithmetic (+, -, *, /, %)

- **factorial.nano** - Recursive factorial
  - Recursion demonstration

**Advanced Features:**

- **nl_data_analytics.nano** - Data analytics engine (SHOWCASE)
  - map() and reduce() built-in functions
  - First-class functions as transformation pipelines
  - Functional vs imperative programming comparison
  - Statistical computations (sum, product, min, max, variance)
  - Real-world analytics pipeline architecture
  - 19 comprehensive shadow tests

- **nl_filter_map_fold.nano** - Filter/Map/Fold patterns
  - Canonical functional-programming example (higher-order functions)

- **fibonacci.nano** - Fibonacci sequence
  - Classic recursive algorithm
  - Loop alternative with `for` and `range`
  - Performance comparison

- **primes.nano** - Prime number checker
  - Boolean logic
  - Optimization with early exit
  - Composite function testing

- **nl_pi_chudnovsky.nano** - High-precision π calculator
  - Machin's formula (1706) for π calculation
  - Verifies results against published π archives
  - Demonstrates stdlib/timing.nano for microsecond-precision benchmarking
  - Calculates π to arbitrary decimal places (10, 20, 50, 100, 500, 1000+)
  - Educational: references record-breaking algorithms (Chudnovsky, Bailey-Borwein-Plouffe)

**Advanced Features:**

- **game_of_life.nano** - Conway's Game of Life
  - 2D cellular automaton
  - Array manipulation
  - Pattern evolution

- **snake.nano** - Snake game with AI
  - Game loop and state management
  - Collision detection
  - Simple AI pathfinding

- **maze.nano** - Maze generator and solver
  - Recursive backtracking generation
  - Depth-first search solving
  - ASCII visualization

### 🧪 Feature Verification (Language Tests)

The canonical feature verification programs live in `tests/` as `nl_*.nano` and are run by:

```bash
make test-quick
# or
make test
```

## Building Examples

```bash
# Build all graphics examples
make

# Build specific example
make boids-sdl
make raytracer-simple

# Build SDL examples only
make sdl

# Build OpenGL examples (requires: brew install glfw glew)
make opengl

# Clean build artifacts
make clean
```

## Running Examples

### Interpreter Mode (Instant)

Perfect for learning and quick iteration:

```bash
../bin/nano hello.nano
../bin/nano calculator.nano
../bin/nano game_of_life.nano
```

### Compiled Mode (Native Performance)

For graphics and performance-intensive programs:

```bash
# After 'make', run from bin directory:
../bin/checkers_sdl
../bin/boids_sdl
../bin/raytracer_simple
```

## Example Structure

Every nanolang program follows this pattern:

```nano
# Import modules (for graphics/external libraries)
import "modules/sdl/sdl.nano"

# Define types
struct Vec3 {
    x: float,
    y: float,
    z: float
}

# Helper functions with shadow tests
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add -1 1) 0)
}

# Main entry point
fn main() -> int {
    (println "Hello from nanolang!")
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

## Key Language Features

### Operator Notation
NanoLang supports both prefix and infix notation for binary operators:
```nano
# Prefix (Lisp-style)
(+ a b)         # Addition
(* (+ 2 3) 4)   # (2 + 3) * 4
(< x 10)        # x < 10

# Infix
a + b           # Addition
(2 + 3) * 4     # Use parens for grouping (no PEMDAS)
x < 10          # Comparison
```
Function calls always use prefix notation: `(println "hello")`

### Type System
- Explicit type annotations required
- Struct types with nested fields
- Enum types for tagged values
- Union types for variants
- First-class function types

### Shadow Tests
Every function must have a shadow test block:
```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
}
```

### Mutability
Variables are immutable by default:
```nano
let x: int = 10        # Immutable
let mut y: int = 20    # Mutable
set y 30               # Update mutable variable
```

### Module System
Automatic C library integration:
```nano
import "modules/sdl/sdl.nano"      # SDL2 graphics
import "modules/opengl/opengl.nano" # OpenGL rendering
import "modules/curl/curl.nano"     # HTTP requests
```

## Graphics Programming

### SDL2 Examples
Use the SDL module for 2D graphics:
- Window creation and event handling
- Rendering primitives (pixels, lines, rectangles)
- Texture loading and blitting
- Mouse and keyboard input
- Audio playback

### OpenGL Examples
Use OpenGL modules for 3D graphics:
- Vertex buffers and shaders
- 3D transformations
- Texture mapping
- Modern OpenGL 3.3+ core profile

## Performance

### Interpreter vs Compiler

| Mode | Startup | Speed | Use Case |
|------|---------|-------|----------|
| Interpreter | Instant | ~10x slower | Learning, testing |
| Compiled | ~1s | Native | Production, graphics |

**Recommendation**: Use interpreter for learning and quick tests, compiler for graphics and performance.

## Contributing Examples

When adding new examples:

1. **Start simple** - One concept at a time
2. **Add shadow tests** - Every function needs tests
3. **Document features** - Comment what you're demonstrating
4. **Follow conventions** - Match existing code style
5. **Update this README** - Add to appropriate category

## Learning Path

**Beginner** (Start here):
1. hello.nano
2. calculator.nano
3. factorial.nano
4. fibonacci.nano

**Intermediate** (Core features):
5. primes.nano
6. 17_struct_test.nano
7. 18_enum_test.nano
8. 31_first_class_functions.nano

**Advanced** (Graphics & Complex):
9. particles_sdl.nano
10. boids_sdl.nano
11. raytracer_simple.nano
12. opengl_cube.nano

## More Information

- Language documentation: `../docs/`
- Module system: `../modules/README.md`
- Contributing guide: `../CONTRIBUTING.md`
- Language spec: `../spec.json`

## Getting Help

```bash
# Test your installation
../bin/nano --version
../bin/nanoc --help

# Run test suite
cd .. && ./test.sh

# Build everything
cd .. && make
```

Happy coding with nanolang! 🚀
