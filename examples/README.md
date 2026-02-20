# My Examples

This directory contains 150+ programs where I demonstrate my features, from simple string output to real-time graphics.

## Learning Path

If you are new to me, I recommend following this path through my examples.

### Level 1: Absolute Beginner

I suggest you read and execute these five examples in this order.

1. **[nl_hello.nano](language/nl_hello.nano)**
   - I show you my first program.
   - I demonstrate my `main` function, `println`, and my required shadow tests.
   - Estimated time: 5 minutes.

2. **[nl_calculator.nano](language/nl_calculator.nano)**
   - I perform basic arithmetic.
   - I demonstrate that I support both prefix and infix notation, multiple functions, and parameters.
   - Estimated time: 10 minutes.

3. **[nl_mutable.nano](language/nl_mutable.nano)**
   - I show how I handle variable declarations.
   - I demonstrate `let`, `mut`, type annotations, and reassignment.
   - Estimated time: 10 minutes.

4. **[nl_control_if_while.nano](language/nl_control_if_while.nano)**
   - I execute conditional logic.
   - I demonstrate `if/else`, comparison operators, boolean logic, and `while` loops.
   - Estimated time: 15 minutes.

5. **[nl_control_for.nano](language/nl_control_for.nano)**
   - I demonstrate iteration.
   - I show my `for` loops and `range` function.
   - Estimated time: 15 minutes.

After these five, you understand my basics.

### Level 2: Beginner (Core Concepts)

When you are comfortable with Level 1, I suggest you explore these.

6. **[nl_functions_basic.nano](language/nl_functions_basic.nano)**
   - I define and call functions.
   - I return values and accept multiple parameters.

7. **[nl_array_complete.nano](language/nl_array_complete.nano)**
   - I use array literals and operations.
   - I show indexing with `at` and `array_length`.

8. **[nl_struct.nano](language/nl_struct.nano)**
   - I define custom data types.
   - I demonstrate field access and struct literals.

9. **[nl_string_operations.nano](language/nl_string_operations.nano)**
   - I manipulate strings.
   - I perform concatenation, substring, and length operations.

10. **[nl_factorial.nano](language/nl_factorial.nano)**
    - I demonstrate recursion.
    - I show a base case and a recursive case.

After these ten, you can write useful programs using my core syntax.

### Level 3: Intermediate (Advanced Features)

I offer more complex features in these examples.

11. **[nl_types_union_construct.nano](language/nl_types_union_construct.nano)**
    - I use union types and pattern matching.
    - I show my `Result<T,E>` type for error handling.

12. **[nl_generics_demo.nano](language/nl_generics_demo.nano)**
    - I demonstrate generic types like `List<T>`.
    - I show type parameters and how I handle monomorphization.

13. **[nl_hashmap.nano](language/nl_hashmap.nano)**
    - I use hash maps for key-value storage.
    - I demonstrate `HashMap<K,V>` operations.

14. **[namespace_demo.nano](advanced/namespace_demo.nano)**
    - I organize my code with modules.
    - I demonstrate module declarations and visibility rules.

15. **[nl_extern_string.nano](language/nl_extern_string.nano)**
    - I call functions in C.
    - I demonstrate my basic FFI capabilities.

After these fifteen, you are proficient in my language.

### Level 4: Advanced (Real Projects)

I can build complex systems and games.

16. **Terminal Games:**
    - [ncurses_snake.nano](terminal/ncurses_snake.nano) - I run a classic snake game.
    - [ncurses_game_of_life.nano](terminal/ncurses_game_of_life.nano) - I simulate Conway's Life.

17. **Graphics & Games:**
    - [sdl_checkers.nano](games/sdl_checkers.nano) - I implement checkers with an AI opponent.
    - [sdl_asteroids.nano](games/sdl_asteroids.nano) - I run a full clone of Asteroids.

18. **Systems Programming:**
    - [performance_optimization.nano](https://github.com/jordanhubbard/nanolang/blob/main/examples/advanced/performance_optimization.nano)
    - [namespace_demo.nano](advanced/namespace_demo.nano)

After these, you are ready to build any system with me.

## Find Examples by Topic

I have categorized my examples to help you find specific features.

| Topic | Examples | Difficulty |
|-------|----------|------------|
| Hello World | nl_hello.nano | Beginner |
| Functions | nl_functions.nano, nl_factorial.nano, nl_fibonacci.nano | Beginner |
| Data Structures | nl_arrays.nano, nl_struct.nano, nl_hashmap.nano | Intermediate |
| Generics | nl_generics.nano, nl_list_operations.nano | Intermediate |
| Pattern Matching | nl_control_match.nano, nl_types_union_construct.nano | Intermediate |
| Modules & FFI | nl_modules.nano, nl_extern_ffi.nano | Intermediate |
| Games | ncurses_snake.nano, sdl_asteroids.nano, sdl_checkers.nano | Advanced |
| Graphics | sdl_boids.nano, sdl_particles.nano, opengl_triangle.nano | Advanced |
| Performance | performance_optimization.nano | Advanced |

## Quick Start Commands

```bash
# Build all compiled examples
cd examples && make

# Run with interpreter (instant feedback)
../bin/nano hello.nano

# Compile to native binary
../bin/nanoc hello.nano -o hello && ./hello
```

## My Examples by Category

### Debugging & Validation

I provide feedback mechanisms for machine-driven code generation.

- **logging_levels_demo.nano** - I demonstrate structured logging with 6 levels.
  - I show TRACE, DEBUG, INFO, WARN, ERROR, and FATAL levels.
  - I use threshold filtering which defaults to INFO.
  - I show my category-free convenience functions.
  
- **logging_categories_demo.nano** - I use category-based logging.
  - I simulate a multi-tier application.
  - I show a user registration workflow.
  - I use categories for validation, database, email, and registration.
  
- **coverage_demo.nano** - I perform runtime instrumentation.
  - I track coverage with `coverage_record`.
  - I trace execution with `trace_record`.
  - I instrument my implementations of fibonacci, classify_number, and sum_array.
  
- **property_test_sorting.nano** - I use property-based testing for algorithms.
  - I implement bubble sort.
  - I verify four properties: length preservation, sorted output, permutation, and idempotence.
  - I run 100 random test cases for each property.
  
- **property_test_math.nano** - I use property-based testing for mathematics.
  - I verify 15 mathematical properties.
  - I demonstrate commutativity, identity, inverse, and symmetry.
  - I show the triangle inequality for `abs` and the distributive property.

See also:
- `stdlib/log.nano` - My logging API.
- `stdlib/coverage.nano` - My coverage and tracing API.
- `docs/DEBUGGING_GUIDE.md` - My complete debugging reference.
- `docs/PROPERTY_TESTING_GUIDE.md` - My property testing workflow.
- `docs/SELF_VALIDATING_CODE_GENERATION.md` - My tutorial for LLM agent self-correction.

---

### Graphics & Games (SDL/OpenGL)

I build these examples with `make`.

- **checkers_sdl** - I run a full checkers game with an AI opponent.
  - I handle board rendering, piece movement, and king promotion.
  - I use a minimax AI with heuristic evaluation.
  
- **boids_sdl** - I simulate flocking using Craig Reynolds' Boids algorithm.
  - I manage 50 boids with cohesion, separation, and alignment rules.
  - I perform real-time physics simulation and screen wrapping.
  
- **particles_sdl** - I simulate a particle explosion.
  - I handle gravity and mouse-controlled spawning.
  - I calculate color gradients based on velocity.
  
- **falling_sand_sdl** - I run a cellular automata sandbox.
  - I simulate sand, water, stone, wood, fire, and smoke.
  - I handle material interactions and physics.
  
- **terrain_explorer_sdl** - I render 3D terrain with a mouse-controlled camera.
  - I generate terrain using Perlin noise.
  - I manage camera movement and height-based coloring.
  
- **raytracer_simple** - I run a real-time ray tracer.
  - I handle mouse-controlled light positioning and ray-sphere intersection.
  - I use the Blinn-Phong lighting model to render spheres on a ground plane.
  
- **opengl_cube** - I render a rotating textured cube.
  - I perform 3D transformations and texture mapping using modern OpenGL shaders.
  
- **opengl_teapot** - I render the Utah teapot with texture cycling.
  - I manage a complex 3D mesh and procedural texture generation.

### Learning Examples (Interpreter-Friendly)

**Basic Concepts:**

- **hello.nano** - My simplest program.
  - I define a function, print a string, and demonstrate my shadow tests.

- **calculator.nano** - I perform basic arithmetic.
  - I show my prefix and infix notation for integer operations.

- **factorial.nano** - I demonstrate a recursive factorial calculation.

**Advanced Features:**

- **nl_data_analytics.nano** - My data analytics engine showcase.
  - I use my built-in `map` and `reduce` functions.
  - I treat functions as first-class pipelines for statistical computations.
  - I include 19 shadow tests to verify my implementation.

- **nl_filter_map_fold.nano** - I demonstrate functional programming patterns.

- **fibonacci.nano** - I calculate the Fibonacci sequence.
  - I show both recursive and iterative approaches for comparison.

- **primes.nano** - I check for prime numbers.
  - I use boolean logic and optimize with early exits.

- **nl_pi_chudnovsky.nano** - I calculate high-precision pi.
  - I use Machin's formula and verify results against published archives.
  - I demonstrate `stdlib/timing.nano` for benchmarking.

- **game_of_life.nano** - I simulate a 2D cellular automaton.

- **snake.nano** - I run a snake game with simple AI pathfinding.

- **maze.nano** - I generate and solve mazes.
  - I use recursive backtracking for generation and depth-first search for solving.

### Feature Verification

I keep my canonical feature verification programs in `tests/`. I run them with `make test` or `make test-quick`.

## Building Examples

```bash
# Build all graphics examples
make

# Build specific example
make boids-sdl
make raytracer-simple

# Build SDL examples only
make sdl

# Build OpenGL examples
make opengl

# Clean build artifacts
make clean
```

## Running Examples

### Interpreter Mode

I recommend my interpreter for learning and quick iteration.

```bash
../bin/nano hello.nano
../bin/nano calculator.nano
../bin/nano game_of_life.nano
```

### Compiled Mode

I use my compiler for programs that require native performance.

```bash
# After 'make', run from bin directory:
../bin/checkers_sdl
../bin/boids_sdl
../bin/raytracer_simple
```

## Example Structure

Every program I execute follows this pattern.

```nano
# I import modules for graphics or external libraries
import "modules/sdl/sdl.nano"

# I define types
struct Vec3 {
    x: float,
    y: float,
    z: float
}

# I define helper functions with mandatory shadow tests
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add -1 1) 0)
}

# I define the main entry point
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
I support both prefix and infix notation for binary operators.
```nano
# Prefix notation
(+ a b)
(* (+ 2 3) 4)
(< x 10)

# Infix notation
a + b
(2 + 3) * 4
x < 10
```
I always use prefix notation for function calls: `(println "hello")`.

### Type System
- I require explicit type annotations.
- I support struct types, enum types, and union types.
- I treat functions as first-class values.

### Shadow Tests
I refuse to compile a function unless you provide a shadow test block.
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
My variables are immutable by default.
```nano
let x: int = 10        # I am immutable
let mut y: int = 20    # I am mutable
set y 30               # I update my value
```

### Module System
I integrate with C libraries automatically.
```nano
import "modules/sdl/sdl.nano"      # I use SDL2
import "modules/opengl/opengl.nano" # I use OpenGL
import "modules/curl/curl.nano"     # I use curl
```

## Graphics Programming

### SDL2 Examples
I use the SDL module for 2D graphics, including window creation, event handling, and rendering primitives.

### OpenGL Examples
I use OpenGL modules for 3D graphics, supporting vertex buffers, shaders, and transformations.

## Performance

| Mode | Startup | Speed | Use Case |
|------|---------|-------|----------|
| Interpreter | Instant | ~10x slower | Learning, testing |
| Compiled | ~1s | Native | Production, graphics |

I recommend my interpreter for learning and my compiler for performance-intensive tasks.

## Contributing Examples

When you add new examples for me:
1. I prefer one concept per example.
2. I require shadow tests for every function.
3. I expect comments that describe the features you demonstrate.
4. I require you to match my existing code style.
5. I expect you to update this directory.

## Learning Path

Beginner:
1. hello.nano
2. calculator.nano
3. factorial.nano
4. fibonacci.nano

Intermediate:
5. primes.nano
6. 17_struct_test.nano
7. 18_enum_test.nano
8. 31_first_class_functions.nano

Advanced:
9. particles_sdl.nano
10. boids_sdl.nano
11. raytracer_simple.nano
12. opengl_cube.nano

## More Information

- My documentation: `../docs/`
- My module system: `../modules/README.md`
- My contributing guide: `../CONTRIBUTING.md`

## Getting Help

```bash
# I test your installation
../bin/nano --version
../bin/nanoc --help

# I run my test suite
cd .. && ./test.sh

# I build all my components
cd .. && make
```
