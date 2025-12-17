# nanolang Examples

This directory contains example programs demonstrating nanolang's features, from simple "Hello World" to complex real-time graphics.

## Quick Start

```bash
# Build all compiled examples
cd examples && make

# Run with interpreter (instant feedback)
../bin/nano hello.nano

# Compile to native binary
../bin/nanoc hello.nano -o hello && ./hello
```

## Example Categories

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
  - Prefix notation for operations
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

- **nl_map_reduce.nano** - Map/Reduce demonstrations
  - First-class functions with map() and reduce()
  - Multiple transformation examples
  - Function composition patterns
  - Conditional returns
  - Edge case handling (0! = 1)

- **fibonacci.nano** - Fibonacci sequence
  - Classic recursive algorithm
  - Loop alternative with `for` and `range`
  - Performance comparison

- **primes.nano** - Prime number checker
  - Boolean logic
  - Optimization with early exit
  - Composite function testing

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

### 🧪 Test Examples (Feature Verification)

Numbered examples (01-34) test specific language features:

- **01-09**: Core operators, strings, floats, loops, mutability
- **10-13**: OS functions, stdlib, advanced math, string operations
- **14-16**: Array operations and bounds checking
- **17-18**: Struct and enum types
- **19-23**: Lists, extern functions, string manipulation
- **24-26**: Random numbers, math algorithms, games
- **27-30**: Tracing, unions, generic lists
- **31-34**: First-class functions, map/filter/fold, closures

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

### Prefix Notation
All operations use prefix (Polish) notation:
```nano
(+ a b)         # Addition
(* (+ 2 3) 4)   # (2 + 3) * 4
(< x 10)        # x < 10
```

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
