# Examples Modernization Plan

## Goal
Refactor all examples to use modern nanolang features:
- ✅ Enums (for named constants)
- ✅ Structs (for grouping data)
- ✅ Top-level constants (for configuration)
- ✅ Dynamic arrays (GC-managed)
- ✅ Type casting (explicit conversions)
- ✅ Unary operators (-, not)

## Current Feature Usage

| Example | Lines | Enums | Structs | Constants | Arrays | Casting | Unary |
|---------|-------|-------|---------|-----------|--------|---------|-------|
| checkers.nano | 1079 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| boids_sdl.nano | 202 | ❌ | ❌ | ✅ (7) | ✅ (6) | ✅ (7) | ✅ |
| particles_sdl.nano | 241 | ❌ | ❌ | ✅ (5) | ✅ (10) | ✅ (8) | ✅ |
| snake.nano | 380 | ❌ | ❌ | ✅ (8) | ✅ (6) | ❌ | ✅ |
| game_of_life.nano | 286 | ❌ | ❌ | ✅ (3) | ✅ (3) | ❌ | ✅ |
| maze.nano | 340 | ❌ | ❌ | ✅ (5) | ✅ (3) | ❌ | ✅ |
| boids_complete.nano | 240 | ❌ | ❌ | ✅ (6) | ✅ (9) | ✅ (5) | ✅ |

## Refactoring Priority

### 1. checkers.nano (Highest Impact - 1079 lines)
**Current:** Uses NO modern features at all!
**Improvements:**
- **Enums:**
  ```nano
  enum PieceType {
      EMPTY = 0,
      RED_PIECE = 1,
      RED_KING = 2,
      BLACK_PIECE = 3,
      BLACK_KING = 4
  }
  
  enum GameState {
      PLAYER_TURN = 0,
      AI_TURN = 1,
      GAME_OVER = 2
  }
  ```

- **Top-level constants:**
  ```nano
  let BOARD_SIZE: int = 8
  let SQUARE_SIZE: int = 80
  let STATUS_HEIGHT: int = 60
  let SDL_INIT_VIDEO: int = 32
  let SDL_WINDOWPOS_UNDEFINED: int = 536805376
  let SDL_WINDOW_SHOWN: int = 4
  let SDL_RENDERER_ACCELERATED: int = 2
  ```

- **Structs:**
  ```nano
  struct Position {
      row: int,
      col: int
  }
  
  struct Move {
      from_row: int,
      from_col: int,
      to_row: int,
      to_col: int,
      is_jump: bool
  }
  ```

**Impact:** Massive - eliminates all magic numbers, makes code self-documenting

### 2. snake.nano (380 lines)
**Current:** Has constants and dynamic arrays, missing enums, structs, type casting
**Improvements:**
- **Enums:**
  ```nano
  enum Direction {
      UP = 0,
      DOWN = 1,
      LEFT = 2,
      RIGHT = 3
  }
  
  enum CellType {
      EMPTY = 0,
      SNAKE = 1,
      FOOD = 2,
      WALL = 3
  }
  ```

- **Structs:**
  ```nano
  struct Position {
      x: int,
      y: int
  }
  ```

- **Type casting:** Add where needed for int/float conversions

### 3. maze.nano (340 lines)
**Current:** Has constants and dynamic arrays
**Improvements:**
- **Enums:**
  ```nano
  enum CellState {
      WALL = 0,
      PATH = 1,
      VISITED = 2
  }
  
  enum Direction {
      NORTH = 0,
      SOUTH = 1,
      EAST = 2,
      WEST = 3
  }
  ```

- **Structs:**
  ```nano
  struct Cell {
      x: int,
      y: int,
      state: CellState
  }
  ```

### 4. game_of_life.nano (286 lines)
**Current:** Has constants and dynamic arrays
**Improvements:**
- **Enums:**
  ```nano
  enum CellState {
      DEAD = 0,
      ALIVE = 1
  }
  ```

- **Structs:**
  ```nano
  struct Grid {
      width: int,
      height: int,
      cells: array<int>
  }
  
  struct Position {
      x: int,
      y: int
  }
  ```

### 5. particles_sdl.nano (241 lines)
**Current:** Has constants, arrays, casting - missing structs and enums
**Improvements:**
- **Structs:**
  ```nano
  struct Particle {
      x: float,
      y: float,
      vx: float,
      vy: float,
      life: float,
      max_life: float
  }
  ```

- **Enums:**
  ```nano
  enum ParticleState {
      ACTIVE = 0,
      DEAD = 1
  }
  ```

### 6. boids_sdl.nano (202 lines)
**Current:** Has constants, arrays, casting - missing structs and enums
**Improvements:**
- **Structs:**
  ```nano
  struct Boid {
      x: float,
      y: float,
      vx: float,
      vy: float
  }
  
  struct Vector2 {
      x: float,
      y: float
  }
  ```

### 7. boids_complete.nano (240 lines)
**Current:** Has constants, arrays, casting - missing structs
**Improvements:**
- **Structs:** (same as boids_sdl.nano)

## Implementation Strategy

1. **Start with checkers.nano** - biggest impact, demonstrates all features
2. **Test after each major change** - ensure functionality preserved
3. **Add shadow tests** for new enum/struct functions
4. **Document patterns** in comments for educational value
5. **Verify in interpreter** before attempting compilation
6. **Commit incrementally** - one example at a time

## Success Criteria

- ✅ All examples use enums where appropriate
- ✅ All examples use structs for grouped data
- ✅ All examples have top-level constants for configuration
- ✅ All examples use dynamic arrays where beneficial
- ✅ All examples compile and run correctly
- ✅ Code is more readable and self-documenting
- ✅ Examples showcase modern nanolang features

## Timeline

- checkers.nano: ~2 hours (major refactor)
- Each other example: ~30 minutes
- Testing: ~1 hour
- **Total: ~5-6 hours**

Let's begin with checkers.nano as it will have the most dramatic improvement!

