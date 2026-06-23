# Nanolang Showcase Games & Demos - Master Roadmap

**Goal**: Create 5 production-quality demos that showcase nanolang's capabilities for game development

---

## 🎮 Priority Order

### 🚀 **1. Asteroids (With Physics)** - **IMPLEMENT NOW**
**Status**: GC infrastructure complete, ready to build  
**Timeline**: 1-2 hours  
**Complexity**: Medium

### 🔊 **2. Music Sequencer (Tracker-Style)** 
**Timeline**: 2-3 hours  
**Complexity**: Medium-High

### 🌱 **3. Procedural Terrain Explorer**
**Timeline**: 2-3 hours  
**Complexity**: Medium

### 🐕 **4. Boids Flocking Simulation**
**Timeline**: 1-2 hours  
**Complexity**: Low-Medium

### 📦 **5. Physics Sandbox (Falling Sand)**
**Timeline**: 2-3 hours  
**Complexity**: Medium-High

---

## 🚀 1. Asteroids (Full Implementation) - **NOW**

### Overview
Classic Asteroids with:
- Vector-based rendering
- Physics simulation (momentum, rotation, thrust)
- Particle effects (explosions, engine thrust)
- Dynamic entity management (GC handles cleanup)
- Smooth delta-time rendering
- Spatial partitioning (simple grid)

### Core Features

**Player Ship**:
- ✅ Rotation (left/right arrows)
- ✅ Thrust (up arrow) with momentum
- ✅ Fire bullets (spacebar)
- ✅ Particle trail when thrusting
- ✅ Collision detection with asteroids

**Asteroids**:
- ✅ Spawn at random positions/velocities
- ✅ Wrap around screen edges
- ✅ Split into smaller asteroids when hit (3 sizes)
- ✅ Rotate while moving
- ✅ Dynamic spawning (GC-managed!)

**Bullets**:
- ✅ Fixed velocity, limited lifetime
- ✅ Destroy asteroids on collision
- ✅ Auto-despawn (GC cleanup)

**Particle Effects**:
- ✅ Explosion particles when asteroid destroyed
- ✅ Engine thrust particles
- ✅ Fade-out animation

**Scoring & UI**:
- ✅ Score display
- ✅ Lives counter
- ✅ Game over screen
- ✅ Restart functionality

### Technical Highlights

**Vector Math Module** (`modules/vector2d/`):
```nano
struct Vector2D {
    x: float,
    y: float
}

fn vec_add(a: Vector2D, b: Vector2D) -> Vector2D
fn vec_sub(a: Vector2D, b: Vector2D) -> Vector2D
fn vec_scale(v: Vector2D, s: float) -> Vector2D
fn vec_length(v: Vector2D) -> float
fn vec_normalize(v: Vector2D) -> Vector2D
fn vec_dot(a: Vector2D, b: Vector2D) -> float
fn vec_rotate(v: Vector2D, angle: float) -> Vector2D
fn vec_angle_to_vec(angle: float) -> Vector2D
fn vec_distance(a: Vector2D, b: Vector2D) -> float
```

**Physics Engine** (simple):
- Velocity integration: `pos = pos + vel * dt`
- Rotation: `angle = angle + angular_vel * dt`
- Thrust: `vel = vel + forward * thrust_power * dt`
- Damping: `vel = vel * 0.99` (space friction)

**Dynamic Entity System** (GC-powered):
```nano
# All entities in dynamic arrays
let mut asteroids: array<Asteroid> = []
let mut bullets: array<Bullet> = []
let mut particles: array<Particle> = []

# Spawn/destroy without manual memory management
fn spawn_asteroid(x: float, y: float, size: int) {
    set asteroids (array_push asteroids (create_asteroid x y size))
}

fn destroy_asteroid(index: int) {
    let ast: Asteroid = (at asteroids index)
    (spawn_explosion ast.x ast.y)
    set asteroids (array_remove_at asteroids index)
    # GC automatically frees the asteroid!
}
```

**Collision Detection** (spatial grid):
- Divide screen into grid cells
- Only check collisions within same cell + neighbors
- O(n) instead of O(n²) for large entity counts

**Rendering**:
- SDL2 for graphics
- Line-based rendering (classic vector look)
- Particle system with alpha blending

### Files to Create

**Core Game**:
- `examples/asteroids.nano` - Main game implementation
- `modules/vector2d/vector2d.nano` - Vector math library
- `modules/vector2d/module.json` - Module metadata

**Support**:
- `examples/asteroids_collision_grid.nano` - Spatial partitioning demo
- `tests/test_vector2d.nano` - Vector math tests

### Implementation Plan

**Step 1: Vector Math Module** (15 min)
- Implement all vector operations
- Add shadow tests for each function
- Test in isolation

**Step 2: Entity Structs** (10 min)
```nano
struct Ship {
    pos: Vector2D,
    vel: Vector2D,
    angle: float,
    thrust_on: bool
}

struct Asteroid {
    pos: Vector2D,
    vel: Vector2D,
    angle: float,
    angular_vel: float,
    size: int,
    health: int
}

struct Bullet {
    pos: Vector2D,
    vel: Vector2D,
    lifetime: float
}

struct Particle {
    pos: Vector2D,
    vel: Vector2D,
    lifetime: float,
    alpha: float
}
```

**Step 3: Game State** (10 min)
```nano
struct GameState {
    ship: Ship,
    asteroids: array<Asteroid>,
    bullets: array<Bullet>,
    particles: array<Particle>,
    score: int,
    lives: int,
    game_over: bool,
    frame_time: float
}
```

**Step 4: Physics & Update Loop** (20 min)
- Ship controls (thrust, rotate)
- Entity movement with velocity integration
- Screen wrapping
- Bullet lifecycle

**Step 5: Collision Detection** (15 min)
- Circle-circle collision
- Bullet-asteroid
- Ship-asteroid
- Asteroid splitting logic

**Step 6: Particle System** (15 min)
- Explosion particles
- Thrust trail
- Fade-out animation

**Step 7: Rendering** (15 min)
- SDL line drawing
- Ship rendering (triangle)
- Asteroid rendering (polygons)
- UI text (score, lives)

**Step 8: Game Loop & Polish** (20 min)
- Delta-time integration
- Game over/restart
- Wave spawning
- Sound effects (optional)

**Total**: ~2 hours for full implementation

---

## 🔊 2. Music Sequencer (Tracker-Style)

### Overview
4-channel tracker-style music sequencer inspired by Amiga trackers.

### Features
- ✅ 4 audio channels
- ✅ Basic waveforms: sine, square, triangle, sawtooth, noise
- ✅ Pattern editor (grid UI)
- ✅ Step sequencer (16 steps per pattern)
- ✅ Tempo control (BPM)
- ✅ Volume control per channel
- ✅ Play/pause/stop controls
- ✅ Save/load custom format (.NANO format)
- ✅ Real-time audio synthesis

### Technical Highlights
- **Audio Synthesis**: Generate waveforms in real-time
- **Event Scheduling**: Precise timing with callback system
- **UI Grid**: Efficient rendering of pattern data
- **File Format**: Custom binary format for songs

### What It Demonstrates
- Audio programming (SDL_mixer or raw PCM)
- Event-driven architecture
- Grid UI rendering
- File I/O and serialization
- State machines (play/edit modes)

### Files
- `examples/tracker.nano`
- `modules/audio/audio.nano` (waveform generation)
- `modules/audio/audio_synth.c` (C backend for synthesis)

---

## 🌱 3. Procedural Terrain Explorer

### Overview
Generate infinite 2D terrain using Perlin noise, explore with WASD + fog-of-war.

### Features
- ✅ Perlin/Simplex noise terrain generation
- ✅ Multiple biomes (ocean, plains, forest, mountains, snow)
- ✅ Fog of war (reveal as you explore)
- ✅ Smooth camera scrolling
- ✅ Minimap
- ✅ Tile-based rendering with LOD
- ✅ Deterministic from seed
- ✅ Infinite world (generate chunks on-demand)

### Technical Highlights
- **Procedural Generation**: Perlin noise implementation
- **Chunking**: Generate terrain in 16x16 chunks
- **LOD**: Lower detail for distant tiles
- **Fog of War**: Bitmap-based visibility
- **Camera**: Smooth interpolation

### What It Demonstrates
- Noise algorithms
- Chunked world generation
- Efficient tile rendering
- Camera systems
- Bitmap manipulation

### Files
- `examples/terrain_explorer.nano`
- `modules/noise/perlin.nano` (Perlin noise)
- `modules/world/chunk.nano` (Chunk management)

---

## 🐕 4. Boids Flocking Simulation

### Overview
200-1000 boids with emergent flocking behavior (separation, alignment, cohesion).

### Features
- ✅ 3 steering behaviors: separation, alignment, cohesion
- ✅ Configurable parameters (sliders)
- ✅ Predator/prey mode
- ✅ Obstacles (boids avoid them)
- ✅ Performance metrics (FPS, boid count)
- ✅ Color-coded by velocity
- ✅ Trail rendering (optional)

### Technical Highlights
- **Spatial Partitioning**: Grid for O(n) neighbor searches
- **Vector Math**: Steering force calculations
- **Batch Rendering**: Render 1000+ boids efficiently
- **UI Sliders**: Interactive parameter tuning

### What It Demonstrates
- Emergent AI behavior
- Spatial data structures
- Vector mathematics
- Performance optimization
- Real-time parameter tuning

### Files
- `examples/boids.nano`
- `modules/spatial/grid.nano` (Spatial grid)

---

## 📦 5. Physics Sandbox (Falling Sand)

### Overview
Cellular automata with sand, water, smoke, fire, wood, etc.

### Features
- ✅ Multiple materials:
  - Sand (falls, piles up)
  - Water (flows, spreads)
  - Smoke (rises, dissipates)
  - Fire (spreads, consumes wood)
  - Wood (static, flammable)
  - Stone (static, indestructible)
- ✅ Material interactions (fire + wood = fire)
- ✅ Tool palette (draw, erase, materials)
- ✅ Brush size control
- ✅ Pause/play
- ✅ Clear all
- ✅ Dirty rectangle optimization

### Technical Highlights
- **Cellular Automata**: Simple state machine per cell
- **Optimization**: Only update "dirty" regions
- **Direct Pixel Manipulation**: Fast rendering
- **Material Properties**: Density, flammability, etc.

### What It Demonstrates
- 2D grid simulation
- State machines
- Dirty rectangle optimization
- Pixel-level rendering
- Interactive tools

### Files
- `examples/falling_sand.nano`
- `modules/cellular/automata.nano`

---

## 📊 Showcase Progression

### Complexity Ladder
1. **Boids** (easiest) - Pure math, no complex state
2. **Asteroids** (easy-medium) - Classic game structure
3. **Terrain Explorer** (medium) - Procedural generation
4. **Falling Sand** (medium-high) - Optimization challenges
5. **Music Sequencer** (hardest) - Audio + UI complexity

### Learning Path
Each demo builds on previous concepts:
- **Boids** → Vector math, entity management
- **Asteroids** → Boids + collision detection + particles
- **Terrain** → Asteroids + procedural generation + chunking
- **Falling Sand** → Terrain + cellular automata + optimization
- **Tracker** → All above + audio synthesis + file I/O

---

## 🎯 Implementation Strategy

### Phase 1: Asteroids (NOW) ⚡
**Priority**: Highest  
**Rationale**: GC infrastructure is fresh, demonstrates dynamic entities perfectly

### Phase 2: Boids (Next) 🐕
**Priority**: High  
**Rationale**: Reuses vector math from Asteroids, pure eye candy

### Phase 3: Terrain Explorer 🌱
**Priority**: High  
**Rationale**: Different genre (exploration), showcases procedural generation

### Phase 4: Falling Sand 📦
**Priority**: Medium  
**Rationale**: Different paradigm (cellular), showcases optimization

### Phase 5: Music Sequencer 🔊
**Priority**: Medium-Low  
**Rationale**: Most complex, requires audio backend work

---

## 🛠️ Shared Infrastructure

### Modules to Create
1. **`modules/vector2d/`** - Vector math (Asteroids, Boids)
2. **`modules/spatial/`** - Spatial partitioning (Asteroids, Boids)
3. **`modules/noise/`** - Perlin/Simplex (Terrain)
4. **`modules/audio/`** - Audio synthesis (Tracker)
5. **`modules/cellular/`** - Cellular automata (Falling Sand)
6. **`modules/ui/`** - UI widgets (All games)

### Reusable Components
- Delta-time calculation
- FPS counter
- Basic text rendering
- Input handling
- Color utilities
- Math utilities (lerp, clamp, etc.)

---

## 📈 Success Metrics

### Per Demo
- ✅ Runs at 60 FPS on modern hardware
- ✅ Zero memory leaks (GC verified)
- ✅ Clean, readable code
- ✅ Comprehensive shadow tests
- ✅ Documentation with screenshots
- ✅ < 500 lines of nanolang code (excluding modules)

### Collective Impact
- **Portfolio Quality**: Each demo is production-ready
- **Educational Value**: Clear progression from simple to complex
- **Language Showcase**: Proves nanolang is game-ready
- **Community**: Seed projects for others to learn from

---

## 📝 Documentation Plan

### For Each Demo
1. **README.md** - Overview, controls, building
2. **TUTORIAL.md** - Step-by-step implementation guide
3. **API.md** - Module API documentation
4. **SCREENSHOT.png** - Visual preview
5. **VIDEO.gif** - Animated demo (optional)

### Master Documentation
- **SHOWCASE_INDEX.md** - Gallery of all demos
- **GETTING_STARTED.md** - Quick start guide
- **ADVANCED_TECHNIQUES.md** - Optimization tips

---

## 🎬 Next Steps

### Immediate (Now)
1. ✅ Complete Asteroids game
2. ✅ Create vector2d module
3. ✅ Test on real hardware

### Short-term (Next Session)
1. Implement Boids simulation
2. Create spatial partitioning module
3. Document both demos

### Medium-term (This Week)
1. Terrain Explorer
2. Falling Sand
3. Shared infrastructure modules

### Long-term (Next Week)
1. Music Sequencer
2. Master showcase documentation
3. Video recordings

---

## 🚀 Let's Start with Asteroids NOW!

**Timeline**: ~2 hours  
**Output**: Production-ready Asteroids game with physics, particles, and GC-managed entities

**Command to run when done**:
```bash
make
./bin/nano examples/asteroids.nano --call main
```

Let's build it! 🎮

