# Asteroids Enhanced - Feature Documentation

## Overview

Enhanced version of the classic Asteroids arcade game with authentic gameplay mechanics including multi-size asteroids, breaking mechanics, and visual thrust effects.

## New Features

### 1. Multi-Size Asteroids

Three distinct asteroid sizes with different properties:

| Size   | Radius | Speed Multiplier | Score Value |
|--------|--------|------------------|-------------|
| Large  | 30px   | 40.0             | 20 points   |
| Medium | 20px   | 60.0             | 50 points   |
| Small  | 10px   | 90.0             | 100 points  |

**Speed Scaling:** Smaller asteroids move 50% faster than medium, which move 50% faster than large asteroids. This creates challenging gameplay where shooting large asteroids creates faster-moving threats.

**Visual Differentiation:** Each size has a different shade of gray:
- Large: RGB(180, 180, 180) - Brightest
- Medium: RGB(150, 150, 150) - Mid-tone
- Small: RGB(120, 120, 120) - Darkest

### 2. Breaking Mechanics (Classic Arcade Style)

#### Large Asteroid Hit
```
Large Asteroid → 2 Medium Asteroids
```
- Velocity rotated ±90 degrees
- Speed multiplied by 1.5x
- Spawn at impact location
- Creates directional spread pattern

**Implementation:**
```nano
# Medium 1 (rotate velocity +90 degrees)
velocity_new = (-old_vy * 1.5, old_vx * 1.5)

# Medium 2 (rotate velocity -90 degrees)
velocity_new = (old_vy * 1.5, -old_vx * 1.5)
```

#### Medium Asteroid Hit
```
Medium Asteroid → 3 Small Asteroids
```
- Three-way split at 120° angles
- Speed multiplied by 2.0x
- Creates triangular spread pattern
- Much faster and more dangerous

**Implementation:**
```nano
# Small 1: Original direction * 2
# Small 2: Rotated +120 degrees * 2
# Small 3: Rotated -120 degrees * 2
```

#### Small Asteroid Hit
```
Small Asteroid → Destroyed
```
- No child asteroids
- Highest score value (100 points)
- Satisfying endgame cleanup

### 3. Thrust Cone Visualization

Visual feedback when ship is under thrust:

**Appearance:**
- Orange/yellow color: RGB(255, 200, 0)
- Cone shape behind ship
- Drawn as triangle connecting ship center and back edges to thrust point

**Dimensions:**
- Length: 80% of ship size (12px with default ship)
- Points opposite to ship's heading direction
- Angle: 180° from ship direction

**Implementation:**
```nano
if thrust {
    let thrust_length: float = (* SHIP_SIZE 0.8)
    let thrust_angle: float = (+ ship_angle PI)
    
    # Draw cone as 3 lines forming triangle
    # From ship center to thrust point
    # From back edge 1 to thrust point
    # From back edge 2 to thrust point
}
```

**Benefits:**
- Immediate visual feedback for thrust input
- Helps player understand ship orientation
- Classic arcade aesthetic
- Shows thrust direction clearly

## Particle Effects

Particle explosions scaled by asteroid size:

| Event                  | Particle Count | Speed Range  |
|------------------------|----------------|--------------|
| Large asteroid hit     | 30 particles   | 50-107 px/s  |
| Medium asteroid hit    | 20 particles   | 50-107 px/s  |
| Small asteroid hit     | 10 particles   | 50-107 px/s  |
| Ship destroyed         | 40 particles   | 80-176 px/s  |

**Color:** Orange RGB(255, 150, 0) with alpha based on lifetime

## Scoring System

Progressive scoring encourages destroying smaller, faster asteroids:

```
Large:  20 points  (easy target, slow)
Medium: 50 points  (2.5x harder)
Small:  100 points (5x harder, fastest)
```

**Strategic Implications:**
- Shooting a large asteroid: 20 + (2×50) = 120 total points
- Breaking down medium: 50 + (3×100) = 350 total points
- Full chain: 20 + 100 + 300 = 420 points per large asteroid

Players are incentivized to break down asteroids rather than avoid them, creating active gameplay.

## Technical Implementation

### Data Structure (Parallel Arrays)

Added `asteroid_size` array alongside existing position/velocity arrays:

```nano
let mut asteroid_x: array<float> = []
let mut asteroid_y: array<float> = []
let mut asteroid_vx: array<float> = []
let mut asteroid_vy: array<float> = []
let mut asteroid_size: array<int> = []      # New!
let mut asteroid_active: array<bool> = []
```

### Size Constants
```nano
let ASTEROID_SIZE_LARGE: int = 3
let ASTEROID_SIZE_MEDIUM: int = 2
let ASTEROID_SIZE_SMALL: int = 1
```

### Helper Functions

**get_asteroid_radius(size: int) -> float**
- Maps size integer to pixel radius
- Used for rendering and collision detection

**get_asteroid_score(size: int) -> int**
- Maps size to score value
- Returns 20, 50, or 100

### Collision Detection Enhancement

```nano
# Get radius based on size
let asize: int = (at asteroid_size j)
let aradius: float = (get_asteroid_radius asize)

# Use size-specific radius for collision
if (circles_collide bx by 2.0 ax ay aradius) {
    # Determine child asteroids based on size
    if (== asize ASTEROID_SIZE_LARGE) {
        # Create 2 medium asteroids
    } else if (== asize ASTEROID_SIZE_MEDIUM) {
        # Create 3 small asteroids
    } else {
        # Small - just destroy
    }
}
```

## Gameplay Comparison

### Original Version (asteroids_sdl.nano)
- All asteroids same size (20px radius)
- All asteroids same speed
- Simple destruction (no breaking)
- Flat scoring
- No thrust visualization

### Enhanced Version (asteroids_enhanced_sdl.nano)
- 3 distinct asteroid sizes
- Speed varies by size (smaller = faster)
- Breaking mechanics (2/3 child asteroids)
- Progressive scoring (20/50/100)
- Thrust cone visualization

## Files

- **asteroids_enhanced_sdl.nano** - Enhanced game implementation
- **asteroids_sdl.nano** - Original simple version (still available)

## Building and Running

```bash
# Build enhanced version
make -C examples asteroids-enhanced-sdl

# Or directly
./bin/nanoc examples/asteroids_enhanced_sdl.nano -o bin/asteroids_enhanced_sdl

# Run
./bin/asteroids_enhanced_sdl
```

## Controls

| Key         | Action         |
|-------------|----------------|
| Up Arrow    | Thrust         |
| Left Arrow  | Rotate Left    |
| Right Arrow | Rotate Right   |
| Space       | Shoot          |
| ESC         | Quit           |

## Code Statistics

- **Lines:** 636 (vs 441 in original)
- **Added Features:** 3 major (sizes, breaking, thrust cone)
- **New Functions:** 2 helper functions
- **Shadow Tests:** 5 test functions
- **Complexity:** Authentic arcade mechanics

## Performance

- No performance impact from size checking
- Breaking creates 2-3 new asteroids per collision
- Particle system scales with asteroid size
- Maintains 60 FPS gameplay

## Future Enhancements

Possible additions while maintaining arcade feel:
- UFOs (small/large with different behaviors)
- Hyperspace escape
- Lives/respawn system
- Wave progression (increasing difficulty)
- Sound effects (thrust, shoot, explosions)

## Design Philosophy

The enhanced version stays true to the original 1979 Atari Asteroids arcade game mechanics:
- Large asteroids break into medium
- Medium asteroids break into small
- Progressive difficulty through breaking
- Visual feedback for all actions
- Pure vector-style graphics

This implementation proves nanolang can handle authentic arcade game mechanics while remaining readable and maintainable.
