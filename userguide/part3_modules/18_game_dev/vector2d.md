# 18.2 vector2d — 2D Vector Math

**The mathematical backbone of 2D game development.**

The `vector2d` module provides a `Vector2D` struct and a rich library of pure functions for 2D math. "Pure" here means every function takes values and returns a new value — nothing is mutated. This makes game logic easy to test, reason about, and compose.

## Quick Start

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_add, vec_sub,
                                             vec_scale, vec_normalize, vec_length,
                                             vec_distance, vec_zero

fn move_toward_target(pos: Vector2D, target: Vector2D, speed: float) -> Vector2D {
    let direction: Vector2D = (vec_sub target pos)
    let unit: Vector2D = (vec_normalize direction)
    let step: Vector2D = (vec_scale unit speed)
    return (vec_add pos step)
}

shadow move_toward_target {
    let pos: Vector2D = (vec_new 0.0 0.0)
    let target: Vector2D = (vec_new 10.0 0.0)
    let result: Vector2D = (move_toward_target pos target 1.0)
    assert (== result.x 1.0)
    assert (== result.y 0.0)
}
```

## The Vector2D Struct

```nano
struct Vector2D {
    x: float,
    y: float
}
```

Fields are accessed with dot notation: `v.x`, `v.y`. Because structs are immutable in NanoLang, "updating" a vector always means creating a new one.

## Creating Vectors

### `vec_new(x, y)` — Create from components

```nano
let pos: Vector2D = (vec_new 3.0 4.0)
# pos.x == 3.0, pos.y == 4.0
```

### `vec_zero()` — The zero vector

```nano
let origin: Vector2D = (vec_zero)
# origin.x == 0.0, origin.y == 0.0
```

### `vec_from_angle(angle)` — Unit vector pointing in a direction

Creates a unit vector from an angle in **radians**. Angle 0 points right (+X), π/2 points up (+Y).

```nano
from "modules/vector2d/vector2d.nano" import vec_from_angle

fn heading_right() -> Vector2D {
    return (vec_from_angle 0.0)   # (1.0, 0.0)
}

shadow heading_right {
    let v: Vector2D = (heading_right)
    assert (== v.x 1.0)
}
```

## Arithmetic Operations

### `vec_add(a, b)` — Add two vectors

Translating a position by a velocity:

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_add

fn apply_velocity(pos: Vector2D, vel: Vector2D) -> Vector2D {
    return (vec_add pos vel)
}

shadow apply_velocity {
    let pos: Vector2D = (vec_new 5.0 5.0)
    let vel: Vector2D = (vec_new 1.0 -1.0)
    let next: Vector2D = (apply_velocity pos vel)
    assert (== next.x 6.0)
    assert (== next.y 4.0)
}
```

### `vec_sub(a, b)` — Subtract two vectors

Computing the vector from one point to another:

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_sub

fn vector_to(from_pos: Vector2D, to_pos: Vector2D) -> Vector2D {
    return (vec_sub to_pos from_pos)
}

shadow vector_to {
    let a: Vector2D = (vec_new 2.0 3.0)
    let b: Vector2D = (vec_new 5.0 7.0)
    let diff: Vector2D = (vector_to a b)
    assert (== diff.x 3.0)
    assert (== diff.y 4.0)
}
```

### `vec_scale(v, scalar)` — Multiply by a scalar

Scaling velocity by time delta:

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_scale

fn apply_delta(vel: Vector2D, dt: float) -> Vector2D {
    return (vec_scale vel dt)
}

shadow apply_delta {
    let vel: Vector2D = (vec_new 100.0 50.0)
    let moved: Vector2D = (apply_delta vel 0.016)
    assert (> moved.x 1.5)
    assert (< moved.x 1.7)
}
```

## Length and Distance

### `vec_length(v)` — Magnitude (Euclidean length)

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_length

fn speed_from_velocity(vel: Vector2D) -> float {
    return (vec_length vel)
}

shadow speed_from_velocity {
    let vel: Vector2D = (vec_new 3.0 4.0)
    assert (== (speed_from_velocity vel) 5.0)
}
```

### `vec_length_squared(v)` — Magnitude squared (no sqrt)

Use this for comparisons to avoid the cost of `sqrt`:

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_length_squared

fn is_within_range_squared(pos: Vector2D, center: Vector2D, range: float) -> bool {
    from "modules/vector2d/vector2d.nano" import vec_sub
    let diff: Vector2D = (vec_sub pos center)
    return (<= (vec_length_squared diff) (* range range))
}

shadow is_within_range_squared {
    let pos: Vector2D = (vec_new 3.0 4.0)
    let center: Vector2D = (vec_new 0.0 0.0)
    assert (is_within_range_squared pos center 6.0)
    assert (not (is_within_range_squared pos center 4.0))
}
```

### `vec_distance(a, b)` — Distance between two points

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_distance

fn is_colliding(a: Vector2D, b: Vector2D, min_dist: float) -> bool {
    return (< (vec_distance a b) min_dist)
}

shadow is_colliding {
    let a: Vector2D = (vec_new 0.0 0.0)
    let b: Vector2D = (vec_new 3.0 4.0)
    assert (is_colliding a b 6.0)
    assert (not (is_colliding a b 4.0))
}
```

### `vec_distance_squared(a, b)` — Distance squared (faster)

Prefer this when comparing distances, e.g. finding the nearest enemy.

## Direction and Normalization

### `vec_normalize(v)` — Make length 1

Returns a **unit vector** pointing in the same direction. If the vector has zero length, returns the zero vector safely.

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_normalize, vec_length

fn aim_direction(from_pos: Vector2D, to_pos: Vector2D) -> Vector2D {
    from "modules/vector2d/vector2d.nano" import vec_sub
    let raw: Vector2D = (vec_sub to_pos from_pos)
    return (vec_normalize raw)
}

shadow aim_direction {
    let shooter: Vector2D = (vec_new 0.0 0.0)
    let target: Vector2D = (vec_new 3.0 4.0)
    let dir: Vector2D = (aim_direction shooter target)
    let len: float = (vec_length dir)
    assert (> len 0.99)
    assert (< len 1.01)
}
```

### `vec_dot(a, b)` — Dot product

The dot product is the foundation of many game calculations:
- Positive means vectors point in a similar direction
- Zero means they are perpendicular
- Negative means they point away from each other

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_normalize, vec_dot

fn is_in_front(facing: Vector2D, to_target: Vector2D) -> bool {
    let facing_n: Vector2D = (vec_normalize facing)
    let target_n: Vector2D = (vec_normalize to_target)
    return (> (vec_dot facing_n target_n) 0.0)
}

shadow is_in_front {
    let facing: Vector2D = (vec_new 1.0 0.0)
    let ahead: Vector2D = (vec_new 5.0 0.0)
    let behind: Vector2D = (vec_new -5.0 0.0)
    assert (is_in_front facing ahead)
    assert (not (is_in_front facing behind))
}
```

### `vec_to_angle(v)` — Get angle from vector (radians)

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_to_angle

fn rotation_degrees(vel: Vector2D) -> float {
    let radians: float = (vec_to_angle vel)
    return (* radians 57.2958)   # radians to degrees
}

shadow rotation_degrees {
    let right: Vector2D = (vec_new 1.0 0.0)
    assert (< (abs (rotation_degrees right)) 0.01)
}
```

## Advanced Operations

### `vec_rotate(v, angle)` — Rotate by angle (radians)

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_rotate

fn rotate_velocity_90_degrees(vel: Vector2D) -> Vector2D {
    let pi_over_2: float = 1.5708
    return (vec_rotate vel pi_over_2)
}

shadow rotate_velocity_90_degrees {
    let right: Vector2D = (vec_new 1.0 0.0)
    let up: Vector2D = (rotate_velocity_90_degrees right)
    assert (< (abs up.x) 0.01)
    assert (> up.y 0.99)
}
```

### `vec_lerp(a, b, t)` — Linear interpolation

Smoothly interpolate between two positions. `t = 0.0` returns `a`, `t = 1.0` returns `b`.

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_lerp

fn smooth_camera(camera: Vector2D, target: Vector2D, smoothing: float) -> Vector2D {
    return (vec_lerp camera target smoothing)
}

shadow smooth_camera {
    let cam: Vector2D = (vec_new 0.0 0.0)
    let tgt: Vector2D = (vec_new 10.0 0.0)
    let result: Vector2D = (smooth_camera cam tgt 0.1)
    assert (== result.x 1.0)
    assert (== result.y 0.0)
}
```

### `vec_reflect(v, normal)` — Reflect off a surface

Compute the reflection of a velocity vector off a surface with the given normal. Used for bouncing projectiles.

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_reflect

fn bounce_off_floor(vel: Vector2D) -> Vector2D {
    let floor_normal: Vector2D = (vec_new 0.0 1.0)
    return (vec_reflect vel floor_normal)
}

shadow bounce_off_floor {
    let falling: Vector2D = (vec_new 1.0 -1.0)
    let bounced: Vector2D = (bounce_off_floor falling)
    assert (== bounced.x 1.0)
    assert (== bounced.y 1.0)
}
```

### `vec_perp(v)` — Perpendicular vector

Returns the vector rotated 90 degrees counterclockwise.

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_perp

fn strafe_direction(facing: Vector2D) -> Vector2D {
    return (vec_perp facing)
}

shadow strafe_direction {
    let forward: Vector2D = (vec_new 1.0 0.0)
    let strafe: Vector2D = (strafe_direction forward)
    assert (== strafe.x 0.0)
    assert (== strafe.y 1.0)
}
```

### `vec_clamp_length(v, max_len)` — Clamp to maximum speed

Cap a velocity vector so it never exceeds a maximum speed:

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_clamp_length, vec_length

fn limit_speed(vel: Vector2D, max_speed: float) -> Vector2D {
    return (vec_clamp_length vel max_speed)
}

shadow limit_speed {
    let fast: Vector2D = (vec_new 100.0 100.0)
    let limited: Vector2D = (limit_speed fast 10.0)
    let len: float = (vec_length limited)
    assert (< len 10.01)
    assert (> len 9.99)
}
```

## Full Function Reference

| Function | Signature | Description |
|---|---|---|
| `vec_new` | `(x: float, y: float) -> Vector2D` | Create vector from components |
| `vec_zero` | `() -> Vector2D` | Zero vector (0, 0) |
| `vec_add` | `(a b: Vector2D) -> Vector2D` | Component-wise addition |
| `vec_sub` | `(a b: Vector2D) -> Vector2D` | Component-wise subtraction |
| `vec_scale` | `(v: Vector2D, s: float) -> Vector2D` | Multiply by scalar |
| `vec_dot` | `(a b: Vector2D) -> float` | Dot product |
| `vec_length` | `(v: Vector2D) -> float` | Euclidean length |
| `vec_length_squared` | `(v: Vector2D) -> float` | Length squared (no sqrt) |
| `vec_distance` | `(a b: Vector2D) -> float` | Distance between two points |
| `vec_distance_squared` | `(a b: Vector2D) -> float` | Distance squared (no sqrt) |
| `vec_normalize` | `(v: Vector2D) -> Vector2D` | Unit vector (length 1) |
| `vec_rotate` | `(v: Vector2D, angle: float) -> Vector2D` | Rotate by radians |
| `vec_from_angle` | `(angle: float) -> Vector2D` | Unit vector from angle |
| `vec_to_angle` | `(v: Vector2D) -> float` | Angle from vector (atan2) |
| `vec_lerp` | `(a b: Vector2D, t: float) -> Vector2D` | Linear interpolation |
| `vec_clamp_length` | `(v: Vector2D, max: float) -> Vector2D` | Cap magnitude |
| `vec_perp` | `(v: Vector2D) -> Vector2D` | 90-degree rotation |
| `vec_reflect` | `(v normal: Vector2D) -> Vector2D` | Reflection off surface |

---

**Previous:** [18.1 event](event.html)
**Next:** [18.3 bullet](bullet.html)
