# vector2d API Reference

The `vector2d` module provides 2D vector mathematics for game development, physics simulations, and any application requiring geometric computation. Import it with:

```nano
import "modules/vector2d/vector2d"
```

## Types

### Vector2D

```nano
struct Vector2D {
    x: float,
    y: float
}
```

Represents a 2D vector or point in space. Structs are immutable in NanoLang — all operations return a new `Vector2D` rather than modifying in place.

---

## Functions

### vec_new

```nano
fn vec_new(x: float, y: float) -> Vector2D
```

Creates a new `Vector2D` with the given components.

**Parameters:**
- `x` — the x component
- `y` — the y component

**Returns:** A new `Vector2D` with the specified x and y values.

**Example:**
```nano
let v: Vector2D = (vec_new 3.0 4.0)
```

---

### vec_zero

```nano
fn vec_zero() -> Vector2D
```

Returns the zero vector `(0.0, 0.0)`. Useful as a default or identity value.

**Returns:** `Vector2D { x: 0.0, y: 0.0 }`

**Example:**
```nano
let origin: Vector2D = (vec_zero)
```

---

### vec_add

```nano
fn vec_add(a: Vector2D, b: Vector2D) -> Vector2D
```

Adds two vectors component-wise.

**Parameters:**
- `a` — the first vector
- `b` — the second vector

**Returns:** A new vector where each component is the sum of the corresponding components of `a` and `b`.

**Example:**
```nano
let a: Vector2D = (vec_new 1.0 2.0)
let b: Vector2D = (vec_new 3.0 4.0)
let c: Vector2D = (vec_add a b)  # c = (4.0, 6.0)
```

---

### vec_sub

```nano
fn vec_sub(a: Vector2D, b: Vector2D) -> Vector2D
```

Subtracts vector `b` from vector `a` component-wise.

**Parameters:**
- `a` — the minuend vector
- `b` — the subtrahend vector

**Returns:** A new vector equal to `a - b`.

**Example:**
```nano
let a: Vector2D = (vec_new 5.0 7.0)
let b: Vector2D = (vec_new 2.0 3.0)
let c: Vector2D = (vec_sub a b)  # c = (3.0, 4.0)
```

---

### vec_scale

```nano
fn vec_scale(v: Vector2D, s: float) -> Vector2D
```

Multiplies a vector by a scalar value.

**Parameters:**
- `v` — the vector to scale
- `s` — the scalar multiplier

**Returns:** A new vector with each component multiplied by `s`.

**Example:**
```nano
let v: Vector2D = (vec_new 2.0 3.0)
let scaled: Vector2D = (vec_scale v 2.0)  # scaled = (4.0, 6.0)
```

---

### vec_dot

```nano
fn vec_dot(a: Vector2D, b: Vector2D) -> float
```

Computes the dot product of two vectors. The dot product is zero when the vectors are perpendicular, positive when they point in the same general direction, and negative when they point in opposite directions.

**Parameters:**
- `a` — the first vector
- `b` — the second vector

**Returns:** The scalar dot product `a.x * b.x + a.y * b.y`.

**Example:**
```nano
let a: Vector2D = (vec_new 2.0 3.0)
let b: Vector2D = (vec_new 4.0 5.0)
let d: float = (vec_dot a b)  # d = 23.0
```

---

### vec_length

```nano
fn vec_length(v: Vector2D) -> float
```

Computes the magnitude (Euclidean length) of a vector.

**Parameters:**
- `v` — the vector

**Returns:** The length `sqrt(v.x^2 + v.y^2)`.

**Example:**
```nano
let v: Vector2D = (vec_new 3.0 4.0)
let len: float = (vec_length v)  # len = 5.0
```

---

### vec_length_squared

```nano
fn vec_length_squared(v: Vector2D) -> float
```

Computes the squared magnitude of a vector. Faster than `vec_length` because it avoids a square root. Useful when comparing distances (compare squared values rather than taking square roots).

**Parameters:**
- `v` — the vector

**Returns:** `v.x^2 + v.y^2`

**Example:**
```nano
let v: Vector2D = (vec_new 3.0 4.0)
let len_sq: float = (vec_length_squared v)  # len_sq = 25.0
```

---

### vec_distance

```nano
fn vec_distance(a: Vector2D, b: Vector2D) -> float
```

Computes the Euclidean distance between two points.

**Parameters:**
- `a` — the first point
- `b` — the second point

**Returns:** The distance between `a` and `b`.

**Example:**
```nano
let a: Vector2D = (vec_new 0.0 0.0)
let b: Vector2D = (vec_new 3.0 4.0)
let dist: float = (vec_distance a b)  # dist = 5.0
```

---

### vec_distance_squared

```nano
fn vec_distance_squared(a: Vector2D, b: Vector2D) -> float
```

Computes the squared distance between two points. Faster than `vec_distance` for comparison purposes.

**Parameters:**
- `a` — the first point
- `b` — the second point

**Returns:** The squared distance between `a` and `b`.

**Example:**
```nano
let a: Vector2D = (vec_new 0.0 0.0)
let b: Vector2D = (vec_new 3.0 4.0)
let dist_sq: float = (vec_distance_squared a b)  # dist_sq = 25.0
```

---

### vec_normalize

```nano
fn vec_normalize(v: Vector2D) -> Vector2D
```

Returns a unit vector (length = 1.0) pointing in the same direction as `v`. If `v` is the zero vector, returns the zero vector.

**Parameters:**
- `v` — the vector to normalize

**Returns:** A vector with the same direction as `v` and length 1.0, or the zero vector if `v` has zero length.

**Example:**
```nano
let v: Vector2D = (vec_new 3.0 4.0)
let n: Vector2D = (vec_normalize v)  # n = (0.6, 0.8)
```

---

### vec_rotate

```nano
fn vec_rotate(v: Vector2D, angle: float) -> Vector2D
```

Rotates a vector by the given angle in radians, counter-clockwise.

**Parameters:**
- `v` — the vector to rotate
- `angle` — the rotation angle in radians

**Returns:** A new vector rotated by `angle` radians.

**Example:**
```nano
let v: Vector2D = (vec_new 1.0 0.0)
let pi_over_2: float = 1.5708
let rotated: Vector2D = (vec_rotate v pi_over_2)  # approximately (0.0, 1.0)
```

---

### vec_from_angle

```nano
fn vec_from_angle(angle: float) -> Vector2D
```

Creates a unit vector pointing in the direction specified by the given angle in radians. This is the inverse of `vec_to_angle`.

**Parameters:**
- `angle` — the angle in radians (0 = right, pi/2 = up)

**Returns:** A unit vector `(cos(angle), sin(angle))`.

**Example:**
```nano
let v: Vector2D = (vec_from_angle 0.0)  # v = (1.0, 0.0)
```

---

### vec_to_angle

```nano
fn vec_to_angle(v: Vector2D) -> float
```

Returns the angle of a vector in radians, measured counter-clockwise from the positive x-axis. Uses `atan2` internally.

**Parameters:**
- `v` — the vector

**Returns:** The angle in radians in the range `(-pi, pi]`.

**Example:**
```nano
let v: Vector2D = (vec_new 1.0 0.0)
let angle: float = (vec_to_angle v)  # angle ≈ 0.0
```

---

### vec_lerp

```nano
fn vec_lerp(a: Vector2D, b: Vector2D, t: float) -> Vector2D
```

Linearly interpolates between two vectors. At `t = 0.0` returns `a`; at `t = 1.0` returns `b`; at `t = 0.5` returns the midpoint.

**Parameters:**
- `a` — the start vector
- `b` — the end vector
- `t` — the interpolation factor (typically in the range `[0.0, 1.0]`)

**Returns:** The interpolated vector `a * (1 - t) + b * t`.

**Example:**
```nano
let a: Vector2D = (vec_new 0.0 0.0)
let b: Vector2D = (vec_new 10.0 10.0)
let mid: Vector2D = (vec_lerp a b 0.5)  # mid = (5.0, 5.0)
```

---

### vec_clamp_length

```nano
fn vec_clamp_length(v: Vector2D, max_len: float) -> Vector2D
```

Clamps the length of a vector to at most `max_len`. If the vector is already shorter than `max_len`, it is returned unchanged. Useful for capping velocity or force magnitudes.

**Parameters:**
- `v` — the vector to clamp
- `max_len` — the maximum allowed length

**Returns:** The original vector if its length is within `max_len`, or a vector with the same direction but length equal to `max_len`.

**Example:**
```nano
let v: Vector2D = (vec_new 30.0 40.0)  # length = 50
let clamped: Vector2D = (vec_clamp_length v 10.0)  # length = 10
```

---

### vec_perp

```nano
fn vec_perp(v: Vector2D) -> Vector2D
```

Returns a vector perpendicular to `v`, rotated 90 degrees counter-clockwise.

**Parameters:**
- `v` — the input vector

**Returns:** A vector perpendicular to `v`: `(-v.y, v.x)`.

**Example:**
```nano
let v: Vector2D = (vec_new 1.0 0.0)
let perp: Vector2D = (vec_perp v)  # perp = (0.0, 1.0)
```

---

### vec_reflect

```nano
fn vec_reflect(v: Vector2D, normal: Vector2D) -> Vector2D
```

Reflects a vector off a surface defined by a normal vector. The normal should be a unit vector for correct results.

**Parameters:**
- `v` — the incident vector to reflect
- `normal` — the surface normal (should be normalized)

**Returns:** The reflected vector.

**Example:**
```nano
let v: Vector2D = (vec_new 1.0 -1.0)
let normal: Vector2D = (vec_new 0.0 1.0)
let reflected: Vector2D = (vec_reflect v normal)  # reflected = (1.0, 1.0)
```
