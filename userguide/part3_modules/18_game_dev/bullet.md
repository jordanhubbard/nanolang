# 18.3 bullet — Physics Engine

**Real-time rigid body and soft body physics via Bullet Physics.**

The `bullet` module gives NanoLang programs access to the [Bullet Physics](https://github.com/bulletphysics/bullet3) engine through a thin FFI layer. You get a full discrete-dynamics world: rigid bodies with mass, restitution, and gravity; soft deformable spheres; and a step function that advances the simulation forward in time.

All body handles are `int` values returned by the creation functions. Pass them back to the query functions to read positions and orientations each frame.

## Quick Start

```nano
from "modules/bullet/bullet.nano" import nl_bullet_init, nl_bullet_cleanup,
                                         nl_bullet_step, nl_bullet_set_gravity,
                                         nl_bullet_create_rigid_sphere,
                                         nl_bullet_get_rigid_body_x,
                                         nl_bullet_get_rigid_body_y,
                                         nl_bullet_get_rigid_body_z

fn run_physics_demo() -> void {
    # 1. Initialise the physics world
    (nl_bullet_init)

    # 2. Set gravity (Earth-like, pointing down in Y)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)

    # 3. Create a ball at height 10, radius 1, mass 1 kg, bouncy
    let ball: int = (nl_bullet_create_rigid_sphere 0.0 10.0 0.0 1.0 1.0 0.7)

    # 4. Simulate 60 frames at 1/60 second each
    let mut frame: int = 0
    while (< frame 60) {
        (nl_bullet_step 0.01666)
        let y: float = (nl_bullet_get_rigid_body_y ball)
        (println (+ "y=" (float_to_string y)))
        set frame (+ frame 1)
    }

    # 5. Clean up
    (nl_bullet_cleanup)
}

shadow run_physics_demo {
    (run_physics_demo)
}
```

## World Management

### Initialising

```
nl_bullet_init() -> int
```

Creates the physics world (broadphase, dispatcher, solver, collision config). Must be called once before any other bullet function. Returns `1` on success.

```nano
(nl_bullet_init)
```

### Setting Gravity

```
nl_bullet_set_gravity(gx: float, gy: float, gz: float) -> void
```

Set the global gravity vector. For a side-scrolling 2D game in the XY plane, a typical call is:

```nano
(nl_bullet_set_gravity 0.0 -9.8 0.0)
```

For zero-gravity (space), pass all zeros:

```nano
(nl_bullet_set_gravity 0.0 0.0 0.0)
```

### Stepping the Simulation

```
nl_bullet_step(time_step: float) -> void
```

Advance the simulation by `time_step` seconds. Call this once per frame. For a 60 fps game:

```nano
(nl_bullet_step 0.01666)   # 1/60 seconds
```

For a 30 fps game:

```nano
(nl_bullet_step 0.03333)   # 1/30 seconds
```

### Cleaning Up

```
nl_bullet_cleanup() -> void
```

Destroy the physics world and free all Bullet resources. Call this when your game exits or when you tear down the physics subsystem.

```nano
(nl_bullet_cleanup)
```

## Creating Rigid Bodies

Rigid bodies are solid, undeformable objects. Each creation function returns an `int` handle you use to query the body's state each frame.

### Rigid Sphere

```
nl_bullet_create_rigid_sphere(x y z: float, radius: float, mass: float, restitution: float) -> int
```

| Parameter | Description |
|---|---|
| `x, y, z` | Initial position in world space |
| `radius` | Sphere radius |
| `mass` | Mass in kg. Use `0.0` for a static (immovable) sphere |
| `restitution` | Bounciness: `0.0` = no bounce, `1.0` = perfectly elastic |

```nano
# A dynamic ball that falls and bounces
let ball: int = (nl_bullet_create_rigid_sphere 0.0 5.0 0.0 0.5 1.0 0.8)

# A static floor (mass = 0 means immovable)
let floor: int = (nl_bullet_create_rigid_sphere 0.0 -1.0 0.0 10.0 0.0 0.3)
```

### Rigid Box

```
nl_bullet_create_rigid_box(x y z: float,
                            half_width half_height half_depth: float,
                            mass: float, restitution: float) -> int
```

Sizes are **half-extents**: a box with `half_width = 1.0` is 2 units wide total.

```nano
# A crate: 2x2x2 units, mass 5 kg, slightly bouncy
let crate: int = (nl_bullet_create_rigid_box 0.0 2.0 0.0 1.0 1.0 1.0 5.0 0.3)

# A flat platform: 20 units wide, 0.5 units tall, static
let platform: int = (nl_bullet_create_rigid_box 0.0 0.0 0.0 10.0 0.25 1.0 0.0 0.1)
```

### Rotated Rigid Box

```
nl_bullet_create_rigid_box_rotated(x y z: float,
                                    half_width half_height half_depth: float,
                                    angle_degrees: float,
                                    mass: float, restitution: float) -> int
```

Like `create_rigid_box` but pre-rotated by `angle_degrees` around the Z-axis. Useful for ramps and sloped terrain.

```nano
# A 45-degree ramp
let ramp: int = (nl_bullet_create_rigid_box_rotated 5.0 0.0 0.0 3.0 0.2 1.0 45.0 0.0 0.1)
```

## Querying Rigid Body State

After each `nl_bullet_step`, read the updated position and orientation of any rigid body.

### Position

```
nl_bullet_get_rigid_body_x(handle: int) -> float
nl_bullet_get_rigid_body_y(handle: int) -> float
nl_bullet_get_rigid_body_z(handle: int) -> float
```

```nano
let x: float = (nl_bullet_get_rigid_body_x ball)
let y: float = (nl_bullet_get_rigid_body_y ball)
let z: float = (nl_bullet_get_rigid_body_z ball)
```

### Orientation (Quaternion)

The rotation is returned as a unit quaternion (x, y, z, w).

```
nl_bullet_get_rigid_body_rot_x(handle: int) -> float
nl_bullet_get_rigid_body_rot_y(handle: int) -> float
nl_bullet_get_rigid_body_rot_z(handle: int) -> float
nl_bullet_get_rigid_body_rot_w(handle: int) -> float
```

```nano
let qx: float = (nl_bullet_get_rigid_body_rot_x crate)
let qy: float = (nl_bullet_get_rigid_body_rot_y crate)
let qz: float = (nl_bullet_get_rigid_body_rot_z crate)
let qw: float = (nl_bullet_get_rigid_body_rot_w crate)
```

For 2D games where you only care about the Z rotation angle, compute it from the quaternion:

```nano
# Approximate Z angle in radians from quaternion (valid for small X/Y rotations)
fn quat_to_angle_z(qz: float, qw: float) -> float {
    return (* 2.0 (atan2 qz qw))
}

shadow quat_to_angle_z {
    # Identity quaternion (no rotation) -> angle 0
    let angle: float = (quat_to_angle_z 0.0 1.0)
    assert (< (abs angle) 0.01)
}
```

### Body Count

```
nl_bullet_get_rigid_body_count() -> int
```

Returns the number of rigid bodies currently in the world.

```nano
let count: int = (nl_bullet_get_rigid_body_count)
(println (+ "Bodies in world: " (int_to_string count)))
```

## Soft Bodies

Soft bodies are deformable meshes. Currently only soft spheres are supported.

### Creating a Soft Sphere

```
nl_bullet_create_soft_sphere(x y z: float, radius: float, resolution: int) -> int
```

`resolution` controls the tessellation: higher values give more nodes and a smoother deformation but cost more per step.

```nano
let blob: int = (nl_bullet_create_soft_sphere 0.0 5.0 0.0 1.0 8)
```

### Querying Soft Body Nodes

After each step you can read the position of every node in the soft mesh:

```
nl_bullet_get_soft_body_node_count(handle: int) -> int
nl_bullet_get_soft_body_node_x(handle: int, node_idx: int) -> float
nl_bullet_get_soft_body_node_y(handle: int, node_idx: int) -> float
nl_bullet_get_soft_body_node_z(handle: int, node_idx: int) -> float
```

```nano
fn print_blob_nodes(blob: int) -> void {
    let n: int = (nl_bullet_get_soft_body_node_count blob)
    let mut i: int = 0
    while (< i n) {
        let nx: float = (nl_bullet_get_soft_body_node_x blob i)
        let ny: float = (nl_bullet_get_soft_body_node_y blob i)
        (println (+ (+ "node " (int_to_string i)) (+ " y=" (float_to_string ny))))
        set i (+ i 1)
    }
}

shadow print_blob_nodes {
    (nl_bullet_init)
    let blob: int = (nl_bullet_create_soft_sphere 0.0 2.0 0.0 1.0 4)
    (nl_bullet_step 0.016)
    (print_blob_nodes blob)
    (nl_bullet_cleanup)
}
```

### Removing a Soft Body

```
nl_bullet_remove_soft_body(handle: int) -> void
```

Removes the soft body from the simulation. Rigid bodies cannot be individually removed once added — they persist until `nl_bullet_cleanup`.

```nano
(nl_bullet_remove_soft_body blob)
```

### Soft Body Count

```
nl_bullet_get_soft_body_count() -> int
```

## Full Example: Stacked Boxes

```nano
from "modules/bullet/bullet.nano" import nl_bullet_init, nl_bullet_cleanup,
                                         nl_bullet_step, nl_bullet_set_gravity,
                                         nl_bullet_create_rigid_box,
                                         nl_bullet_get_rigid_body_y,
                                         nl_bullet_get_rigid_body_count

fn simulate_stack() -> void {
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)

    # Static ground plane
    let ground: int = (nl_bullet_create_rigid_box 0.0 -0.5 0.0 20.0 0.5 1.0 0.0 0.2)

    # Three boxes stacked above the ground
    let box1: int = (nl_bullet_create_rigid_box 0.0 1.0 0.0 1.0 1.0 1.0 2.0 0.3)
    let box2: int = (nl_bullet_create_rigid_box 0.0 3.0 0.0 1.0 1.0 1.0 2.0 0.3)
    let box3: int = (nl_bullet_create_rigid_box 0.0 5.0 0.0 1.0 1.0 1.0 2.0 0.3)

    (println (+ "Bodies: " (int_to_string (nl_bullet_get_rigid_body_count))))

    # Simulate for 120 frames
    let mut frame: int = 0
    while (< frame 120) {
        (nl_bullet_step 0.01666)
        set frame (+ frame 1)
    }

    # Print final heights
    (println (+ "box1 y=" (float_to_string (nl_bullet_get_rigid_body_y box1))))
    (println (+ "box2 y=" (float_to_string (nl_bullet_get_rigid_body_y box2))))
    (println (+ "box3 y=" (float_to_string (nl_bullet_get_rigid_body_y box3))))

    (nl_bullet_cleanup)
}

shadow simulate_stack {
    (simulate_stack)
}
```

---

**Previous:** [18.2 vector2d](vector2d.html)
**Next:** [18.4 Simple Game Tutorial](simple_game.html)
