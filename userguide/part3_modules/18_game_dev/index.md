# Chapter 18: Game Development

**Build games with event systems, 2D math, and physics.**

Game development in NanoLang brings together three complementary modules: an event system for driving your game loop, a 2D vector math library for movement and collision geometry, and a Bullet physics engine for realistic simulation. Together they give you the building blocks for everything from arcade games to physics sandboxes.

## What You'll Learn

- How to structure a game loop using the `event` module's libevent-backed dispatch system
- How to do 2D math (position, velocity, direction) using the `vector2d` module's `Vector2D` type
- How to add physical simulation—rigid bodies and soft bodies—using the `bullet` module
- How all three work together in a complete runnable example

## The Three Modules at a Glance

### event — Asynchronous Event Loop

The `event` module wraps libevent, a cross-platform asynchronous event notification library. In a game context you typically use it to:

- Create an **event base** (the event loop) that drives your game tick
- Register **timer events** to fire at a fixed interval (e.g., 16 ms for 60 fps)
- Use `nl_event_base_dispatch` to enter the loop and `nl_event_base_loopexit` or `nl_event_base_loopbreak` to exit it

```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_dispatch,
                                        nl_event_base_free, nl_event_base_loopbreak,
                                        nl_event_get_version
```

### vector2d — 2D Vector Mathematics

The `vector2d` module provides a `Vector2D` struct and a complete set of pure functions for working with 2D positions, velocities, and directions. All operations return new vectors rather than mutating their inputs, which makes it easy to reason about game state.

```nano
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_add, vec_sub,
                                             vec_scale, vec_normalize, vec_length,
                                             vec_distance, vec_lerp, vec_dot
```

### bullet — Physics Simulation

The `bullet` module exposes the Bullet Physics engine through a lightweight FFI layer. You can create a physics world, populate it with rigid spheres and boxes (as well as soft spheres), step the simulation forward in time, and read back the positions and orientations of every body.

```nano
from "modules/bullet/bullet.nano" import nl_bullet_init, nl_bullet_cleanup,
                                         nl_bullet_step, nl_bullet_set_gravity,
                                         nl_bullet_create_rigid_sphere,
                                         nl_bullet_create_rigid_box,
                                         nl_bullet_get_rigid_body_x,
                                         nl_bullet_get_rigid_body_y,
                                         nl_bullet_get_rigid_body_z
```

## Typical Game Architecture in NanoLang

A NanoLang game follows a straightforward structure:

```
main()
  └── initialise all subsystems
        ├── nl_bullet_init()            # physics world
        ├── nl_event_base_new()         # event loop
        └── load game state
  └── register a timer event (game tick)
  └── nl_event_base_dispatch()          # blocks until game exits
  └── cleanup
        ├── nl_bullet_cleanup()
        └── nl_event_base_free()
```

Your game logic lives inside the timer callback that fires every frame. On each tick you:

1. Read player input (or process any queued events)
2. Update positions using `vec_add`, `vec_scale`, etc.
3. Step the physics simulation with `nl_bullet_step`
4. Read back updated positions from Bullet and render them

### Minimal Game Loop Skeleton

```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_dispatch,
                                        nl_event_base_free, nl_evtimer_new,
                                        nl_evtimer_add_timeout, nl_event_free,
                                        nl_event_base_loopbreak
from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_add, vec_scale, vec_zero
from "modules/bullet/bullet.nano" import nl_bullet_init, nl_bullet_cleanup,
                                         nl_bullet_step, nl_bullet_set_gravity,
                                         nl_bullet_create_rigid_sphere,
                                         nl_bullet_get_rigid_body_x,
                                         nl_bullet_get_rigid_body_y

# --- Game state (mutable globals via let mut) ---
let mut player_pos: Vector2D = (vec_zero)
let mut player_vel: Vector2D = (vec_zero)
let mut frame_count: int = 0
let mut sphere_handle: int = 0

fn game_tick(base: int) -> void {
    set frame_count (+ frame_count 1)

    # 1. Update player position from velocity
    set player_pos (vec_add player_pos player_vel)

    # 2. Step physics at 1/60 second
    (nl_bullet_step 0.01666)

    # 3. Read physics body position
    let phys_x: float = (nl_bullet_get_rigid_body_x sphere_handle)
    let phys_y: float = (nl_bullet_get_rigid_body_y sphere_handle)

    # 4. Render / update display (call your rendering module here)

    # 5. Exit after 300 frames (5 seconds at 60 fps)
    if (>= frame_count 300) {
        (nl_event_base_loopbreak base)
    } else {
        (print "")   # no-op else branch required by language
    }
}

fn main() -> int {
    # Initialise physics
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)

    # Create a falling sphere at height 10
    set sphere_handle (nl_bullet_create_rigid_sphere 0.0 10.0 0.0 1.0 1.0 0.5)

    # Set initial player velocity
    set player_vel (vec_new 0.1 0.0)

    # Create the event loop
    let base: int = (nl_event_base_new)
    let timer: int = (nl_evtimer_new base)
    (nl_evtimer_add_timeout timer 0)   # fire immediately, re-schedule each tick

    (nl_event_base_dispatch base)

    # Cleanup
    (nl_event_free timer)
    (nl_event_base_free base)
    (nl_bullet_cleanup)

    return 0
}

shadow main { assert true }
```

> **Note:** The timer callback signature shown here is illustrative. In practice, libevent callbacks receive the event base as context. See [Section 18.1](event.html) for the full event callback pattern.

## Module Design Philosophy

Each module is intentionally minimal and orthogonal:

- `vector2d` is **pure** — no global state, no I/O. Every function takes values and returns new values. This makes it trivial to test.
- `bullet` is **stateful but contained** — the physics world is a singleton managed by the C library. Call `nl_bullet_init` once, `nl_bullet_cleanup` once, and never mix multiple worlds.
- `event` is **infrastructure** — it provides the heartbeat of your game but does not impose any particular game-object model. You decide what happens each tick.

---

**Sections:**
- [18.1 event — Event System](event.html)
- [18.2 vector2d — 2D Math](vector2d.html)
- [18.3 bullet — Physics Engine](bullet.html)
- [18.4 Example: Building a Simple Game](simple_game.html)

---

**Previous:** [Chapter 17: OpenGL Graphics](../17_opengl/index.html)
**Next:** [18.1 event](event.html)
