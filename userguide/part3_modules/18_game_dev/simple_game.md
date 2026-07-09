# 18.4 Example: Building a Simple Game

**A complete walkthrough: a minimal 2D game combining events, vectors, and physics.**

This tutorial builds a small but complete game skeleton step by step. We won't use a graphical renderer (that requires linking SDL or OpenGL which is covered in other chapters), but we will produce a runnable terminal simulation you can actually compile and run. By the end you will have:

- A fixed-rate game loop driven by the `event` module
- A player character whose position is tracked with `vector2d`
- A physics object (a falling ball) simulated by `bullet`
- Simple "collision" detection using vector distance
- A clean setup and teardown path

## Game Design

Our game is called **Ball Drop**:

- A ball falls from above under gravity
- The player is a point in 2D space that can move horizontally
- If the player is within 1 unit of the ball when it hits the "ground" (y ≤ 0), they score a point
- The game runs for 5 seconds then prints the score

All positions use the `Vector2D` type. The ball's Y position comes from Bullet physics. The player's X position is controlled by a simple oscillating strategy (in a real game you would read keyboard input here).

## Step 1: Imports and Global State

NanoLang modules are fully imported at the top level. Mutable game state is declared as top-level `let mut` bindings.

```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_dispatch,
                                        nl_event_base_free, nl_event_base_loopbreak,
                                        nl_evtimer_new, nl_evtimer_add_timeout,
                                        nl_event_free

from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_zero,
                                             vec_add, vec_scale, vec_distance

from "modules/bullet/bullet.nano" import nl_bullet_init, nl_bullet_cleanup,
                                         nl_bullet_step, nl_bullet_set_gravity,
                                         nl_bullet_create_rigid_sphere,
                                         nl_bullet_get_rigid_body_x,
                                         nl_bullet_get_rigid_body_y

# --- Mutable game state ---
let mut player_pos: Vector2D = (vec_zero)
let mut score: int = 0
let mut frame_count: int = 0
let mut ball_handle: int = 0
let mut game_base: int = 0
let mut game_timer: int = 0

let GAME_FRAMES: int = 300      # 5 seconds at 60 fps
let CATCH_RADIUS: float = 1.0   # how close the player must be
let BALL_START_Y: float = 20.0  # initial ball height
let PLAYER_SPEED: float = 0.08  # horizontal speed per frame
```

## Step 2: Player Update

The player moves horizontally. In this demo the player oscillates left and right, simulating an AI that is trying to be under the ball. In a real game you would map this to keyboard input.

```nano
fn update_player(frame: int) -> Vector2D {
    # Oscillate horizontally: move right for first half-period, left for second
    let period: int = 120   # 2-second oscillation
    let phase: int = (% frame period)
    if (< phase 60) {
        return (vec_add player_pos (vec_new PLAYER_SPEED 0.0))
    } else {
        return (vec_add player_pos (vec_new (- 0.0 PLAYER_SPEED) 0.0))
    }
}

shadow update_player {
    set player_pos (vec_zero)
    set frame_count 0
    let p: Vector2D = (update_player 0)
    assert (> p.x 0.0)
}
```

## Step 3: Collision Check

After stepping physics we check whether the ball has reached the ground and whether the player is nearby:

```nano
fn check_catch(ball_x: float, ball_y: float) -> bool {
    if (> ball_y 0.5) {
        return false    # Ball not yet near ground
    } else {
        let ball_pos: Vector2D = (vec_new ball_x ball_y)
        let dist: float = (vec_distance player_pos ball_pos)
        return (< dist CATCH_RADIUS)
    }
}

shadow check_catch {
    set player_pos (vec_new 1.0 0.0)
    assert (check_catch 1.3 0.2)    # close enough
    assert (not (check_catch 5.0 0.2))  # too far
    assert (not (check_catch 1.3 5.0))  # ball still high
}
```

## Step 4: Spawning a New Ball

When the current ball hits the ground (whether caught or not), we reset it by cleaning up and reinitialising the physics world with a fresh ball:

```nano
fn spawn_ball() -> int {
    (nl_bullet_cleanup)
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)
    # Spawn at a random-ish X based on current frame count to vary difficulty
    let spawn_x: float = (* (float_from_int (% frame_count 5)) 2.0)
    return (nl_bullet_create_rigid_sphere spawn_x BALL_START_Y 0.0 0.5 1.0 0.1)
}

shadow spawn_ball {
    set frame_count 10
    let h: int = (spawn_ball)
    assert (> h 0)
    (nl_bullet_cleanup)
}
```

## Step 5: The Game Tick

The game tick is the core of the event loop. It fires every frame (we re-arm the timer to get a repeating tick), updates all state, and exits the loop when done:

```nano
fn game_tick() -> void {
    set frame_count (+ frame_count 1)

    # Update player position
    set player_pos (update_player frame_count)

    # Step physics
    (nl_bullet_step 0.01666)

    # Read ball state
    let ball_x: float = (nl_bullet_get_rigid_body_x ball_handle)
    let ball_y: float = (nl_bullet_get_rigid_body_y ball_handle)

    # Check catch condition
    if (check_catch ball_x ball_y) {
        set score (+ score 1)
        (println (+ "Caught! Score: " (int_to_string score)))
        set ball_handle (spawn_ball)
    } else {
        if (<= ball_y (- 0.0 1.0)) {
            # Ball hit ground uncaught — respawn
            (println "Missed!")
            set ball_handle (spawn_ball)
        } else {
            (print "")   # no-op else required
        }
    }

    # Exit after GAME_FRAMES frames
    if (>= frame_count GAME_FRAMES) {
        (nl_event_base_loopbreak game_base)
    } else {
        # Re-arm timer for next tick (0 = fire ASAP)
        (nl_evtimer_add_timeout game_timer 0)
    }
}

shadow game_tick {
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)
    set ball_handle (nl_bullet_create_rigid_sphere 0.0 BALL_START_Y 0.0 0.5 1.0 0.1)
    set player_pos (vec_zero)
    set frame_count 0
    set score 0
    set game_base 0
    set game_timer 0
    (game_tick)
    assert (== frame_count 1)
    (nl_bullet_cleanup)
}
```

## Step 6: Main — Putting It All Together

```nano
fn main() -> int {
    # Initialise physics
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)
    set ball_handle (nl_bullet_create_rigid_sphere 0.0 BALL_START_Y 0.0 0.5 1.0 0.1)

    # Initialise player
    set player_pos (vec_zero)

    # Initialise event loop
    set game_base (nl_event_base_new)
    set game_timer (nl_evtimer_new game_base)
    (nl_evtimer_add_timeout game_timer 0)

    (println "Ball Drop — 5 second run")

    # Run game loop (blocks until loopbreak)
    (nl_event_base_dispatch game_base)

    # Print result
    (println (+ "Game over! Final score: " (int_to_string score)))

    # Cleanup
    (nl_event_free game_timer)
    (nl_event_base_free game_base)
    (nl_bullet_cleanup)

    return 0
}

shadow main { assert true }
```

## Complete Source Code

Here is the entire program in one block, ready to compile:

```nano
from "modules/event/event.nano" import nl_event_base_new, nl_event_base_dispatch,
                                        nl_event_base_free, nl_event_base_loopbreak,
                                        nl_evtimer_new, nl_evtimer_add_timeout,
                                        nl_event_free

from "modules/vector2d/vector2d.nano" import Vector2D, vec_new, vec_zero,
                                             vec_add, vec_distance

from "modules/bullet/bullet.nano" import nl_bullet_init, nl_bullet_cleanup,
                                         nl_bullet_step, nl_bullet_set_gravity,
                                         nl_bullet_create_rigid_sphere,
                                         nl_bullet_get_rigid_body_x,
                                         nl_bullet_get_rigid_body_y

let mut player_pos: Vector2D = (vec_zero)
let mut score: int = 0
let mut frame_count: int = 0
let mut ball_handle: int = 0
let mut game_base: int = 0
let mut game_timer: int = 0

let GAME_FRAMES: int = 300
let CATCH_RADIUS: float = 1.0
let BALL_START_Y: float = 20.0
let PLAYER_SPEED: float = 0.08

fn update_player(frame: int) -> Vector2D {
    let phase: int = (% frame 120)
    if (< phase 60) {
        return (vec_add player_pos (vec_new PLAYER_SPEED 0.0))
    } else {
        return (vec_add player_pos (vec_new (- 0.0 PLAYER_SPEED) 0.0))
    }
}

shadow update_player {
    set player_pos (vec_zero)
    let p: Vector2D = (update_player 0)
    assert (> p.x 0.0)
}

fn check_catch(ball_x: float, ball_y: float) -> bool {
    if (> ball_y 0.5) {
        return false
    } else {
        let ball_pos: Vector2D = (vec_new ball_x ball_y)
        let dist: float = (vec_distance player_pos ball_pos)
        return (< dist CATCH_RADIUS)
    }
}

shadow check_catch {
    set player_pos (vec_new 1.0 0.0)
    assert (check_catch 1.3 0.2)
    assert (not (check_catch 5.0 0.2))
}

fn spawn_ball() -> int {
    (nl_bullet_cleanup)
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)
    let spawn_x: float = (* (float_from_int (% frame_count 5)) 2.0)
    return (nl_bullet_create_rigid_sphere spawn_x BALL_START_Y 0.0 0.5 1.0 0.1)
}

shadow spawn_ball {
    set frame_count 10
    let h: int = (spawn_ball)
    assert (> h 0)
    (nl_bullet_cleanup)
}

fn game_tick() -> void {
    set frame_count (+ frame_count 1)
    set player_pos (update_player frame_count)
    (nl_bullet_step 0.01666)
    let ball_x: float = (nl_bullet_get_rigid_body_x ball_handle)
    let ball_y: float = (nl_bullet_get_rigid_body_y ball_handle)
    if (check_catch ball_x ball_y) {
        set score (+ score 1)
        (println (+ "Caught! Score: " (int_to_string score)))
        set ball_handle (spawn_ball)
    } else {
        if (<= ball_y (- 0.0 1.0)) {
            (println "Missed!")
            set ball_handle (spawn_ball)
        } else {
            (print "")
        }
    }
    if (>= frame_count GAME_FRAMES) {
        (nl_event_base_loopbreak game_base)
    } else {
        (nl_evtimer_add_timeout game_timer 0)
    }
}

shadow game_tick {
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)
    set ball_handle (nl_bullet_create_rigid_sphere 0.0 BALL_START_Y 0.0 0.5 1.0 0.1)
    set player_pos (vec_zero)
    set frame_count 0
    set score 0
    set game_base 0
    set game_timer 0
    (game_tick)
    assert (== frame_count 1)
    (nl_bullet_cleanup)
}

fn main() -> int {
    (nl_bullet_init)
    (nl_bullet_set_gravity 0.0 -9.8 0.0)
    set ball_handle (nl_bullet_create_rigid_sphere 0.0 BALL_START_Y 0.0 0.5 1.0 0.1)
    set player_pos (vec_zero)
    set game_base (nl_event_base_new)
    set game_timer (nl_evtimer_new game_base)
    (nl_evtimer_add_timeout game_timer 0)
    (println "Ball Drop — 5 second run")
    (nl_event_base_dispatch game_base)
    (println (+ "Game over! Final score: " (int_to_string score)))
    (nl_event_free game_timer)
    (nl_event_base_free game_base)
    (nl_bullet_cleanup)
    return 0
}

shadow main { assert true }
```

## What to Try Next

- **Add keyboard input:** Connect to SDL or ncurses to let the player control horizontal movement.
- **Add multiple balls:** Track an array of ball handles and spawn several at once.
- **Add a render pass:** After each `nl_bullet_step`, print an ASCII-art view of the play field based on ball and player positions.
- **Tune physics:** Experiment with `restitution` (bounciness) values to make the ball bounce more or less before settling.
- **Add difficulty scaling:** Reduce `PLAYER_SPEED` or increase ball spawn height as the score grows.

---

**Previous:** [18.3 bullet](bullet.html)
**Next:** [Chapter 19: Terminal UI](../19_terminal_ui/index.html)
