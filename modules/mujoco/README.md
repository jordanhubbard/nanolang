# MuJoCo Module

I bind MuJoCo as an owned simulation handle. A handle owns one `mjModel` and one `mjData`. You load MJCF, step time, set controls, read state, and render from body, site, or geom metadata.

```nano
unsafe module "modules/mujoco/mujoco.nano"

fn main() -> int {
    let sim: MjSim = (mujoco_load "examples/mujoco/assets/cartpole.xml")
    if (not (mujoco_valid sim)) {
        (println (mujoco_last_error))
        return 1
    } else {}

    let cart: int = (nl_mj_body_id sim "cart")
    (nl_mj_step_n sim 8)
    (println (nl_mj_body_x sim cart))

    (mujoco_free sim)
    return 0
}
```

## Install

MuJoCo releases are published at `https://github.com/google-deepmind/mujoco/releases`.

On Linux I install MuJoCo from release metadata into `.nanolang/native/mujoco`
when you import this module and the headers are missing.

If you install it yourself, I also look in `/opt/mujoco` and `/usr/local`:

```bash
sudo mkdir -p /opt/mujoco
# unpack the release so /opt/mujoco/include/mujoco/mujoco.h exists
rm -rf modules/mujoco/.build
```

If you install somewhere else:

```bash
export CPATH=/path/to/mujoco/include:$CPATH
rm -rf modules/mujoco/.build
```

I need the shared library at run time:

```bash
export NANOLANG_MUJOCO_LIB=/path/to/mujoco/lib/libmujoco.so
```

My Linux auto-install path does not need that variable when you run from the
repository root.

On macOS the library name is usually `libmujoco.dylib`.

The rebuild matters. My module cache does not know when an external header
appears after a fallback build.

## API Shape

- `mujoco_available() -> bool`
- `mujoco_last_error() -> string`
- `mujoco_version_string() -> string`
- `mujoco_load(path) -> MjSim`
- `mujoco_free(sim) -> void`
- `mujoco_valid(sim) -> bool`
- `nl_mj_reset(sim)` and `nl_mj_forward(sim)`
- `nl_mj_step(sim)`, `nl_mj_step_n(sim, count)`, and `mujoco_step_seconds(sim, seconds)`
- `nl_mj_time(sim)`, `nl_mj_timestep(sim)`, and `nl_mj_set_timestep(sim, value)`
- `nl_mj_nq`, `nl_mj_nv`, `nl_mj_nu`, `nl_mj_nbody`, `nl_mj_ngeom`, `nl_mj_nsite`, and `nl_mj_njoint`
- `nl_mj_body_id`, `nl_mj_geom_id`, `nl_mj_site_id`, `nl_mj_joint_id`, and `nl_mj_actuator_id`
- `nl_mj_body_name` and `nl_mj_geom_name`
- `nl_mj_joint_qpos_addr` and `nl_mj_joint_qvel_addr`
- `nl_mj_body_qpos_addr` and `nl_mj_body_qvel_addr` for the first joint owned by a body
- `nl_mj_body_{x,y,z}(sim, body_id)`
- `nl_mj_body_quat(sim, body_id, axis)`
- `nl_mj_geom_{x,y,z}(sim, geom_id)`, `nl_mj_geom_xmat`, `nl_mj_geom_size`, `nl_mj_geom_rgba`, `nl_mj_geom_type`, and `nl_mj_geom_body`
- `nl_mj_site_{x,y,z}(sim, site_id)`
- `nl_mj_qpos` / `nl_mj_set_qpos`
- `nl_mj_qvel` / `nl_mj_set_qvel`
- `nl_mj_ctrl` / `nl_mj_set_ctrl`
- `mujoco_body_pos`, `mujoco_geom_pos`, `mujoco_site_pos`, and `mujoco_body_quat`
- `mujoco_set_ctrl_clamped`, `mujoco_geom_is_sphere`, and `mujoco_geom_is_box`

I expose low-level `nl_mj_*` calls because MuJoCo is explicit. The small `mujoco_*` helpers are there where they remove repetition without hiding the simulator.

## Examples

- `examples/mujoco/mujoco_state_audit.nano` loads `cartpole.xml`, writes `qpos`, `qvel`, and `ctrl`, advances deterministic time, and prints the state.
- `examples/mujoco/mujoco_cartpole_balance.nano` reads state each step and applies a bounded feedback controller.
- `examples/mujoco/mujoco_opengl_cartpole.nano` renders cartpole body and site transforms with OpenGL.
- `examples/mujoco/mujoco_opengl_drop_lab.nano` renders a small contact scene from body transforms and keeps reusing the balls while the window is open.
- `examples/mujoco/mujoco_opengl_geom_browser.nano` renders `drop_lab.xml` from geom type, size, color, transform, and body ownership metadata while the same ball stream keeps running.
- `examples/mujoco/mujoco_opengl_robot_sorter.nano` loads a three-joint arm, two bins, and five freejoint balls, then scripts a pick, carry, and drop cycle.
