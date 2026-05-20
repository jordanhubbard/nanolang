# MuJoCo Module

I bind MuJoCo as an owned simulation handle. A handle owns one `mjModel` and one `mjData`. You load MJCF, step time, set controls, and read body or geom positions for rendering.

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

I need the C header at build time:

```bash
export CPATH=/path/to/mujoco/include:$CPATH
rm -rf modules/mujoco/.build
```

I need the shared library at run time:

```bash
export NANOLANG_MUJOCO_LIB=/path/to/mujoco/lib/libmujoco.dylib
```

On Linux the library name is usually `libmujoco.so`.

The rebuild matters. My module cache does not know when an external header
appears after a fallback build.

## API Shape

- `mujoco_load(path) -> MjSim`
- `mujoco_valid(sim) -> bool`
- `nl_mj_step(sim)` and `nl_mj_step_n(sim, count)`
- `nl_mj_set_ctrl(sim, actuator_index, value)`
- `nl_mj_body_{x,y,z}(sim, body_id)`
- `nl_mj_geom_{x,y,z}(sim, geom_id)`
- `nl_mj_qpos` / `nl_mj_set_qpos`
- `nl_mj_qvel` / `nl_mj_set_qvel`

I expose low-level `nl_mj_*` calls because MuJoCo is explicit. The small `mujoco_*` helpers are there where they remove repetition without hiding the simulator.
