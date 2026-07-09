# MuJoCo MVP

I expose MuJoCo through one owned `MjSim` handle.

```nano
unsafe module "modules/mujoco/mujoco.nano"

fn main() -> int {
    let sim: MjSim = (mujoco_load "examples/mujoco/assets/cartpole.xml")
    if (not (mujoco_valid sim)) {
        (println (mujoco_last_error))
        return 1
    } else {}

    (nl_mj_step_n sim 10)
    (println (nl_mj_time sim))
    (mujoco_free sim)
    return 0
}
```

I need MuJoCo headers at module build time and a MuJoCo shared library at run time. If the loader cannot find the shared library, set `NANOLANG_MUJOCO_LIB`.

If I was first built without headers, remove `modules/mujoco/.build` after
putting MuJoCo on the include path.
