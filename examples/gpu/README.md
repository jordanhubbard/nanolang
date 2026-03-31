# nanolang GPU / PTX Backend

nanolang can emit NVIDIA PTX assembly for GPU kernels annotated with `gpu fn`.

## Syntax

```nano
/* gpu fn — annotates a function as a GPU kernel */
gpu fn vec_add(a: int, b: int) -> int {
    let tid = (thread_id_x)
    let bid = (block_id_x)
    let bsz = (block_dim_x)
    let gid = (+ (* bid bsz) tid)
    return (+ (+ a b) gid)
}
```

## Compile to PTX

```bash
nanoc --target ptx my_kernels.nano -o my_kernels.ptx
```

Output: `my_kernels.ptx` — valid NVIDIA PTX text (`.version 8.0`, `.target sm_90`).

## Thread indexing builtins

| nanolang           | PTX special register |
|--------------------|----------------------|
| `(thread_id_x)`    | `%tid.x`             |
| `(thread_id_y)`    | `%tid.y`             |
| `(thread_id_z)`    | `%tid.z`             |
| `(block_id_x)`     | `%ctaid.x`           |
| `(block_id_y)`     | `%ctaid.y`           |
| `(block_dim_x)`    | `%ntid.x`            |
| `(block_dim_y)`    | `%ntid.y`            |

## Load with CUDA driver API

```c
// load_ptx.c — minimal CUDA driver API loader
#include <cuda.h>
#include <stdio.h>

int main(void) {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule   mod;
    CUfunction fn;
    cuModuleLoad(&mod, "my_kernels.ptx");
    cuModuleGetFunction(&fn, mod, "vec_add");

    /* Launch kernel: 1 block, 256 threads */
    int a = 10, b = 20;
    void *args[] = { &a, &b };
    cuLaunchKernel(fn, 1, 1, 1,   /* gridDim */
                       256, 1, 1,  /* blockDim */
                       0, NULL, args, NULL);
    cuCtxSynchronize();
    printf("kernel launched OK\n");

    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
```

```bash
gcc load_ptx.c -o load_ptx -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcuda
./load_ptx
```

## Supported types

| nanolang   | PTX register |
|------------|--------------|
| `int`      | `.s64 %rd`   |
| `float`    | `.f64 %fd`   |
| `bool`     | `.s64 %rd`   |

## Supported operations

- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Control flow: `if`/`else`, `return`
- Variables: `let`
- Thread indexing: `(thread_id_x)`, `(block_id_x)`, `(block_dim_x)` etc.

## Target architecture

The PTX backend targets `sm_90` (H100/GB10 Blackwell/Hopper).
The emitted PTX is forward-compatible with newer architectures and
backward-compatible with sm_80+ (Ampere) via driver-level JIT.

## Integration with agentOS GPU scheduler

The emitted `.ptx` files are suitable for loading into agentOS GPU scheduler
slots via the CUDA driver API.  Combine with the `--profile-runtime` flag
(on the native target) to profile the dispatch overhead.
