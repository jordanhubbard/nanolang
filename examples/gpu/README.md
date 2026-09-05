# My GPU Backends

I compile `gpu fn` kernels to NVIDIA PTX and OpenCL C. My runtime tries CUDA
first, then OpenCL. These paths have different requirements; a machine having a
library named OpenCL is not enough to establish that a generated kernel runs.

## Runtime requirements

The runtime loads drivers dynamically, so building the host executable requires
no CUDA or OpenCL SDK. Running a kernel still requires all of the following:

| Path | Requirements |
| --- | --- |
| CUDA | NVIDIA driver, a CUDA-capable device, and driver support for the PTX version and target emitted by the compiler |
| OpenCL | A working ICD/runtime, an OpenCL device, OpenCL C 1.2 compilation, and every extension used by the generated kernel |
| OpenCL CPU | A CPU implementation such as POCL; this is a runtime dependency, not a built-in fallback |

My generated OpenCL currently requests `cl_khr_int64_base_atomics`,
`cl_khr_byte_addressable_store`, and `cl_khr_fp64`, and builds with
`-cl-std=CL1.2 -cl-unsafe-math-optimizations`. A device may enumerate correctly
and still reject the program at `clBuildProgram`. Read `gpu_last_error()`; that
build log is the relevant result.

The runtime receives a `.ptx` path. On OpenCL it replaces the suffix with `.cl`,
so both generated files must be deployed together with matching kernel names and
arguments.

On macOS I load `/System/Library/Frameworks/OpenCL.framework/OpenCL`. Apple has
deprecated OpenCL. Framework presence is not a portability guarantee, and I do
not describe the OpenCL Julia showcase as a supported macOS guarantee.

## Quick start

```nano
import "modules/gpu/gpu.nano"

gpu fn vector_add(a: int, b: int, c: int, n: int) -> void {
    let i: int = (global_id_x)
    if (< i n) {
        # ... per-element work ...
        return
    }
}

fn main() -> int {
    if (not (gpu_init)) { print "No GPU" return 1 }
    let d_a: int = (gpu_alloc (* n 8))
    let d_b: int = (gpu_alloc (* n 8))
    let d_c: int = (gpu_alloc (* n 8))
    (gpu_launch "vector_add.ptx" "vector_add" blocks threads d_a d_b d_c n)
    (gpu_sync)
    (gpu_free d_a)
    (gpu_free d_b)
    (gpu_free d_c)
    return 0
}
```

See `vector_add.nano` for the full working example.

## Compiling

```bash
# Compile kernel to PTX
nanoc --target ptx vector_add.nano -o vector_add.ptx

# Compile host program (links cuda_runtime.c from the gpu module)
nanoc vector_add.nano -o vector_add

# Run (requires CUDA-capable GPU; gracefully fails without one)
./vector_add
```

## `gpu fn` syntax

Annotate a function with `gpu fn` to mark it as a GPU kernel.
The compiler emits it as a PTX `.entry` when `--target ptx` is used.

```nano
gpu fn kernel_name(param: int, ...) -> void {
    # body — only GPU intrinsics and arithmetic allowed
}
```

## Thread indexing builtins

All builtins return `int` (64-bit signed, zero-extended from u32 PTX registers).

| nanolang call      | PTX register   | Meaning                          |
|--------------------|----------------|----------------------------------|
| `(thread_id_x)`    | `%tid.x`       | Thread index within block, X     |
| `(thread_id_y)`    | `%tid.y`       | Thread index within block, Y     |
| `(thread_id_z)`    | `%tid.z`       | Thread index within block, Z     |
| `(block_id_x)`     | `%ctaid.x`     | Block index within grid, X       |
| `(block_id_y)`     | `%ctaid.y`     | Block index within grid, Y       |
| `(block_id_z)`     | `%ctaid.z`     | Block index within grid, Z       |
| `(block_dim_x)`    | `%ntid.x`      | Threads per block, X             |
| `(block_dim_y)`    | `%ntid.y`      | Threads per block, Y             |
| `(block_dim_z)`    | `%ntid.z`      | Threads per block, Z             |
| `(grid_dim_x)`     | `%nctaid.x`    | Blocks per grid, X               |
| `(grid_dim_y)`     | `%nctaid.y`    | Blocks per grid, Y               |
| `(grid_dim_z)`     | `%nctaid.z`    | Blocks per grid, Z               |
| `(global_id_x)`    | computed       | `block_id_x * block_dim_x + thread_id_x` |
| `(global_id_y)`    | computed       | `block_id_y * block_dim_y + thread_id_y` |
| `(gpu_barrier)`    | `bar.sync 0`   | Synchronize all threads in block |

## `gpu` module API

Import with `import gpu` to use these from host code:

### Device management

```nano
gpu_init()              -> bool    # Initialize CUDA; true if GPU available
gpu_device_count()      -> int     # Number of CUDA devices
gpu_device_info(n: int) -> string  # "Name | X.Y GB | sm_XY"
gpu_sync()              -> void    # Wait for all GPU work to complete
gpu_available()         -> bool    # Non-initializing availability check
```

### Device memory

```nano
gpu_alloc(bytes: int)                              -> int   # Allocate; 0 on fail
gpu_free(ptr: int)                                 -> void
gpu_memcpy_to_device(dst: int, src: array<int>, bytes: int)   -> bool
gpu_memcpy_from_device(dst: array<int>, src: int, bytes: int) -> bool
```

Device pointers are opaque `int` values (64-bit encoded `CUdeviceptr`).

### Kernel launch

```nano
# 1D launch
gpu_launch(ptx_file: string, kernel_name: string,
           grid_x: int, block_x: int,
           arg0: int, arg1: int, arg2: int, arg3: int) -> bool

# 2D launch (for matrix/image kernels)
gpu_launch2d(ptx_file: string, kernel_name: string,
             grid_x: int, grid_y: int,
             block_x: int, block_y: int,
             arg0: int, arg1: int, arg2: int, arg3: int) -> bool
```

### Error handling

```nano
gpu_last_error() -> string   # Last CUDA error string; "" if none
```

## CUDA availability and fallback

`cuda_runtime.c` uses `dlopen` to load `libcuda.so.1` (Linux) /
`libcuda.dylib` (macOS) / `nvcuda.dll` (Windows) at runtime.
No CUDA headers or libraries are required at compile time.

When no CUDA driver is present:
- `gpu_init()` and `gpu_available()` return `false`
- `gpu_alloc()` returns `0`
- `gpu_launch()` returns `false`
- `gpu_last_error()` returns `""` (no CUDA loaded)

## Supported types in `gpu fn`

| nanolang | PTX register class |
|----------|--------------------|
| `int`    | `.s64 %rd`         |
| `float`  | `.f64 %fd`         |
| `bool`   | `.s64 %rd` (0/1)   |

## Supported operations in `gpu fn`

- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparisons**: `==`, `!=`, `<`, `<=`, `>`, `>=`
- **Control flow**: `if`/`else`, `return`
- **Variables**: `let`
- **Thread builtins**: all listed in the table above
- **Block sync**: `(gpu_barrier)` → `bar.sync 0`

## Target architecture

PTX is emitted for `.target sm_90` (H100 Hopper / GB10 Blackwell).
The CUDA driver JIT-compiles PTX to native ISA at load time, so the
same `.ptx` file runs on any sm_80+ (Ampere) or newer GPU.
