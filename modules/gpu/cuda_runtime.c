/*
 * cuda_runtime.c — CUDA GPU computing runtime for nanolang
 *
 * Implements all extern functions declared in gpu.nano.
 * Loads the CUDA driver API dynamically at runtime via dlopen so that
 * nanolang programs compile and run on machines without CUDA installed.
 * When no GPU is present, all functions return safe "unavailable" values.
 *
 * Link: no special flags needed (dlopen is in libdl on Linux, implicit on macOS).
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#  define CUDA_LIB "nvcuda.dll"
typedef HMODULE DL;
static DL   dl_open(const char *n) { return LoadLibraryA(n); }
static void *dl_sym(DL h, const char *s) { return (void*)GetProcAddress(h, s); }
#else
#  include <dlfcn.h>
#  ifdef __APPLE__
#    define CUDA_LIB "libcuda.dylib"
#  else
#    define CUDA_LIB "libcuda.so.1"
#  endif
typedef void *DL;
static DL   dl_open(const char *n) { return dlopen(n, RTLD_LAZY | RTLD_LOCAL); }
static void *dl_sym(DL h, const char *s) { return dlsym(h, s); }
#endif

/* ── CUDA driver API types (avoid requiring cuda.h) ─────────────────────── */
typedef int          CUresult;
typedef unsigned int CUdevice;
typedef void        *CUcontext;
typedef void        *CUmodule;
typedef void        *CUfunction;
typedef uint64_t     CUdeviceptr;

#define CUDA_SUCCESS 0

/* Function pointer types for driver API calls we use */
typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGetCount)(int *);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfn_cuDeviceGetName)(char *, int, CUdevice);
typedef CUresult (*pfn_cuDeviceTotalMem)(size_t *, CUdevice);
typedef CUresult (*pfn_cuDeviceComputeCapability)(int *, int *, CUdevice);
typedef CUresult (*pfn_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction,
    unsigned int, unsigned int, unsigned int,   /* grid  x/y/z */
    unsigned int, unsigned int, unsigned int,   /* block x/y/z */
    unsigned int, void *,                       /* shared mem, stream */
    void **, void **);                          /* kernelParams, extra */
typedef CUresult (*pfn_cuGetErrorString)(CUresult, const char **);

/* ── Driver singleton ────────────────────────────────────────────────────── */
static struct {
    DL   lib;
    bool initialized;
    bool available;
    CUcontext ctx;
    CUresult  last_error;
    char      last_error_str[256];

    pfn_cuInit                   cuInit;
    pfn_cuDeviceGetCount         cuDeviceGetCount;
    pfn_cuDeviceGet              cuDeviceGet;
    pfn_cuDeviceGetName          cuDeviceGetName;
    pfn_cuDeviceTotalMem         cuDeviceTotalMem;
    pfn_cuDeviceComputeCapability cuDeviceComputeCapability;
    pfn_cuCtxCreate              cuCtxCreate;
    pfn_cuCtxSynchronize         cuCtxSynchronize;
    pfn_cuMemAlloc               cuMemAlloc;
    pfn_cuMemFree                cuMemFree;
    pfn_cuMemcpyHtoD             cuMemcpyHtoD;
    pfn_cuMemcpyDtoH             cuMemcpyDtoH;
    pfn_cuModuleLoadData         cuModuleLoadData;
    pfn_cuModuleGetFunction      cuModuleGetFunction;
    pfn_cuLaunchKernel           cuLaunchKernel;
    pfn_cuGetErrorString         cuGetErrorString;
} g_cuda;

#define LOAD_SYM(name) \
    g_cuda.name = (pfn_##name) dl_sym(g_cuda.lib, #name); \
    if (!g_cuda.name) goto fail;

static void cuda_set_error(CUresult r) {
    g_cuda.last_error = r;
    if (r == CUDA_SUCCESS) {
        g_cuda.last_error_str[0] = '\0';
        return;
    }
    if (g_cuda.cuGetErrorString) {
        const char *s = NULL;
        g_cuda.cuGetErrorString(r, &s);
        if (s) { strncpy(g_cuda.last_error_str, s, sizeof(g_cuda.last_error_str)-1); return; }
    }
    snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
             "CUDA error %d", r);
}

static bool cuda_load(void) {
    if (g_cuda.initialized) return g_cuda.available;
    g_cuda.initialized = true;

    g_cuda.lib = dl_open(CUDA_LIB);
    if (!g_cuda.lib) return false;

    LOAD_SYM(cuInit);
    LOAD_SYM(cuDeviceGetCount);
    LOAD_SYM(cuDeviceGet);
    LOAD_SYM(cuDeviceGetName);
    LOAD_SYM(cuDeviceTotalMem);
    LOAD_SYM(cuDeviceComputeCapability);
    LOAD_SYM(cuCtxCreate);
    LOAD_SYM(cuCtxSynchronize);
    LOAD_SYM(cuMemAlloc);
    LOAD_SYM(cuMemFree);
    LOAD_SYM(cuMemcpyHtoD);
    LOAD_SYM(cuMemcpyDtoH);
    LOAD_SYM(cuModuleLoadData);
    LOAD_SYM(cuModuleGetFunction);
    LOAD_SYM(cuLaunchKernel);
    /* cuGetErrorString is optional — may not exist in older drivers */
    g_cuda.cuGetErrorString = (pfn_cuGetErrorString) dl_sym(g_cuda.lib, "cuGetErrorString");

    CUresult r = g_cuda.cuInit(0);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }

    int count = 0;
    r = g_cuda.cuDeviceGetCount(&count);
    if (r != CUDA_SUCCESS || count == 0) { cuda_set_error(r); return false; }

    CUdevice dev;
    r = g_cuda.cuDeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }

    r = g_cuda.cuCtxCreate(&g_cuda.ctx, 0, dev);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }

    g_cuda.available = true;
    return true;

fail:
    g_cuda.available = false;
    return false;
}

/* ── Module cache (avoid reloading the same PTX file repeatedly) ─────────── */
#define MAX_MODULES 32
static struct { char path[512]; CUmodule mod; } g_mod_cache[MAX_MODULES];
static int g_mod_cache_n = 0;

static CUmodule get_or_load_module(const char *ptx_file) {
    for (int i = 0; i < g_mod_cache_n; i++)
        if (strcmp(g_mod_cache[i].path, ptx_file) == 0)
            return g_mod_cache[i].mod;

    /* Read PTX source */
    FILE *f = fopen(ptx_file, "rb");
    if (!f) {
        snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
                 "cannot open PTX file: %s", ptx_file);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *ptx = malloc((size_t)sz + 1);
    if (!ptx) { fclose(f); return NULL; }
    fread(ptx, 1, (size_t)sz, f);
    ptx[sz] = '\0';
    fclose(f);

    CUmodule mod = NULL;
    CUresult r = g_cuda.cuModuleLoadData(&mod, ptx);
    free(ptx);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return NULL; }

    if (g_mod_cache_n < MAX_MODULES) {
        strncpy(g_mod_cache[g_mod_cache_n].path, ptx_file,
                sizeof(g_mod_cache[0].path)-1);
        g_mod_cache[g_mod_cache_n].mod = mod;
        g_mod_cache_n++;
    }
    return mod;
}

/* ── nanolang runtime ABI helpers ────────────────────────────────────────── */
/*
 * nanolang's C transpiler represents `string` as `const char *`
 * and `array<int>` / `array<float>` as a NanoArray struct:
 *   { int64_t *data; int64_t len; int64_t cap; }
 * (see runtime/nano_string.h and runtime/dyn_array.h)
 * We use void* here to avoid a compile-time dependency on those headers.
 */

/* ── Public API ──────────────────────────────────────────────────────────── */

bool nl_gpu_available(void) {
    return cuda_load();
}

bool nl_gpu_init(void) {
    return cuda_load();
}

int64_t nl_gpu_device_count(void) {
    if (!cuda_load()) return 0;
    int count = 0;
    CUresult r = g_cuda.cuDeviceGetCount(&count);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return 0; }
    return (int64_t)count;
}

const char *nl_gpu_device_info(int64_t device) {
    static char buf[512];
    buf[0] = '\0';
    if (!cuda_load()) return buf;

    CUdevice dev;
    CUresult r = g_cuda.cuDeviceGet(&dev, (int)device);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return buf; }

    char name[256] = {0};
    g_cuda.cuDeviceGetName(name, sizeof(name)-1, dev);

    size_t total_mem = 0;
    g_cuda.cuDeviceTotalMem(&total_mem, dev);

    int major = 0, minor = 0;
    g_cuda.cuDeviceComputeCapability(&major, &minor, dev);

    snprintf(buf, sizeof(buf), "%s | %.1f GB | sm_%d%d",
             name,
             (double)total_mem / (1024.0*1024.0*1024.0),
             major, minor);
    return buf;
}

void nl_gpu_sync(void) {
    if (!g_cuda.available) return;
    CUresult r = g_cuda.cuCtxSynchronize();
    cuda_set_error(r);
}

int64_t nl_gpu_alloc(int64_t bytes) {
    if (!cuda_load() || bytes <= 0) return 0;
    CUdeviceptr ptr = 0;
    CUresult r = g_cuda.cuMemAlloc(&ptr, (size_t)bytes);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return 0; }
    return (int64_t)ptr;
}

void nl_gpu_free(int64_t ptr) {
    if (!g_cuda.available || ptr == 0) return;
    CUresult r = g_cuda.cuMemFree((CUdeviceptr)ptr);
    cuda_set_error(r);
}

/* nanolang array<int> / array<float> is passed as DynArray* from generated C.
 * DynArray layout (from runtime/dyn_array.h — must stay in sync):
 *   int64_t length;        offset 0
 *   int64_t capacity;      offset 8
 *   int32_t elem_type;     offset 16   (ElementType enum)
 *   uint8_t elem_size;     offset 20
 *   [3 bytes padding]
 *   void *data;            offset 24
 * We use a local mirror struct to access .data without a header dependency. */
typedef struct {
    int64_t  length;
    int64_t  capacity;
    int32_t  elem_type;
    uint8_t  elem_size;
    uint8_t  _pad[3];
    void    *data;
} NLArray;

bool nl_gpu_memcpy_to_device(int64_t dst, NLArray *src, int64_t bytes) {
    if (!cuda_load() || !src || !src->data || bytes <= 0) return false;
    CUresult r = g_cuda.cuMemcpyHtoD((CUdeviceptr)dst, src->data, (size_t)bytes);
    cuda_set_error(r);
    return r == CUDA_SUCCESS;
}

bool nl_gpu_memcpy_from_device(NLArray *dst, int64_t src, int64_t bytes) {
    if (!cuda_load() || !dst || !dst->data || bytes <= 0) return false;
    CUresult r = g_cuda.cuMemcpyDtoH(dst->data, (CUdeviceptr)src, (size_t)bytes);
    cuda_set_error(r);
    return r == CUDA_SUCCESS;
}

bool nl_gpu_launch(const char *ptx_file, const char *kernel_name,
                int64_t grid_x, int64_t block_x,
                int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3) {
    if (!cuda_load()) return false;

    CUmodule mod = get_or_load_module(ptx_file);
    if (!mod) return false;

    CUfunction fn = NULL;
    CUresult r = g_cuda.cuModuleGetFunction(&fn, mod, kernel_name);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }

    void *args[] = { &arg0, &arg1, &arg2, &arg3 };
    r = g_cuda.cuLaunchKernel(fn,
        (unsigned int)grid_x,  1, 1,
        (unsigned int)block_x, 1, 1,
        0, NULL, args, NULL);
    cuda_set_error(r);
    return r == CUDA_SUCCESS;
}

bool nl_gpu_launch5(const char *ptx_file, const char *kernel_name,
                 int64_t grid_x, int64_t block_x,
                 int64_t arg0, int64_t arg1, int64_t arg2,
                 int64_t arg3, int64_t arg4) {
    if (!cuda_load()) return false;

    CUmodule mod = get_or_load_module(ptx_file);
    if (!mod) return false;

    CUfunction fn = NULL;
    CUresult r = g_cuda.cuModuleGetFunction(&fn, mod, kernel_name);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }

    void *args[] = { &arg0, &arg1, &arg2, &arg3, &arg4 };
    r = g_cuda.cuLaunchKernel(fn,
        (unsigned int)grid_x,  1, 1,
        (unsigned int)block_x, 1, 1,
        0, NULL, args, NULL);
    cuda_set_error(r);
    return r == CUDA_SUCCESS;
}

bool nl_gpu_launch2d(const char *ptx_file, const char *kernel_name,
                  int64_t grid_x, int64_t grid_y,
                  int64_t block_x, int64_t block_y,
                  int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3) {
    if (!cuda_load()) return false;

    CUmodule mod = get_or_load_module(ptx_file);
    if (!mod) return false;

    CUfunction fn = NULL;
    CUresult r = g_cuda.cuModuleGetFunction(&fn, mod, kernel_name);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }

    void *args[] = { &arg0, &arg1, &arg2, &arg3 };
    r = g_cuda.cuLaunchKernel(fn,
        (unsigned int)grid_x,  (unsigned int)grid_y,  1,
        (unsigned int)block_x, (unsigned int)block_y, 1,
        0, NULL, args, NULL);
    cuda_set_error(r);
    return r == CUDA_SUCCESS;
}

const char *nl_gpu_last_error(void) {
    return g_cuda.last_error_str;
}

int64_t nl_gpu_atomic_add(int64_t ptr, int64_t delta) {
    /* Atomic adds require custom PTX or CUDA device-side code.
     * On the host, this is a no-op stub — real use is inside gpu fn bodies
     * via PTX atomicAdd instruction (not yet supported by ptx_backend). */
    (void)ptr; (void)delta;
    return 0;
}

/* ── GPU intrinsic stubs for native C compilation ────────────────────────────
 * When nanoc compiles a file containing `gpu fn` to native C (not PTX),
 * the transpiler emits calls to nl_gpu_* for all GPU builtins used in the
 * kernel body. These stubs satisfy the linker; they are never called at
 * runtime since gpu fn bodies only execute on-device via PTX. */
int64_t nl_gpu_thread_id_x(void) { return 0; }
int64_t nl_gpu_thread_id_y(void) { return 0; }
int64_t nl_gpu_thread_id_z(void) { return 0; }
int64_t nl_gpu_block_id_x(void)  { return 0; }
int64_t nl_gpu_block_id_y(void)  { return 0; }
int64_t nl_gpu_block_id_z(void)  { return 0; }
int64_t nl_gpu_block_dim_x(void) { return 256; }
int64_t nl_gpu_block_dim_y(void) { return 256; }
int64_t nl_gpu_block_dim_z(void) { return 1; }
int64_t nl_gpu_grid_dim_x(void)  { return 1; }
int64_t nl_gpu_grid_dim_y(void)  { return 1; }
int64_t nl_gpu_grid_dim_z(void)  { return 1; }
int64_t nl_gpu_global_id_x(void) { return 0; }
int64_t nl_gpu_global_id_y(void) { return 0; }
void    nl_gpu_barrier(void)      { }
/* GPU memory access stubs (no-op on CPU; real work done by PTX ld/st.global) */
int64_t nl_gpu_load(int64_t ptr)             { (void)ptr; return 0; }
void    nl_gpu_store(int64_t ptr, int64_t v) { (void)ptr; (void)v; }
double  nl_gpu_load_float(int64_t ptr)       { (void)ptr; return 0.0; }
void    nl_gpu_store_float(int64_t ptr, double v) { (void)ptr; (void)v; }
