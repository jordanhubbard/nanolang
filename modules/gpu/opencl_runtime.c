/*
 * opencl_runtime.c — unified GPU runtime for nanolang
 *
 * Implements all extern functions declared in gpu.nano.
 * Tries CUDA first (via dlopen of libcuda); if unavailable falls back to
 * OpenCL (via dlopen of libOpenCL).  OpenCL's CPU platform (POCL or Intel)
 * provides a genuine CPU-execution fallback with no special hardware needed.
 *
 * Key design decisions:
 *   - CUDA path: identical to cuda_runtime.c (PTX file loaded by driver).
 *   - OpenCL path: loads .cl file (same base name as .ptx, extension swapped).
 *   - Kernel args: int64 values that match g_ocl_alloc sentinels are passed
 *     as cl_mem (buffer) args; all others as long scalars.  Sentinel pattern:
 *       0x0C1A000000000000LL | (uint8_t index)  — cannot collide with user ints.
 *   - Module/kernel cache: keyed by (file_path, kernel_name).
 *
 * No compile-time dependency on CUDA or OpenCL headers.
 * Link flags: none (dlopen is in libdl on Linux, implicit on macOS/Windows).
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Dynamic library helpers ─────────────────────────────────────────────── */
#ifdef _WIN32
#  include <windows.h>
typedef HMODULE DL;
static DL   dl_open(const char *n)            { return LoadLibraryA(n); }
static void *dl_sym(DL h, const char *s)      { return (void*)GetProcAddress(h, s); }
#else
#  include <dlfcn.h>
typedef void *DL;
static DL   dl_open(const char *n)            { return dlopen(n, RTLD_LAZY | RTLD_LOCAL); }
static void *dl_sym(DL h, const char *s)      { return dlsym(h, s); }
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * CUDA DRIVER PATH
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef int          CUresult;
typedef unsigned int CUdevice;
typedef void        *CUcontext;
typedef void        *CUmodule;
typedef void        *CUfunction;
typedef uint64_t     CUdeviceptr;
#define CUDA_SUCCESS 0

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
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, void *, void **, void **);
typedef CUresult (*pfn_cuGetErrorString)(CUresult, const char **);

static struct {
    DL   lib;
    bool initialized, available;
    CUcontext ctx;
    CUresult  last_error;
    char      last_error_str[256];
    pfn_cuInit                    cuInit;
    pfn_cuDeviceGetCount          cuDeviceGetCount;
    pfn_cuDeviceGet               cuDeviceGet;
    pfn_cuDeviceGetName           cuDeviceGetName;
    pfn_cuDeviceTotalMem          cuDeviceTotalMem;
    pfn_cuDeviceComputeCapability cuDeviceComputeCapability;
    pfn_cuCtxCreate               cuCtxCreate;
    pfn_cuCtxSynchronize          cuCtxSynchronize;
    pfn_cuMemAlloc                cuMemAlloc;
    pfn_cuMemFree                 cuMemFree;
    pfn_cuMemcpyHtoD              cuMemcpyHtoD;
    pfn_cuMemcpyDtoH              cuMemcpyDtoH;
    pfn_cuModuleLoadData          cuModuleLoadData;
    pfn_cuModuleGetFunction       cuModuleGetFunction;
    pfn_cuLaunchKernel            cuLaunchKernel;
    pfn_cuGetErrorString          cuGetErrorString;
} g_cuda;

#define CUDA_LIB_NAME \
    (defined(_WIN32) ? "nvcuda.dll" : (defined(__APPLE__) ? "libcuda.dylib" : "libcuda.so.1"))

#ifdef _WIN32
#  define CUDA_LIB "nvcuda.dll"
#elif defined(__APPLE__)
#  define CUDA_LIB "libcuda.dylib"
#else
#  define CUDA_LIB "libcuda.so.1"
#endif

#define LOAD_CUDA(name) \
    g_cuda.name = (pfn_##name) dl_sym(g_cuda.lib, #name); \
    if (!g_cuda.name) goto cuda_fail;

static void cuda_set_error(CUresult r) {
    g_cuda.last_error = r;
    if (r == CUDA_SUCCESS) { g_cuda.last_error_str[0] = '\0'; return; }
    if (g_cuda.cuGetErrorString) {
        const char *s = NULL;
        g_cuda.cuGetErrorString(r, &s);
        if (s) { strncpy(g_cuda.last_error_str, s, sizeof(g_cuda.last_error_str)-1); return; }
    }
    snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str), "CUDA error %d", r);
}

static bool cuda_load(void) {
    if (g_cuda.initialized) return g_cuda.available;
    g_cuda.initialized = true;
    g_cuda.lib = dl_open(CUDA_LIB);
    if (!g_cuda.lib) return false;
    LOAD_CUDA(cuInit);
    LOAD_CUDA(cuDeviceGetCount);
    LOAD_CUDA(cuDeviceGet);
    LOAD_CUDA(cuDeviceGetName);
    LOAD_CUDA(cuDeviceTotalMem);
    LOAD_CUDA(cuDeviceComputeCapability);
    LOAD_CUDA(cuCtxCreate);
    LOAD_CUDA(cuCtxSynchronize);
    LOAD_CUDA(cuMemAlloc);
    LOAD_CUDA(cuMemFree);
    LOAD_CUDA(cuMemcpyHtoD);
    LOAD_CUDA(cuMemcpyDtoH);
    LOAD_CUDA(cuModuleLoadData);
    LOAD_CUDA(cuModuleGetFunction);
    LOAD_CUDA(cuLaunchKernel);
    g_cuda.cuGetErrorString = (pfn_cuGetErrorString) dl_sym(g_cuda.lib, "cuGetErrorString");
    if (g_cuda.cuInit(0) != CUDA_SUCCESS) goto cuda_fail;
    { int cnt = 0; if (g_cuda.cuDeviceGetCount(&cnt) != CUDA_SUCCESS || cnt == 0) goto cuda_fail; }
    { CUdevice dev; if (g_cuda.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) goto cuda_fail;
      if (g_cuda.cuCtxCreate(&g_cuda.ctx, 0, dev) != CUDA_SUCCESS) goto cuda_fail; }
    g_cuda.available = true;
    return true;
cuda_fail:
    g_cuda.available = false;
    return false;
}

/* CUDA PTX module cache */
#define MAX_CUDA_MODS 32
static struct { char path[512]; CUmodule mod; } g_cuda_mods[MAX_CUDA_MODS];
static int g_cuda_mods_n = 0;

static CUmodule cuda_get_module(const char *ptx_file) {
    for (int i = 0; i < g_cuda_mods_n; i++)
        if (strcmp(g_cuda_mods[i].path, ptx_file) == 0) return g_cuda_mods[i].mod;
    FILE *f = fopen(ptx_file, "rb");
    if (!f) { snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
                        "cannot open PTX file: %s", ptx_file); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *src = malloc((size_t)sz + 1);
    if (!src) { fclose(f); return NULL; }
    fread(src, 1, (size_t)sz, f); src[sz] = '\0'; fclose(f);
    CUmodule mod = NULL;
    CUresult r = g_cuda.cuModuleLoadData(&mod, src);
    free(src);
    if (r != CUDA_SUCCESS) { cuda_set_error(r); return NULL; }
    if (g_cuda_mods_n < MAX_CUDA_MODS) {
        strncpy(g_cuda_mods[g_cuda_mods_n].path, ptx_file,
                sizeof(g_cuda_mods[0].path)-1);
        g_cuda_mods[g_cuda_mods_n++].mod = mod;
    }
    return mod;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * OPENCL PATH
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Minimal OpenCL types — avoid requiring OpenCL headers */
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef int64_t  cl_long;
typedef uint64_t cl_ulong;
typedef void    *cl_platform_id;
typedef void    *cl_device_id;
typedef void    *cl_context;
typedef void    *cl_command_queue;
typedef void    *cl_program;
typedef void    *cl_kernel;
typedef void    *cl_mem;
typedef uint64_t cl_mem_flags;
typedef size_t   cl_size_t;

#define CL_SUCCESS               0
#define CL_DEVICE_TYPE_GPU       (1<<2)
#define CL_DEVICE_TYPE_CPU       (1<<1)
#define CL_DEVICE_TYPE_ALL       0xFFFFFFFF
#define CL_DEVICE_NAME           0x102B
#define CL_PLATFORM_NAME         0x0902
#define CL_MEM_READ_WRITE        (1<<0)
#define CL_QUEUE_PROFILING_ENABLE (1<<3)
#define CL_PROGRAM_BUILD_LOG     0x1183

/* Function pointer types */
typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, uint64_t, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (*pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*,
    void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (*pfn_clCreateCommandQueue)(cl_context, cl_device_id, uint64_t, cl_int*);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
    const void*, cl_uint, const void*, void*);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
    void*, cl_uint, const void*, void*);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**,
    const size_t*, cl_int*);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
    void (*)(cl_program, void*), void*);
typedef cl_int (*pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint,
    const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseContext)(cl_context);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);

#ifdef _WIN32
#  define OCL_LIB "OpenCL.dll"
#elif defined(__APPLE__)
#  define OCL_LIB "/System/Library/Frameworks/OpenCL.framework/OpenCL"
#else
#  define OCL_LIB "libOpenCL.so.1"
#endif

static struct {
    DL              lib;
    bool            initialized, available;
    cl_context      ctx;
    cl_device_id    device;
    cl_command_queue queue;
    char            last_error_str[512];

    pfn_clGetPlatformIDs           clGetPlatformIDs;
    pfn_clGetPlatformInfo          clGetPlatformInfo;
    pfn_clGetDeviceIDs             clGetDeviceIDs;
    pfn_clGetDeviceInfo            clGetDeviceInfo;
    pfn_clCreateContext            clCreateContext;
    pfn_clCreateCommandQueue       clCreateCommandQueue;
    pfn_clCreateBuffer             clCreateBuffer;
    pfn_clReleaseMemObject         clReleaseMemObject;
    pfn_clEnqueueWriteBuffer       clEnqueueWriteBuffer;
    pfn_clEnqueueReadBuffer        clEnqueueReadBuffer;
    pfn_clCreateProgramWithSource  clCreateProgramWithSource;
    pfn_clBuildProgram             clBuildProgram;
    pfn_clGetProgramBuildInfo      clGetProgramBuildInfo;
    pfn_clCreateKernel             clCreateKernel;
    pfn_clSetKernelArg             clSetKernelArg;
    pfn_clEnqueueNDRangeKernel     clEnqueueNDRangeKernel;
    pfn_clFinish                   clFinish;
    pfn_clReleaseKernel            clReleaseKernel;
    pfn_clReleaseProgram           clReleaseProgram;
} g_ocl;

#define LOAD_OCL(name) \
    g_ocl.name = (pfn_##name) dl_sym(g_ocl.lib, #name); \
    if (!g_ocl.name) goto ocl_fail;

static void ocl_set_error(cl_int err, const char *msg) {
    if (msg)
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "%s (cl_int=%d)", msg, err);
    else
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "OpenCL error %d", err);
}

static bool ocl_load(void) {
    if (g_ocl.initialized) return g_ocl.available;
    g_ocl.initialized = true;
    g_ocl.lib = dl_open(OCL_LIB);
    if (!g_ocl.lib) return false;
    LOAD_OCL(clGetPlatformIDs);
    LOAD_OCL(clGetPlatformInfo);
    LOAD_OCL(clGetDeviceIDs);
    LOAD_OCL(clGetDeviceInfo);
    LOAD_OCL(clCreateContext);
    LOAD_OCL(clCreateCommandQueue);
    LOAD_OCL(clCreateBuffer);
    LOAD_OCL(clReleaseMemObject);
    LOAD_OCL(clEnqueueWriteBuffer);
    LOAD_OCL(clEnqueueReadBuffer);
    LOAD_OCL(clCreateProgramWithSource);
    LOAD_OCL(clBuildProgram);
    LOAD_OCL(clGetProgramBuildInfo);
    LOAD_OCL(clCreateKernel);
    LOAD_OCL(clSetKernelArg);
    LOAD_OCL(clEnqueueNDRangeKernel);
    LOAD_OCL(clFinish);
    LOAD_OCL(clReleaseKernel);
    LOAD_OCL(clReleaseProgram);

    /* Pick best device: prefer GPU, fall back to CPU (POCL / Intel) */
    cl_platform_id platforms[16];
    cl_uint nplat = 0;
    if (g_ocl.clGetPlatformIDs(16, platforms, &nplat) != CL_SUCCESS || nplat == 0)
        goto ocl_fail;

    cl_device_id best_dev = NULL;
    int best_score = -1;
    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[8]; cl_uint ndev = 0;
        /* Try GPU first */
        if (g_ocl.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &ndev) == CL_SUCCESS
            && ndev > 0 && best_score < 2) { best_dev = devs[0]; best_score = 2; }
        /* Then CPU */
        if (g_ocl.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_CPU, 8, devs, &ndev) == CL_SUCCESS
            && ndev > 0 && best_score < 1) { best_dev = devs[0]; best_score = 1; }
    }
    if (!best_dev) goto ocl_fail;

    cl_int err;
    g_ocl.device = best_dev;
    g_ocl.ctx = g_ocl.clCreateContext(NULL, 1, &best_dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || !g_ocl.ctx) goto ocl_fail;
    g_ocl.queue = g_ocl.clCreateCommandQueue(g_ocl.ctx, best_dev, 0, &err);
    if (err != CL_SUCCESS || !g_ocl.queue) goto ocl_fail;

    g_ocl.available = true;
    return true;
ocl_fail:
    g_ocl.available = false;
    return false;
}

/* ── OpenCL alloc map ─────────────────────────────────────────────────────
 * nl_gpu_alloc with OpenCL returns a sentinel fake pointer so we can
 * distinguish buffer args from scalars in clSetKernelArg.
 * Sentinel: 0x0C1A000000000000LL | uint8_t_index
 * This pattern will never appear in real user scalar values (frame counters,
 * array sizes, etc. are small non-negative integers). */
#define OCL_SENTINEL ((int64_t)0x0C1A000000000000LL)
#define MAX_OCL_ALLOCS 256

static struct { int64_t fake; cl_mem buf; } g_ocl_allocs[MAX_OCL_ALLOCS];
static int g_ocl_nallocs = 0;

static cl_mem ocl_find_buf(int64_t val) {
    if ((val & ~(int64_t)0xFF) != OCL_SENTINEL) return NULL;
    int idx = (int)(val & 0xFF);
    if (idx >= g_ocl_nallocs) return NULL;
    return g_ocl_allocs[idx].buf;
}

/* ── OpenCL kernel cache ──────────────────────────────────────────────────
 * Key: (cl_path, kernel_name). Caches compiled program + kernel handle. */
#define MAX_OCL_KERNELS 64
static struct {
    char     cl_path[512];
    char     kname[128];
    cl_program prog;
    cl_kernel  kernel;
} g_ocl_kernels[MAX_OCL_KERNELS];
static int g_ocl_nkernels = 0;

static cl_kernel ocl_get_kernel(const char *cl_path, const char *kernel_name) {
    for (int i = 0; i < g_ocl_nkernels; i++)
        if (strcmp(g_ocl_kernels[i].cl_path, cl_path) == 0 &&
            strcmp(g_ocl_kernels[i].kname,   kernel_name) == 0)
            return g_ocl_kernels[i].kernel;

    /* Read .cl source */
    FILE *f = fopen(cl_path, "r");
    if (!f) {
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "cannot open .cl file: %s", cl_path);
        return NULL;
    }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *src = malloc((size_t)sz + 1);
    if (!src) { fclose(f); return NULL; }
    fread(src, 1, (size_t)sz, f); src[sz] = '\0'; fclose(f);

    cl_int err;
    const char *csrc = src;
    cl_program prog = g_ocl.clCreateProgramWithSource(g_ocl.ctx, 1, &csrc, NULL, &err);
    free(src);
    if (err != CL_SUCCESS || !prog) { ocl_set_error(err, "clCreateProgramWithSource"); return NULL; }

    cl_int build_rc = g_ocl.clBuildProgram(prog, 1, &g_ocl.device,
        "-cl-std=CL1.2 -cl-unsafe-math-optimizations", NULL, NULL);
    if (build_rc != CL_SUCCESS) {
        /* Extract build log for diagnostics */
        size_t log_sz = 0;
        g_ocl.clGetProgramBuildInfo(prog, g_ocl.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        if (log_sz > 1) {
            char *log = malloc(log_sz + 1);
            if (log) {
                g_ocl.clGetProgramBuildInfo(prog, g_ocl.device, CL_PROGRAM_BUILD_LOG,
                                            log_sz, log, NULL);
                log[log_sz] = '\0';
                snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                         "clBuildProgram failed:\n%.400s", log);
                free(log);
            }
        } else {
            ocl_set_error(build_rc, "clBuildProgram");
        }
        g_ocl.clReleaseProgram(prog);
        return NULL;
    }

    cl_kernel kern = g_ocl.clCreateKernel(prog, kernel_name, &err);
    if (err != CL_SUCCESS || !kern) {
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "clCreateKernel('%s') failed: %d", kernel_name, err);
        g_ocl.clReleaseProgram(prog);
        return NULL;
    }

    if (g_ocl_nkernels < MAX_OCL_KERNELS) {
        int i = g_ocl_nkernels++;
        strncpy(g_ocl_kernels[i].cl_path, cl_path, sizeof(g_ocl_kernels[i].cl_path)-1);
        strncpy(g_ocl_kernels[i].kname,   kernel_name, sizeof(g_ocl_kernels[i].kname)-1);
        g_ocl_kernels[i].prog   = prog;
        g_ocl_kernels[i].kernel = kern;
    }
    return kern;
}

/* Build .cl path from a .ptx path (replace last extension with .cl) */
static void ptx_to_cl_path(const char *ptx, char *out, size_t out_sz) {
    strncpy(out, ptx, out_sz-1);
    out[out_sz-1] = '\0';
    char *dot = strrchr(out, '.');
    if (dot) { *dot = '\0'; }
    strncat(out, ".cl", out_sz - strlen(out) - 1);
}

/* Set up to 5 kernel args; buf args use cl_mem, scalars use long */
static bool ocl_set_args(cl_kernel kern, int argc, int64_t *argv) {
    for (int i = 0; i < argc; i++) {
        cl_mem buf = ocl_find_buf(argv[i]);
        cl_int err;
        if (buf) {
            err = g_ocl.clSetKernelArg(kern, (cl_uint)i, sizeof(cl_mem), &buf);
        } else {
            cl_long scalar = (cl_long)argv[i];
            err = g_ocl.clSetKernelArg(kern, (cl_uint)i, sizeof(cl_long), &scalar);
        }
        if (err != CL_SUCCESS) {
            snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                     "clSetKernelArg[%d] failed: %d", i, err);
            return false;
        }
    }
    return true;
}

static bool ocl_launch_nd(const char *ptx_or_cl, const char *kernel_name,
                           size_t gx, size_t gy, size_t lx, size_t ly,
                           int argc, int64_t *argv) {
    char cl_path[1024];
    /* Accept explicit .cl path or derive from .ptx */
    if (strlen(ptx_or_cl) > 3 &&
        strcmp(ptx_or_cl + strlen(ptx_or_cl) - 3, ".cl") == 0) {
        strncpy(cl_path, ptx_or_cl, sizeof(cl_path)-1);
        cl_path[sizeof(cl_path)-1] = '\0';
    } else {
        ptx_to_cl_path(ptx_or_cl, cl_path, sizeof(cl_path));
    }

    cl_kernel kern = ocl_get_kernel(cl_path, kernel_name);
    if (!kern) return false;
    if (!ocl_set_args(kern, argc, argv)) return false;

    size_t global_sz[2] = { gx, gy };
    size_t local_sz[2]  = { lx, ly };
    cl_uint dims = (gy > 1) ? 2 : 1;

    cl_int err = g_ocl.clEnqueueNDRangeKernel(
        g_ocl.queue, kern, dims, NULL, global_sz, local_sz, 0, NULL, NULL);
    if (err != CL_SUCCESS) { ocl_set_error(err, "clEnqueueNDRangeKernel"); return false; }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * RUNTIME SELECTION
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum { RT_NONE, RT_CUDA, RT_OCL } Runtime;
static Runtime g_runtime = RT_NONE;
static bool    g_rt_init = false;

static Runtime runtime_select(void) {
    if (g_rt_init) return g_runtime;
    g_rt_init = true;
    if (cuda_load()) { g_runtime = RT_CUDA; return RT_CUDA; }
    if (ocl_load())  { g_runtime = RT_OCL;  return RT_OCL;  }
    g_runtime = RT_NONE;
    return RT_NONE;
}

static const char *last_error_str(void) {
    if (g_runtime == RT_CUDA) return g_cuda.last_error_str;
    if (g_runtime == RT_OCL)  return g_ocl.last_error_str;
    return "";
}

/* ── nanolang DynArray ABI (must match runtime/dyn_array.h) ─────────────── */
typedef struct {
    int64_t  length;
    int64_t  capacity;
    int32_t  elem_type;
    uint8_t  elem_size;
    uint8_t  _pad[3];
    void    *data;
} NLArray;

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC nl_gpu_* API
 * ═══════════════════════════════════════════════════════════════════════════ */

bool nl_gpu_available(void) { return runtime_select() != RT_NONE; }
bool nl_gpu_init(void)      { return runtime_select() != RT_NONE; }

int64_t nl_gpu_device_count(void) {
    if (runtime_select() == RT_CUDA) {
        int cnt = 0; g_cuda.cuDeviceGetCount(&cnt); return cnt;
    }
    return (g_runtime == RT_OCL) ? 1 : 0;
}

const char *nl_gpu_device_info(int64_t device) {
    static char buf[512];
    buf[0] = '\0';
    if (runtime_select() == RT_CUDA) {
        CUdevice dev; g_cuda.cuDeviceGet(&dev, (int)device);
        char name[256] = {0}; g_cuda.cuDeviceGetName(name, 255, dev);
        size_t mem = 0; g_cuda.cuDeviceTotalMem(&mem, dev);
        int maj = 0, min = 0; g_cuda.cuDeviceComputeCapability(&maj, &min, dev);
        snprintf(buf, sizeof(buf), "%s | %.1f GB | sm_%d%d",
                 name, (double)mem/(1024.*1024.*1024.), maj, min);
    } else if (g_runtime == RT_OCL) {
        char name[256] = {0};
        g_ocl.clGetDeviceInfo(g_ocl.device, CL_DEVICE_NAME, 255, name, NULL);
        snprintf(buf, sizeof(buf), "%s (OpenCL)", name);
    }
    return buf;
}

void nl_gpu_sync(void) {
    if (g_runtime == RT_CUDA) { g_cuda.cuCtxSynchronize(); return; }
    if (g_runtime == RT_OCL)  { g_ocl.clFinish(g_ocl.queue); }
}

int64_t nl_gpu_alloc(int64_t bytes) {
    if (runtime_select() == RT_NONE || bytes <= 0) return 0;
    if (g_runtime == RT_CUDA) {
        CUdeviceptr ptr = 0;
        CUresult r = g_cuda.cuMemAlloc(&ptr, (size_t)bytes);
        if (r != CUDA_SUCCESS) { cuda_set_error(r); return 0; }
        return (int64_t)ptr;
    }
    /* OpenCL: create cl_mem and return sentinel fake ptr */
    if (g_ocl_nallocs >= MAX_OCL_ALLOCS) {
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "nl_gpu_alloc: alloc table full (max %d)", MAX_OCL_ALLOCS);
        return 0;
    }
    cl_int err;
    cl_mem buf = g_ocl.clCreateBuffer(g_ocl.ctx, CL_MEM_READ_WRITE, (size_t)bytes, NULL, &err);
    if (err != CL_SUCCESS || !buf) { ocl_set_error(err, "clCreateBuffer"); return 0; }
    int idx = g_ocl_nallocs++;
    int64_t fake = OCL_SENTINEL | (int64_t)(idx & 0xFF);
    g_ocl_allocs[idx].fake = fake;
    g_ocl_allocs[idx].buf  = buf;
    return fake;
}

void nl_gpu_free(int64_t ptr) {
    if (g_runtime == RT_CUDA && ptr) { g_cuda.cuMemFree((CUdeviceptr)ptr); return; }
    if (g_runtime == RT_OCL) {
        cl_mem buf = ocl_find_buf(ptr);
        if (buf) {
            g_ocl.clReleaseMemObject(buf);
            /* Remove from alloc table */
            for (int i = 0; i < g_ocl_nallocs; i++) {
                if (g_ocl_allocs[i].fake == ptr) {
                    g_ocl_allocs[i] = g_ocl_allocs[--g_ocl_nallocs];
                    break;
                }
            }
        }
    }
}

bool nl_gpu_memcpy_to_device(int64_t dst, NLArray *src, int64_t bytes) {
    if (!src || !src->data || bytes <= 0) return false;
    if (runtime_select() == RT_NONE) return false;
    if (g_runtime == RT_CUDA) {
        CUresult r = g_cuda.cuMemcpyHtoD((CUdeviceptr)dst, src->data, (size_t)bytes);
        cuda_set_error(r); return r == CUDA_SUCCESS;
    }
    cl_mem buf = ocl_find_buf(dst);
    if (!buf) { snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                          "memcpy_to_device: unknown dst 0x%llx", (unsigned long long)dst); return false; }
    cl_int err = g_ocl.clEnqueueWriteBuffer(g_ocl.queue, buf, 1/*blocking*/, 0, (size_t)bytes,
                                             src->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) { ocl_set_error(err, "clEnqueueWriteBuffer"); return false; }
    return true;
}

bool nl_gpu_memcpy_from_device(NLArray *dst, int64_t src, int64_t bytes) {
    if (!dst || !dst->data || bytes <= 0) return false;
    if (runtime_select() == RT_NONE) return false;
    if (g_runtime == RT_CUDA) {
        CUresult r = g_cuda.cuMemcpyDtoH(dst->data, (CUdeviceptr)src, (size_t)bytes);
        cuda_set_error(r); return r == CUDA_SUCCESS;
    }
    cl_mem buf = ocl_find_buf(src);
    if (!buf) { snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                          "memcpy_from_device: unknown src 0x%llx", (unsigned long long)src); return false; }
    cl_int err = g_ocl.clEnqueueReadBuffer(g_ocl.queue, buf, 1/*blocking*/, 0, (size_t)bytes,
                                            dst->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) { ocl_set_error(err, "clEnqueueReadBuffer"); return false; }
    return true;
}

/* Common launcher used by gpu_launch and gpu_launch5 */
static bool do_launch1d(const char *kernel_file, const char *kernel_name,
                         int64_t grid_x, int64_t block_x,
                         int argc, int64_t *argv) {
    if (runtime_select() == RT_NONE) return false;
    if (g_runtime == RT_CUDA) {
        CUmodule mod = cuda_get_module(kernel_file);
        if (!mod) return false;
        CUfunction fn = NULL;
        CUresult r = g_cuda.cuModuleGetFunction(&fn, mod, kernel_name);
        if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }
        void *args[5];
        for (int i = 0; i < argc && i < 5; i++) args[i] = &argv[i];
        r = g_cuda.cuLaunchKernel(fn, (unsigned int)grid_x, 1, 1,
            (unsigned int)block_x, 1, 1, 0, NULL, args, NULL);
        cuda_set_error(r); return r == CUDA_SUCCESS;
    }
    /* OpenCL 1D */
    size_t global_sz = (size_t)(grid_x * block_x);
    size_t local_sz  = (size_t)block_x;
    return ocl_launch_nd(kernel_file, kernel_name, global_sz, 1, local_sz, 1, argc, argv);
}

bool nl_gpu_launch(const char *ptx_file, const char *kernel_name,
                   int64_t grid_x, int64_t block_x,
                   int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3) {
    int64_t argv[4] = {arg0, arg1, arg2, arg3};
    return do_launch1d(ptx_file, kernel_name, grid_x, block_x, 4, argv);
}

bool nl_gpu_launch5(const char *ptx_file, const char *kernel_name,
                    int64_t grid_x, int64_t block_x,
                    int64_t arg0, int64_t arg1, int64_t arg2,
                    int64_t arg3, int64_t arg4) {
    int64_t argv[5] = {arg0, arg1, arg2, arg3, arg4};
    return do_launch1d(ptx_file, kernel_name, grid_x, block_x, 5, argv);
}

bool nl_gpu_launch2d(const char *ptx_file, const char *kernel_name,
                     int64_t grid_x, int64_t grid_y,
                     int64_t block_x, int64_t block_y,
                     int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3) {
    if (runtime_select() == RT_NONE) return false;
    if (g_runtime == RT_CUDA) {
        CUmodule mod = cuda_get_module(ptx_file);
        if (!mod) return false;
        CUfunction fn = NULL;
        CUresult r = g_cuda.cuModuleGetFunction(&fn, mod, kernel_name);
        if (r != CUDA_SUCCESS) { cuda_set_error(r); return false; }
        void *args[] = { &arg0, &arg1, &arg2, &arg3 };
        r = g_cuda.cuLaunchKernel(fn,
            (unsigned int)grid_x, (unsigned int)grid_y, 1,
            (unsigned int)block_x, (unsigned int)block_y, 1,
            0, NULL, args, NULL);
        cuda_set_error(r); return r == CUDA_SUCCESS;
    }
    /* OpenCL 2D: total threads = grid * block per dimension */
    size_t gx = (size_t)(grid_x * block_x), gy = (size_t)(grid_y * block_y);
    size_t lx = (size_t)block_x,            ly = (size_t)block_y;
    int64_t argv[4] = {arg0, arg1, arg2, arg3};
    return ocl_launch_nd(ptx_file, kernel_name, gx, gy, lx, ly, 4, argv);
}

const char *nl_gpu_last_error(void) {
    return last_error_str();
}

int64_t nl_gpu_atomic_add(int64_t ptr, int64_t delta) {
    (void)ptr; (void)delta; return 0;
}

/* ── GPU intrinsic stubs (never called at runtime; satisfy linker) ──────── */
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
void    nl_gpu_barrier(void)     {}
int64_t nl_gpu_load(int64_t ptr) { (void)ptr; return 0; }
void    nl_gpu_store(int64_t ptr, int64_t val) { (void)ptr; (void)val; }
double  nl_gpu_load_float(int64_t ptr) { (void)ptr; return 0.0; }
void    nl_gpu_store_float(int64_t ptr, double val) { (void)ptr; (void)val; }
