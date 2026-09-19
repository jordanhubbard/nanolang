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
 *     as cl_mem (buffer) args. I reserve prefix 0x0C1A for buffer tokens;
 *     unknown tokens in that interval refuse, and other integers are scalars.
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
    if (!f) {
        snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
                 "I cannot open GPU source: %.200s", ptx_file);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
                 "I cannot seek GPU source: %.200s", ptx_file);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0 || (uintmax_t)sz >= (uintmax_t)SIZE_MAX ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
                 "I cannot size or rewind GPU source: %.200s", ptx_file);
        return NULL;
    }
    char *src = malloc((size_t)sz + 1);
    if (!src) {
        fclose(f);
        snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
                 "I cannot allocate GPU source storage");
        return NULL;
    }
    size_t count = fread(src, 1, (size_t)sz, f);
    int read_error = ferror(f);
    int close_error = fclose(f);
    if (count != (size_t)sz || read_error || close_error != 0) {
        free(src);
        snprintf(g_cuda.last_error_str, sizeof(g_cuda.last_error_str),
                 "I cannot completely read and close GPU source: %.200s", ptx_file);
        return NULL;
    }
    src[(size_t)sz] = '\0';
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
#define CL_KERNEL_NUM_ARGS       0x1191

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
typedef cl_int (*pfn_clGetKernelInfo)(cl_kernel, cl_uint, size_t, void*, size_t*);
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
    pfn_clGetKernelInfo            clGetKernelInfo;
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
    LOAD_OCL(clGetKernelInfo);
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

/* I retain stable slots and never wrap a published identity's generation.
 * The integer ABI reserves prefix 0x0C1A; it cannot distinguish a scalar
 * numerically equal to a live token from that buffer. Calls are serialized. */
#define OCL_SENTINEL UINT64_C(0x0C1A000000000000)
#define OCL_PREFIX_MASK UINT64_C(0xffff000000000000)
#define OCL_GENERATION_MAX UINT64_C(0xffffffffff)
#define MAX_OCL_ALLOCS 256

typedef enum { OCL_FREE, OCL_RESERVED, OCL_LIVE,
               OCL_QUARANTINED, OCL_EXHAUSTED } OclAllocState;
typedef struct {
    OclAllocState state;
    uint64_t generation;
    int64_t token;
    cl_mem buf;
    size_t bytes;
    bool release_attempted;
    cl_int release_error;
} OclAlloc;
static OclAlloc g_ocl_allocs[MAX_OCL_ALLOCS];

static bool ocl_reserved_token(int64_t value) {
    return ((uint64_t)value & OCL_PREFIX_MASK) == OCL_SENTINEL;
}

static OclAlloc *ocl_find_alloc(int64_t value) {
    if (!ocl_reserved_token(value)) return NULL;
    OclAlloc *entry = &g_ocl_allocs[(uint64_t)value & UINT64_C(0xff)];
    if (entry->state != OCL_LIVE || !entry->generation ||
        entry->token != value) return NULL;
    return entry;
}

static cl_mem ocl_find_buf(int64_t value) {
    OclAlloc *entry = ocl_find_alloc(value);
    return entry ? entry->buf : NULL;
}

/* I invalidate before one release attempt. Unknown outcomes keep their slot
 * and raw object until process exit, without a retry or closure claim. */
static cl_int ocl_release_alloc(OclAlloc *entry) {
    entry->state = OCL_QUARANTINED;
    entry->release_attempted = true;
    entry->release_error = g_ocl.clReleaseMemObject(entry->buf);
    if (entry->release_error == CL_SUCCESS) {
        entry->buf = NULL;
        entry->state = entry->generation == OCL_GENERATION_MAX
                     ? OCL_EXHAUSTED : OCL_FREE;
    }
    return entry->release_error;
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
                 "I cannot open GPU source: %.200s", cl_path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "I cannot seek GPU source: %.200s", cl_path);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0 || (uintmax_t)sz >= (uintmax_t)SIZE_MAX ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "I cannot size or rewind GPU source: %.200s", cl_path);
        return NULL;
    }
    char *src = malloc((size_t)sz + 1);
    if (!src) {
        fclose(f);
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "I cannot allocate GPU source storage");
        return NULL;
    }
    size_t count = fread(src, 1, (size_t)sz, f);
    int read_error = ferror(f);
    int close_error = fclose(f);
    if (count != (size_t)sz || read_error || close_error != 0) {
        free(src);
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "I cannot completely read and close GPU source: %.200s", cl_path);
        return NULL;
    }
    src[(size_t)sz] = '\0';

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
                         "clBuildProgram failed:\n%.200s", log);
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
    if (argc < 0 || argc > 5 || (argc && !argv)) return false;
    cl_uint actual_argc = 0;
    size_t actual_size = 0;
    cl_int query_error = g_ocl.clGetKernelInfo(kern, CL_KERNEL_NUM_ARGS,
                                              sizeof(actual_argc), &actual_argc,
                                              &actual_size);
    if (query_error != CL_SUCCESS) {
        ocl_set_error(query_error, "clGetKernelInfo");
        return false;
    }
    if (actual_size != sizeof(actual_argc) || actual_argc != (cl_uint)argc) {
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "I require the complete OpenCL kernel argument list");
        return false;
    }
    for (int i = 0; i < argc; ++i) {
        if (ocl_reserved_token(argv[i]) && !ocl_find_alloc(argv[i])) {
            snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                     "I require a live OpenCL buffer token for argument %d", i);
            return false;
        }
    }
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

#include "../../src/runtime/dyn_array.h"
typedef DynArray NLArray;
NANO_EXPORT_ARRAY_ABI(nl_gpu_memcpy_to_device);
NANO_EXPORT_ARRAY_ABI(nl_gpu_memcpy_from_device);

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
    if (bytes <= 0 || (uint64_t)bytes > SIZE_MAX) return 0;
    if (runtime_select() == RT_NONE) return 0;
    if (g_runtime == RT_CUDA) {
        CUdeviceptr ptr = 0;
        CUresult r = g_cuda.cuMemAlloc(&ptr, (size_t)bytes);
        if (r != CUDA_SUCCESS) { cuda_set_error(r); return 0; }
        return (int64_t)ptr;
    }
    OclAlloc *entry = NULL;
    int slot = 0;
    for (; slot < MAX_OCL_ALLOCS; ++slot) {
        if (g_ocl_allocs[slot].state == OCL_FREE) {
            entry = &g_ocl_allocs[slot];
            break;
        }
    }
    if (!entry) {
        snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                 "I have no available OpenCL allocation slot");
        return 0;
    }
    /* I reserve before entering the driver and publish only complete success. */
    entry->state = OCL_RESERVED;
    entry->release_attempted = false;
    entry->release_error = CL_SUCCESS;
    entry->bytes = (size_t)bytes;
    cl_int err = CL_SUCCESS;
    entry->buf = g_ocl.clCreateBuffer(g_ocl.ctx, CL_MEM_READ_WRITE,
                                     entry->bytes, NULL, &err);
    if (err != CL_SUCCESS || !entry->buf) {
        if (entry->buf) {
            cl_int cleanup = ocl_release_alloc(entry);
            snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                     "I could not create an OpenCL buffer (error %d; release %d)",
                     err, cleanup);
        } else {
            entry->state = OCL_FREE;
            if (err == CL_SUCCESS)
                snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                         "I received no OpenCL buffer from a successful allocation");
            else ocl_set_error(err, "clCreateBuffer");
        }
        return 0;
    }
    ++entry->generation; /* FREE excludes exhausted and quarantined slots. */
    entry->token = (int64_t)(OCL_SENTINEL | (entry->generation << 8) |
                             (uint64_t)slot);
    entry->state = OCL_LIVE;
    return entry->token;
}

void nl_gpu_free(int64_t ptr) {
    if (!ptr) return;
    if (g_runtime == RT_CUDA) { g_cuda.cuMemFree((CUdeviceptr)ptr); return; }
    if (g_runtime == RT_OCL) {
        OclAlloc *entry = ocl_find_alloc(ptr);
        if (!entry) {
            snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                     "I require a live OpenCL buffer token to free");
            return;
        }
        cl_int err = ocl_release_alloc(entry);
        if (err != CL_SUCCESS) ocl_set_error(err, "clReleaseMemObject");
    }
}

bool nl_gpu_memcpy_to_device(int64_t dst, NLArray *src, int64_t bytes) {
    if (bytes <= 0 || (uint64_t)bytes > SIZE_MAX || !dyn_array_has_storage(src, ELEM_INT, sizeof(int64_t), (uint64_t)bytes)) return false;
    if (runtime_select() == RT_NONE) return false;
    if (g_runtime == RT_CUDA) {
        CUresult r = g_cuda.cuMemcpyHtoD((CUdeviceptr)dst, src->data, (size_t)bytes);
        cuda_set_error(r); return r == CUDA_SUCCESS;
    }
    OclAlloc *entry = ocl_find_alloc(dst);
    if (!entry || (uint64_t)bytes > entry->bytes) { snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                          "memcpy_to_device: I require a live destination buffer with sufficient bytes: 0x%llx", (unsigned long long)dst); return false; }
    cl_int err = g_ocl.clEnqueueWriteBuffer(g_ocl.queue, entry->buf, 1/*blocking*/, 0, (size_t)bytes,
                                             src->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) { ocl_set_error(err, "clEnqueueWriteBuffer"); return false; }
    return true;
}

bool nl_gpu_memcpy_from_device(NLArray *dst, int64_t src, int64_t bytes) {
    if (bytes <= 0 || (uint64_t)bytes > SIZE_MAX || !dyn_array_has_storage(dst, ELEM_INT, sizeof(int64_t), (uint64_t)bytes)) return false;
    if (runtime_select() == RT_NONE) return false;
    if (g_runtime == RT_CUDA) {
        CUresult r = g_cuda.cuMemcpyDtoH(dst->data, (CUdeviceptr)src, (size_t)bytes);
        cuda_set_error(r); return r == CUDA_SUCCESS;
    }
    OclAlloc *entry = ocl_find_alloc(src);
    if (!entry || (uint64_t)bytes > entry->bytes) { snprintf(g_ocl.last_error_str, sizeof(g_ocl.last_error_str),
                          "memcpy_from_device: I require a live source buffer with sufficient bytes: 0x%llx", (unsigned long long)src); return false; }
    cl_int err = g_ocl.clEnqueueReadBuffer(g_ocl.queue, entry->buf, 1/*blocking*/, 0, (size_t)bytes,
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
