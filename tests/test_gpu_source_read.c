#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *phase;
static bool observe;
static int closes, allocated, freed, published, seeks;
static int model_seek(FILE *file, long offset, int origin) {
    if (observe) {
        ++seeks;
        if ((!strcmp(phase, "seek") && origin == SEEK_END) ||
            (!strcmp(phase, "rewind") && origin == SEEK_SET)) return -1;
    }
    return fseek(file, offset, origin);
}
static long model_tell(FILE *file) {
    if (observe && !strcmp(phase, "tell")) return -1;
    return ftell(file);
}
static size_t model_read(void *buffer, size_t size, size_t count, FILE *file) {
    size_t result = fread(buffer, size, count, file);
    if (observe && !strcmp(phase, "short")) return 0;
    return result;
}
static int model_error(FILE *file) {
    return observe && !strcmp(phase, "error") ? 1 : ferror(file);
}
static int model_close(FILE *file) {
    int result = fclose(file);
    if (observe) {
        ++closes;
        if (!strcmp(phase, "close")) return -1;
    }
    return result;
}
static void *model_malloc(size_t size) {
    if (observe && !strcmp(phase, "allocation")) return NULL;
    void *result = malloc(size);
    if (observe && result) ++allocated;
    return result;
}
static void model_free(void *ptr) {
    if (observe && ptr) ++freed;
    free(ptr);
}
#define fseek model_seek
#define ftell model_tell
#define fread model_read
#define ferror model_error
#define fclose model_close
#define malloc model_malloc
#define free model_free
#ifdef TEST_UNIFIED_GPU
#include "../modules/gpu/opencl_runtime.c"
#else
#include "../modules/gpu/cuda_runtime.c"
#endif

static int cookie;
static void check_source(const char *source) {
    assert(closes == 1 && allocated == 1);
    assert(!strcmp(source, !strcmp(phase, "empty") ? "" : "hello\n"));
    ++published;
}
static CUresult load_module(CUmodule *out, const void *source) {
    check_source(source); *out = &cookie; return CUDA_SUCCESS;
}
#ifdef TEST_UNIFIED_GPU
static cl_program create_program(cl_context ctx, cl_uint n, const char **sources,
                                  const size_t *sizes, cl_int *err) {
    assert(ctx == g_ocl.ctx && n == 1 && !sizes);
    check_source(sources[0]); *err = CL_SUCCESS; return &cookie;
}
static cl_int build_program(cl_program program, cl_uint n, const cl_device_id *devices,
                            const char *options, void (*notify)(cl_program, void*), void *data) {
    assert(program == &cookie && n == 1 && devices && !options && !notify && !data);
    return CL_SUCCESS;
}
static cl_kernel create_kernel(cl_program program, const char *name, cl_int *err) {
    assert(program == &cookie && !strcmp(name, "ordinary"));
    *err = CL_SUCCESS; return &cookie;
}
#endif
int main(int argc, char **argv) {
    assert(argc == 3); phase = argv[1];
    char path[] = "/tmp/nanolang-gpu-source-read-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    if (strcmp(phase, "empty")) assert(write(fd, "hello\n", 6) == 6);
    assert(close(fd) == 0);
    g_cuda.cuModuleLoadData = load_module;
    void *result = NULL;
    observe = true;
#ifdef TEST_UNIFIED_GPU
    if (!strcmp(argv[2], "opencl")) {
        g_ocl.clCreateProgramWithSource = create_program;
        g_ocl.clBuildProgram = build_program; g_ocl.clCreateKernel = create_kernel;
        result = ocl_get_kernel(path, "ordinary");
    } else result = cuda_get_module(path);
#else
    (void)argv;
    result = get_or_load_module(path);
#endif
    observe = false;
    assert(unlink(path) == 0);
    bool success = !strcmp(phase, "normal") || !strcmp(phase, "empty");
    assert((result != NULL) == success && published == (int)success);
    assert(closes == 1 && allocated == freed);
    printf("I passed source-reader %s/%s: close=%d allocated=%d freed=%d driver-publications=%d seeks=%d\n",
           argv[2], phase, closes, allocated, freed, published, seeks);
    return 0;
}
