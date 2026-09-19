#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <dlfcn.h>
#include <string.h>
static int loader_cookie;
static void *model_open(const char *path, int flags) {
    (void)path; (void)flags; return &loader_cookie;
}
static void *model_symbol(void *library, const char *name) {
    assert(library == &loader_cookie);
    return !strcmp(name, "clGetKernelInfo") ? NULL : &loader_cookie;
}
#define dlopen model_open
#define dlsym model_symbol
#include "../modules/gpu/opencl_runtime.c"

/* I model only the host interface here; these objects are not GPU resources. */
typedef struct { unsigned char data[32]; } ModelBuffer;
static int creates, releases, writes, reads, sets, enqueues, queries;
static int create_mode, release_error, query_error, set_error;
static cl_uint num_args = 4;
static size_t query_size = sizeof(cl_uint);
static cl_mem seen[5];
static ModelBuffer *quarantine;

static cl_mem create_buffer(cl_context ctx, cl_mem_flags flags, size_t bytes,
                            void *host, cl_int *error) {
    assert(ctx == g_ocl.ctx && flags == CL_MEM_READ_WRITE && !host && bytes <= 32);
    bool reserved = false;
    for (int i = 0; i < MAX_OCL_ALLOCS; ++i)
        reserved |= g_ocl_allocs[i].state == OCL_RESERVED;
    assert(reserved);
    ++creates;
    *error = create_mode == 1 || create_mode == 2 ? -5 : CL_SUCCESS;
    if (create_mode == 1 || create_mode == 3) return NULL;
    return calloc(1, sizeof(ModelBuffer));
}
static cl_int release_buffer(cl_mem buffer) {
    ++releases;
    for (int i = 0; i < MAX_OCL_ALLOCS; ++i)
        if (g_ocl_allocs[i].buf == buffer) assert(g_ocl_allocs[i].state != OCL_LIVE);
    if (release_error) { quarantine = buffer; return release_error; }
    free(buffer);
    return CL_SUCCESS;
}
static cl_int write_buffer(cl_command_queue queue, cl_mem buffer, cl_uint block,
                            size_t offset, size_t bytes, const void *source,
                            cl_uint n, const void *wait, void *event) {
    assert(queue == g_ocl.queue && block == 1 && offset == 0 && !n && !wait && !event);
    ++writes;
    memcpy(((ModelBuffer *)buffer)->data, source, bytes);
    return CL_SUCCESS;
}
static cl_int read_buffer(cl_command_queue queue, cl_mem buffer, cl_uint block,
                           size_t offset, size_t bytes, void *dest,
                           cl_uint n, const void *wait, void *event) {
    assert(queue == g_ocl.queue && block == 1 && offset == 0 && !n && !wait && !event);
    ++reads;
    memcpy(dest, ((ModelBuffer *)buffer)->data, bytes);
    return CL_SUCCESS;
}
static cl_int kernel_info(cl_kernel kernel, cl_uint key, size_t size,
                          void *result, size_t *returned) {
    assert(kernel && key == CL_KERNEL_NUM_ARGS && size == sizeof(cl_uint));
    ++queries;
    *(cl_uint *)result = num_args;
    *returned = query_size;
    return query_error;
}
static cl_int set_arg(cl_kernel kernel, cl_uint index, size_t size, const void *arg) {
    assert(kernel && index < 5);
    ++sets;
    if (size == sizeof(cl_mem)) memcpy(&seen[index], arg, sizeof(cl_mem));
    return set_error;
}
static cl_int enqueue(cl_command_queue queue, cl_kernel kernel, cl_uint dims,
                       const size_t *offset, const size_t *global, const size_t *local,
                       cl_uint n, const void *wait, void *event) {
    assert(queue == g_ocl.queue && kernel && dims == 1 && !offset);
    assert(global[0] == 1 && local[0] == 1 && !n && !wait && !event);
    ++enqueues;
    return CL_SUCCESS;
}
static void setup(void) {
    g_runtime = RT_OCL; g_rt_init = true;
    g_ocl.clCreateBuffer = create_buffer; g_ocl.clReleaseMemObject = release_buffer;
    g_ocl.clEnqueueWriteBuffer = write_buffer; g_ocl.clEnqueueReadBuffer = read_buffer;
    g_ocl.clGetKernelInfo = kernel_info; g_ocl.clSetKernelArg = set_arg;
    g_ocl.clEnqueueNDRangeKernel = enqueue;
    strcpy(g_ocl_kernels[0].cl_path, "model.cl");
    strcpy(g_ocl_kernels[0].kname, "ordinary");
    g_ocl_kernels[0].kernel = &g_ocl_kernels[0]; g_ocl_nkernels = 1;
}
static bool launch(int64_t *args, int count) {
    return ocl_launch_nd("model.cl", "ordinary", 1, 1, 1, 1, count, args);
}
static void ordinary(void) {
    int64_t tokens[256];
    int64_t input[4] = {0, 11, 22, 33}, output[4] = {0};
    NLArray src = {4, 4, ELEM_INT, sizeof(int64_t), input};
    NLArray dst = {4, 4, ELEM_INT, sizeof(int64_t), output};
    assert(!nl_gpu_alloc(-1) && !nl_gpu_alloc(0) && creates == 0);
    for (int i = 0; i < 256; ++i) {
        tokens[i] = nl_gpu_alloc(sizeof input);
        assert(tokens[i] && ocl_find_alloc(tokens[i]));
    }
    assert(!nl_gpu_alloc(8) && creates == 256);
    for (int i = 0; i < 256; ++i) {
        assert(nl_gpu_memcpy_to_device(tokens[i], &src, sizeof input));
        assert(nl_gpu_memcpy_from_device(&dst, tokens[i], sizeof output));
        assert(!memcmp(input, output, sizeof input));
    }
    for (int i = 0; i < 256; ++i) nl_gpu_free(tokens[i]);
    assert(releases == 256);
    for (int i = 0; i < 600; ++i) {
        int64_t token = nl_gpu_alloc(32);
        assert(token && token != tokens[0]);
        assert(nl_gpu_memcpy_to_device(token, &src, 32));
        nl_gpu_free(token);
    }
    assert(releases == 856);
    for (int i = 0; i < 256; ++i) assert(g_ocl_allocs[i].state == OCL_FREE);
}
static void arguments(void) {
    int64_t token = nl_gpu_alloc(8), args[5] = {token, 1, 2, 3, 4};
    assert(launch(args, 4) && sets == 4 && enqueues == 1 && queries == 1);
    assert(seen[0] == ocl_find_buf(token));
    int before = sets, launched = enqueues;
    num_args = 5; assert(!launch(args, 4));
    num_args = 3; assert(!launch(args, 4));
    num_args = 4; query_size = 0; assert(!launch(args, 4));
    query_size = sizeof(cl_uint); query_error = -5; assert(!launch(args, 4));
    query_error = 0;
    args[3] = (int64_t)OCL_SENTINEL; assert(!launch(args, 4));
    args[3] = token + 256; assert(!launch(args, 4));
    assert(sets == before && enqueues == launched);
    args[3] = 3; set_error = -5; assert(!launch(args, 4));
    assert(sets == before + 1 && enqueues == launched);
    set_error = 0; assert(launch(args, 4));
    assert(sets == before + 5 && enqueues == launched + 1);
    num_args = 5; assert(launch(args, 5));
    nl_gpu_free(token);
}
static void bounds(void) {
    int64_t token = nl_gpu_alloc(8), data[4] = {0};
    NLArray array = {4, 4, ELEM_INT, sizeof(int64_t), data};
    int64_t counts[] = {-1, 0, 9, INT64_MAX};
    for (size_t i = 0; i < sizeof counts / sizeof counts[0]; ++i) {
        assert(!nl_gpu_memcpy_to_device(token, &array, counts[i]));
        assert(!nl_gpu_memcpy_from_device(&array, token, counts[i]));
    }
    assert(!nl_gpu_memcpy_to_device(token + 256, &array, 8));
    assert(!nl_gpu_memcpy_from_device(&array, 7, 8));
    assert(writes == 0 && reads == 0);
    assert(nl_gpu_memcpy_to_device(token, &array, 8));
    nl_gpu_free(token);
    assert(!nl_gpu_memcpy_from_device(&array, token, 8));
    nl_gpu_free(token); nl_gpu_free(0);
    assert(releases == 1 && reads == 0);
}
static void exhaustion(void) {
    g_ocl_allocs[0].generation = OCL_GENERATION_MAX - 1;
    int64_t token = nl_gpu_alloc(8);
    assert(ocl_find_alloc(token) == &g_ocl_allocs[0]);
    assert(g_ocl_allocs[0].generation == OCL_GENERATION_MAX);
    nl_gpu_free(token);
    assert(g_ocl_allocs[0].state == OCL_EXHAUSTED);
    int64_t next = nl_gpu_alloc(8);
    assert(ocl_find_alloc(next) == &g_ocl_allocs[1]); nl_gpu_free(next);
}
static void allocation(int mode, bool unknown) {
    create_mode = mode; release_error = unknown ? -6 : 0;
    assert(!nl_gpu_alloc(8));
    OclAlloc *entry = &g_ocl_allocs[0];
    assert(entry->generation == 0 && !ocl_find_alloc(entry->token));
    assert(releases == (mode == 2));
    assert(entry->state == (unknown ? OCL_QUARANTINED : OCL_FREE));
    if (unknown) {
        assert(entry->release_attempted && entry->release_error == -6 && entry->buf);
        assert(strstr(nl_gpu_last_error(), "error -5; release -6"));
        free(quarantine); /* I reclaim only my host-model object, not a GPU. */
    }
    if (mode == 3) assert(strstr(nl_gpu_last_error(), "no OpenCL buffer"));
}
static void failed_release(void) {
    int64_t token = nl_gpu_alloc(8);
    release_error = -6; nl_gpu_free(token);
    OclAlloc *entry = &g_ocl_allocs[0];
    assert(entry->state == OCL_QUARANTINED && entry->buf == quarantine);
    assert(entry->release_attempted && entry->release_error == -6);
    nl_gpu_free(token); assert(releases == 1 && !ocl_find_alloc(token));
    release_error = 0;
    int64_t next = nl_gpu_alloc(8);
    assert(ocl_find_alloc(next) == &g_ocl_allocs[1]); nl_gpu_free(next);
    free(quarantine);
}
int main(int argc, char **argv) {
    assert(argc == 2); setup();
    if (!strcmp(argv[1], "loader")) {
        assert(!ocl_load() && !g_ocl.available && !g_ocl.clGetKernelInfo);
    }
    else if (!strcmp(argv[1], "ordinary")) ordinary();
    else if (!strcmp(argv[1], "arguments")) arguments();
    else if (!strcmp(argv[1], "bounds")) bounds();
    else if (!strcmp(argv[1], "exhaustion")) exhaustion();
    else if (!strcmp(argv[1], "allocation-error")) allocation(1, false);
    else if (!strcmp(argv[1], "allocation-null")) allocation(3, false);
    else if (!strcmp(argv[1], "rollback")) allocation(2, false);
    else if (!strcmp(argv[1], "rollback-unknown")) allocation(2, true);
    else if (!strcmp(argv[1], "release-unknown")) failed_release();
    else return 2;
    printf("I passed host-model %s: create=%d release=%d queries=%d sets=%d enqueue=%d\n",
           argv[1], creates, releases, queries, sets, enqueues);
    return 0;
}
