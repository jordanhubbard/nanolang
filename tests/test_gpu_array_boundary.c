#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#ifdef TEST_UNIFIED_GPU
#include "../modules/gpu/opencl_runtime.c"
#else
#include "../modules/gpu/cuda_runtime.c"
#endif

static int copies;
static int64_t device = 1;
static int64_t memory[2] = {11, 22};
static CUresult copy_to(CUdeviceptr dst, const void *src, size_t bytes) {
    assert(dst == 1 && src == memory && bytes <= sizeof memory);
    ++copies;
    return CUDA_SUCCESS;
}
static CUresult copy_from(void *dst, CUdeviceptr src, size_t bytes) {
    assert(src == 1 && dst == memory && bytes <= sizeof memory);
    ++copies;
    return CUDA_SUCCESS;
}

static void reject(DynArray *array, int64_t bytes) {
    int before = copies;
    assert(!nl_gpu_memcpy_to_device(device, array, bytes));
    assert(!nl_gpu_memcpy_from_device(array, device, bytes));
    assert(copies == before);
}

#ifdef TEST_UNIFIED_GPU
static cl_int copy_cl_to(cl_command_queue queue, cl_mem buffer, cl_uint blocking,
                         size_t offset, size_t bytes, const void *source,
                         cl_uint event_count, const void *waits, void *event) {
    assert(queue == g_ocl.queue && buffer == g_ocl_allocs[0].buf && blocking == 1);
    assert(offset == 0 && bytes <= sizeof memory && source == memory);
    assert(event_count == 0 && waits == NULL && event == NULL);
    ++copies;
    return CL_SUCCESS;
}
static cl_int copy_cl_from(cl_command_queue queue, cl_mem buffer, cl_uint blocking,
                           size_t offset, size_t bytes, void *destination,
                           cl_uint event_count, const void *waits, void *event) {
    return copy_cl_to(queue, buffer, blocking, offset, bytes, destination, event_count, waits, event);
}
#endif

int main(void) {
#ifdef TEST_UNIFIED_GPU
    g_rt_init = true;
    g_runtime = RT_CUDA;
#else
    g_cuda.initialized = g_cuda.available = true;
#endif
    g_cuda.cuMemcpyHtoD = copy_to;
    g_cuda.cuMemcpyDtoH = copy_from;
    DynArray array = {.length = 2, .capacity = 2, .elem_type = ELEM_INT,
                      .elem_size = sizeof(int64_t), .data = memory};
    assert(nl_gpu_memcpy_to_device(1, &array, 16));
    assert(nl_gpu_memcpy_from_device(&array, 1, 8));
    assert(nl_gpu_memcpy_to_device(1, &array, 1));
    assert(copies == 3);
    reject(&array, 17);
    reject(&array, INT64_MAX);
    reject(&array, 0);
    reject(&array, -1);
    reject(NULL, 8);
    array.elem_type = ELEM_STRING;
    reject(&array, 8);
    array.elem_type = ELEM_INT;
    array.elem_size = 1;
    reject(&array, 8);
    array.elem_size = sizeof(int64_t);
    array.length = -1;
    reject(&array, 8);
    array.length = 3;
    reject(&array, 8);
    array.length = array.capacity = INT64_MAX;
    reject(&array, 8);
    array.length = array.capacity = 2;
    array.data = NULL;
    reject(&array, 8);
#ifdef TEST_UNIFIED_GPU
    array.data = memory;
    g_runtime = RT_OCL;
    device = (int64_t)(OCL_SENTINEL | UINT64_C(0x100));
    g_ocl_allocs[0].state = OCL_LIVE;
    g_ocl_allocs[0].generation = 1;
    g_ocl_allocs[0].token = device;
    g_ocl_allocs[0].bytes = sizeof memory;
    g_ocl_allocs[0].buf = (cl_mem)(uintptr_t)1;
    g_ocl.clEnqueueWriteBuffer = copy_cl_to;
    g_ocl.clEnqueueReadBuffer = copy_cl_from;
    assert(nl_gpu_memcpy_to_device(device, &array, 16));
    assert(nl_gpu_memcpy_from_device(&array, device, 8));
    assert(copies == 5);
    reject(&array, 17);
    reject(&array, INT64_MAX);
    reject(&array, -1);
    array.elem_type = ELEM_STRING;
    reject(&array, 8);
#endif
    puts("I passed GPU host-array boundary tests without loading a driver.");
    return 0;
}
