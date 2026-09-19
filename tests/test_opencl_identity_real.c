#define _GNU_SOURCE
#include <assert.h>
#include "../modules/gpu/opencl_runtime.c"

static void device_text(cl_uint key, const char *label) {
    char value[512] = {0};
    assert(g_ocl.clGetDeviceInfo(g_ocl.device, key, sizeof value, value, NULL) == CL_SUCCESS);
    assert(memchr(value, 0, sizeof value));
    printf("DEVICE %s=%s\n", label, value);
}
int main(void) {
    /* I select OpenCL explicitly in this fixture; production selection is unchanged. */
    if (!ocl_load()) {
        fprintf(stderr, "I cannot qualify an actual OpenCL GPU: %s\n", g_ocl.last_error_str);
        return 77;
    }
    uint64_t type = 0;
    assert(g_ocl.clGetDeviceInfo(g_ocl.device, 0x1000, sizeof type, &type, NULL) == CL_SUCCESS);
    if (!(type & CL_DEVICE_TYPE_GPU)) {
        fprintf(stderr, "I refuse CPU fallback as OpenCL GPU acceptance: type=%llu\n",
                (unsigned long long)type);
        return 77;
    }
    printf("DEVICE type=%llu GPU=true\n", (unsigned long long)type);
    device_text(CL_DEVICE_NAME, "name"); device_text(0x102c, "vendor");
    device_text(0x102d, "driver"); device_text(0x102f, "version");
    cl_platform_id platform = NULL;
    assert(g_ocl.clGetDeviceInfo(g_ocl.device, 0x1031, sizeof platform, &platform, NULL) == CL_SUCCESS);
    char name[512] = {0};
    assert(g_ocl.clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof name, name, NULL) == CL_SUCCESS);
    assert(memchr(name, 0, sizeof name)); printf("PLATFORM name=%s\n", name);
    Dl_info library;
    assert(dladdr((void *)g_ocl.clGetKernelInfo, &library));
    printf("LIBRARY query=%s\n", library.dli_fname);
    g_rt_init = true; g_runtime = RT_OCL;
    int64_t input[16], output[16];
    for (int i = 0; i < 16; ++i) input[i] = i * 11;
    memset(output, 0, sizeof output);
    NLArray src = {16, 16, ELEM_INT, sizeof(int64_t), input};
    NLArray dst = {16, 16, ELEM_INT, sizeof(int64_t), output};
    for (int pass = 0; pass < 24; ++pass) {
        int64_t source = nl_gpu_alloc(sizeof input), destination = nl_gpu_alloc(sizeof output);
        assert(source && destination && source != destination);
        assert(nl_gpu_memcpy_to_device(source, &src, sizeof input));
        assert(nl_gpu_launch("tests/gpu/identity.cl", "identity_add", 1, 16,
                             source, destination, 16, pass));
        assert(g_ocl.clFinish(g_ocl.queue) == CL_SUCCESS);
        assert(nl_gpu_memcpy_from_device(&dst, destination, sizeof output));
        for (int i = 0; i < 16; ++i) assert(output[i] == input[i] + pass);
        nl_gpu_free(source); nl_gpu_free(destination);
        assert(g_ocl_allocs[0].release_attempted && g_ocl_allocs[0].release_error == CL_SUCCESS);
        assert(g_ocl_allocs[1].release_attempted && g_ocl_allocs[1].release_error == CL_SUCCESS);
    }
    for (int i = 0; i < MAX_OCL_ALLOCS; ++i) assert(g_ocl_allocs[i].state == OCL_FREE);
    /* The legacy API has no shutdown: I explicitly clean fixture-owned runtime
     * cache/context roots, without claiming a production shutdown API. */
    for (int i = 0; i < g_ocl_nkernels; ++i) {
        assert(g_ocl.clReleaseKernel(g_ocl_kernels[i].kernel) == CL_SUCCESS);
        assert(g_ocl.clReleaseProgram(g_ocl_kernels[i].prog) == CL_SUCCESS);
    }
    pfn_clReleaseCommandQueue release_queue = (pfn_clReleaseCommandQueue)dlsym(g_ocl.lib, "clReleaseCommandQueue");
    pfn_clReleaseContext release_context = (pfn_clReleaseContext)dlsym(g_ocl.lib, "clReleaseContext");
    assert(release_queue && release_context);
    assert(release_queue(g_ocl.queue) == CL_SUCCESS);
    assert(release_context(g_ocl.ctx) == CL_SUCCESS);
    assert(dlclose(g_ocl.lib) == 0);
    puts("I passed actual OpenCL GPU: 48 buffers, 24 kernels, 384 exact integer observations; fixture cache/context cleanup explicit.");
    return 0;
}
