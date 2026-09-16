#include <GL/glew.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "../src/runtime/dyn_array.h"

static int calls, fail_alloc;
static GLsizeiptr bytes;
static unsigned char copied[32];
static void GLAPIENTRY upload(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
    assert(target == 1 && usage == 2 && size >= 0 && (size_t)size <= sizeof copied);
    ++calls; bytes = size;
    if (size) { assert(data); memcpy(copied, data, (size_t)size); }
}
PFNGLBUFFERDATAPROC __glewBufferData = upload;
static void *allocate(size_t size) { return fail_alloc ? NULL : malloc(size); }
#define malloc allocate
#include "../modules/glew/glew_arrays.c"
#undef malloc

int main(void) {
    gc_init();
    assert(nl_gl3_buffer_data_f32__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_gl3_buffer_data_u32__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    DynArray *a = dyn_array_new(ELEM_FLOAT);
    dyn_array_push_float(a, 1.25); dyn_array_push_float(a, -2.5);
    nl_gl3_buffer_data_f32(1, a, 2);
    float floats[2]; memcpy(floats, copied, sizeof floats);
    assert(calls == 1 && bytes == sizeof floats && floats[0] == 1.25f && floats[1] == -2.5f);
    fail_alloc = 1; nl_gl3_buffer_data_f32(1, a, 2); assert(calls == 1);
    fail_alloc = 0;
    ((double *)a->data)[0] = DBL_MAX;
    nl_gl3_buffer_data_f32(1, a, 2); assert(calls == 1);
    gc_release(a);
    a = dyn_array_new(ELEM_INT);
    dyn_array_push_int(a, UINT32_MAX);
    nl_gl3_buffer_data_u32(1, a, 2);
    uint32_t integer; memcpy(&integer, copied, sizeof integer);
    assert(calls == 2 && integer == UINT32_MAX);
    for (int bad = 0; bad < 8; ++bad) {
        DynArray invalid = *a;
        int64_t value = bad == 0 ? -1 : (int64_t)UINT32_MAX + 1;
        if (bad < 2) invalid.data = &value;
        if (bad == 2) invalid.elem_type = ELEM_FLOAT;
        if (bad == 3) invalid.elem_size = 1;
        if (bad == 4) invalid.length = -1;
        if (bad == 5) invalid.capacity = 0;
        if (bad == 6) invalid.data = NULL;
        if (bad == 7) invalid.length = invalid.capacity = INT64_MAX;
        nl_gl3_buffer_data_u32(1, &invalid, 2);
        assert(calls == 2);
    }
    fail_alloc = 1; nl_gl3_buffer_data_u32(1, a, 2); assert(calls == 2);
    fail_alloc = 0;
    nl_gl3_buffer_data_u32(1, NULL, 2); assert(calls == 2);
    a->length = 0;
    nl_gl3_buffer_data_u32(1, a, 2); assert(calls == 3 && bytes == 0);
    gc_release(a);
    gc_shutdown();
    return 0;
}
