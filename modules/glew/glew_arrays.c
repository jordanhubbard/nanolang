#include <GL/glew.h>
#include "glew_wrappers.h"
#include "dyn_array.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

NANO_EXPORT_ARRAY_ABI(nl_gl3_buffer_data_f32);
NANO_EXPORT_ARRAY_ABI(nl_gl3_buffer_data_u32);

static bool upload_valid(const DynArray *a, ElementType type) {
    if (!a || a->length < 0 || (uint64_t)a->length > SIZE_MAX / sizeof(int64_t) ||
        (uint64_t)a->length > (uint64_t)PTRDIFF_MAX / sizeof(uint32_t)) return false;
    return dyn_array_has_storage(a, type, sizeof(int64_t), (uint64_t)a->length * sizeof(int64_t));
}

/* I leave the GL buffer unchanged on invalid input or allocation failure.
 * An empty, correctly typed array still requests an empty upload. */
void nl_gl3_buffer_data_f32(int64_t target, DynArray *data, int64_t usage) {
    if (!upload_valid(data, ELEM_FLOAT)) return;
    const double *source = data->data;
    for (int64_t i = 0; i < data->length; ++i)
        if (isfinite(source[i]) && (source[i] > FLT_MAX || source[i] < -FLT_MAX)) return;
    float *converted = data->length ? malloc((size_t)data->length * sizeof(float)) : NULL;
    if (data->length && !converted) return;
    for (int64_t i = 0; i < data->length; ++i) converted[i] = (float)source[i];
    glBufferData((GLenum)target, (GLsizeiptr)(data->length * sizeof(float)), converted, (GLenum)usage);
    free(converted);
}

void nl_gl3_buffer_data_u32(int64_t target, DynArray *data, int64_t usage) {
    if (!upload_valid(data, ELEM_INT)) return;
    const int64_t *source = data->data;
    for (int64_t i = 0; i < data->length; ++i)
        if (source[i] < 0 || (uint64_t)source[i] > UINT32_MAX) return;
    uint32_t *converted = data->length ? malloc((size_t)data->length * sizeof(uint32_t)) : NULL;
    if (data->length && !converted) return;
    for (int64_t i = 0; i < data->length; ++i) converted[i] = (uint32_t)source[i];
    glBufferData((GLenum)target, (GLsizeiptr)(data->length * sizeof(uint32_t)), converted, (GLenum)usage);
    free(converted);
}
