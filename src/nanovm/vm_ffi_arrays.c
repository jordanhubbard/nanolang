#include "vm_ffi_arrays.h"
#include "../utf8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool array_layout(uint8_t tag, ElementType *type, size_t *width) {
    switch (tag) {
        case TAG_INT: *type = ELEM_INT; *width = sizeof(int64_t); return true;
        case TAG_FLOAT: *type = ELEM_FLOAT; *width = sizeof(double); return true;
        case TAG_BOOL: *type = ELEM_BOOL; *width = sizeof(bool); return true;
        case TAG_U8: *type = ELEM_U8; *width = sizeof(uint8_t); return true;
        case TAG_STRING: *type = ELEM_STRING; *width = sizeof(char *); return true;
        default: return false;
    }
}

static bool array_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "I cannot marshal a foreign array: %s", message);
    return false;
}

bool vm_ffi_array_argument(VmFfiArrayFrame *frame, NanoValue value, void **out,
                           char *error, size_t size) {
    if (value.tag != TAG_ARRAY || !value.as.array)
        return array_error(error, size, "I require a live array argument");
    VmArray *original = value.as.array;
    for (int i = 0; i < frame->count; ++i) {
        if (frame->entries[i].original == original) {
            *out = frame->entries[i].native;
            return true;
        }
    }
    ElementType type;
    size_t width;
    if (!array_layout(original->elem_type, &type, &width))
        return array_error(error, size, "the element layout is unsupported");
    if (original->unboxed != vm_array_type_unboxable(original->elem_type))
        return array_error(error, size, "the VM array storage does not match its element type");
    if (frame->count == NANO_MAX_FFI_ARGS)
        return array_error(error, size, "too many distinct arrays");
    VmFfiArrayEntry *entry = &frame->entries[frame->count++];
    entry->original = original;
    entry->element_tag = original->elem_type;
    vm_retain(frame->heap, value);
    entry->native = dyn_array_new_with_capacity(type, original->length);
    if (!entry->native) return array_error(error, size, "native allocation failed");
    if (type == ELEM_STRING && original->length) {
        entry->strings = calloc(original->length, sizeof(char *));
        if (!entry->strings) return array_error(error, size, "string snapshot allocation failed");
    }
    for (uint32_t i = 0; i < original->length; ++i) {
        NanoValue element = vm_array_get(original, i);
        if (element.tag != original->elem_type)
            return array_error(error, size, "an element disagrees with its array type");
        switch (type) {
            case ELEM_INT: dyn_array_push_int(entry->native, element.as.i64); break;
            case ELEM_FLOAT: dyn_array_push_float(entry->native, element.as.f64); break;
            case ELEM_BOOL: dyn_array_push_bool(entry->native, element.as.boolean); break;
            case ELEM_U8: dyn_array_push_u8(entry->native, element.as.u8); break;
            case ELEM_STRING: {
                const char *text = element.as.string ? vmstring_cstr(element.as.string) : "";
                uint32_t length = element.as.string ? vmstring_len(element.as.string) : 0;
                if ((uint64_t)length + 1 > SIZE_MAX || strlen(text) != length ||
                    !nl_utf8_validate(text, length, NULL))
                    return array_error(error, size, "I require NUL-free UTF-8 string elements");
                char *copy = malloc((size_t)length + 1);
                if (!copy) return array_error(error, size, "string snapshot allocation failed");
                memcpy(copy, text, (size_t)length + 1);
                entry->strings[entry->string_count++] = copy;
                dyn_array_push_string(entry->native, copy);
                break;
            }
            default: return array_error(error, size, "the element layout is unsupported");
        }
    }
    *out = entry->native;
    return true;
}

VmArray *vm_ffi_array_import(VmHeap *heap, const DynArray *array, char *error, size_t size) {
    uint8_t tag = TAG_VOID;
    if (array) switch (array->elem_type) {
        case ELEM_INT: tag = TAG_INT; break;
        case ELEM_FLOAT: tag = TAG_FLOAT; break;
        case ELEM_BOOL: tag = TAG_BOOL; break;
        case ELEM_U8: tag = TAG_U8; break;
        case ELEM_STRING: tag = TAG_STRING; break;
        default: break;
    }
    ElementType type;
    size_t width;
    if (!array || !array_layout(tag, &type, &width) || array->elem_size != width ||
        array->length < 0 || (uint64_t)array->length > UINT32_MAX ||
        array->capacity < array->length || (uint64_t)array->length > SIZE_MAX / width ||
        (array->length && !array->data)) {
        array_error(error, size, "invalid native metadata or unsupported element layout");
        return NULL;
    }
    VmArray *copy = vm_array_new(heap, tag, (uint32_t)array->length);
    if (!copy) { array_error(error, size, "VM allocation failed"); return NULL; }
    for (uint32_t i = 0; i < (uint32_t)array->length; ++i) {
        NanoValue element = val_void();
        switch (type) {
            case ELEM_INT: element = val_int(((int64_t *)array->data)[i]); break;
            case ELEM_FLOAT: element = val_float(((double *)array->data)[i]); break;
            case ELEM_BOOL: element = val_bool(((uint8_t *)array->data)[i] != 0); break;
            case ELEM_U8: element = val_u8(((uint8_t *)array->data)[i]); break;
            case ELEM_STRING: {
                const char *text = ((const char **)array->data)[i];
                if (!text) text = "";
                size_t length = strlen(text);
                if (length > UINT32_MAX || !nl_utf8_validate(text, length, NULL)) {
                    array_error(error, size, "invalid native UTF-8 string");
                    vm_release(heap, val_array(copy));
                    return NULL;
                }
                VmString *string = vm_string_new(heap, text, (uint32_t)length);
                if (!string) {
                    array_error(error, size, "VM string allocation failed");
                    vm_release(heap, val_array(copy));
                    return NULL;
                }
                element = val_string(string);
                break;
            }
            default: break;
        }
        bool pushed = vm_array_push(heap, copy, element);
        vm_release(heap, element);
        if (!pushed) {
            array_error(error, size, "VM array growth failed");
            vm_release(heap, val_array(copy));
            return NULL;
        }
    }
    return copy;
}

bool vm_ffi_arrays_commit(VmFfiArrayFrame *frame, char *error, size_t size) {
    for (int i = 0; i < frame->count; ++i) {
        VmFfiArrayEntry *entry = &frame->entries[i];
        entry->pending = vm_ffi_array_import(frame->heap, entry->native, error, size);
        if (!entry->pending) return false;
        if (entry->pending->elem_type != entry->element_tag ||
            entry->original->elem_type != entry->element_tag)
            return array_error(error, size, "the foreign call changed the element type");
    }
    /* I publish only after every conversion succeeds. Native side effects
     * cannot be rolled back if validation or allocation fails after the call. */
    for (int i = 0; i < frame->count; ++i)
        vm_array_swap_scalar_storage(frame->entries[i].original, frame->entries[i].pending);
    return true;
}

bool vm_ffi_array_alias_result(VmFfiArrayFrame *frame, const void *pointer, NanoValue *out) {
    for (int i = 0; i < frame->count; ++i) {
        if (pointer == frame->entries[i].native) {
            *out = val_array(frame->entries[i].original);
            vm_retain(frame->heap, *out);
            return true;
        }
    }
    return false;
}

void vm_ffi_arrays_dispose(VmFfiArrayFrame *frame) {
    for (int i = 0; i < frame->count; ++i) {
        VmFfiArrayEntry *entry = &frame->entries[i];
        if (entry->pending) vm_release(frame->heap, val_array(entry->pending));
        if (entry->native) gc_release(entry->native);
        for (uint32_t j = 0; j < entry->string_count; ++j) free(entry->strings[j]);
        free(entry->strings);
        if (entry->original) vm_release(frame->heap, val_array(entry->original));
        memset(entry, 0, sizeof *entry);
    }
    frame->count = 0;
}
