/* I dispatch only the exact length-aware builtin ABI. Legacy STRING calls
 * keep their existing symbols and representation. */
#include "../bstring_contracts.h"

static const NanoBstringContract *vm_bstring_import(const NvmModule *module, uint32_t index) {
    if (!module || index >= module->import_count || !module->imports || !module->string_lengths) return NULL;
    const NvmImportEntry *import = &module->imports[index];
    const char *space = nvm_get_string(module, import->module_name_idx);
    const char *name = nvm_get_string(module, import->function_name_idx);
    if (import->kind != NVM_IMPORT_FFI || !space || !name || space[0] ||
        module->string_lengths[import->module_name_idx] != 0 ||
        module->string_lengths[import->function_name_idx] != strlen(name)) return NULL;
    return nano_bstring_host_contract(name);
}

static bool vm_bstring_intrinsic(const NvmModule *module, uint32_t index,
                                const NanoBstringContract *contract,
                                NanoValue *args, int count, NanoValue *result,
                                VmHeap *heap, char *error, size_t error_size) {
    const NvmImportEntry *import = &module->imports[index];
    const uint8_t *types = module->import_param_types ? module->import_param_types[index] : NULL;
    if (!heap || !result || count != contract->argc || import->param_count != contract->argc ||
        import->return_type != contract->result || !types || !args) {
        snprintf(error, error_size, "I require the exact binary-string builtin signature");
        return false;
    }
    nl_string_t *values[3] = {0};
    for (int i = 0; i < count; ++i) {
        if (types[i] != contract->parameters[i] || args[i].tag != types[i]) {
            snprintf(error, error_size, "I require exact binary-string operand tags");
            return false;
        }
        if (types[i] == TAG_BSTRING) {
            VmBString *value = args[i].as.obj;
            if (!value || value->header.obj_type != TAG_BSTRING || !value->value) {
                snprintf(error, error_size, "I cannot use a released or missing binary string");
                return false;
            }
            values[i] = value->value;
        } else if ((types[i] == TAG_STRING && !args[i].as.string) ||
                   (types[i] == TAG_ARRAY && !args[i].as.array)) {
            snprintf(error, error_size, "I require a present binary-string input");
            return false;
        }
    }
    nl_string_t *owned = NULL;
    *result = val_void();
    switch (contract->operation) {
    case NANO_BSTR_NEW:
        owned = nl_string_new(vmstring_cstr(args[0].as.string));
        break;
    case NANO_BSTR_BINARY: {
        VmArray *array = args[0].as.array;
        if (array->elem_type != TAG_U8 || !array->length) {
            owned = nl_string_new_binary("", 0);
            break;
        }
        uint8_t *bytes = malloc(array->length);
        if (!bytes) break;
        for (uint32_t i = 0; i < array->length; ++i) {
            NanoValue value = vm_array_get(array, i);
            if (value.tag != TAG_U8) {
                free(bytes);
                snprintf(error, error_size, "I require consistent byte-array storage for a binary string");
                return false;
            }
            bytes[i] = value.as.u8;
        }
        owned = nl_string_new_binary(bytes, array->length);
        free(bytes);
        break;
    }
    case NANO_BSTR_LENGTH:
        if (values[0]->length > INT64_MAX) {
            snprintf(error, error_size, "I cannot represent this binary-string length as INT");
            return false;
        }
        *result = val_int((int64_t)nl_string_length(values[0]));
        return true;
    case NANO_BSTR_CONCAT:
        if (values[1]->length >= SIZE_MAX - values[0]->length) {
            snprintf(error, error_size, "I cannot overflow a binary-string concatenation");
            return false;
        }
        owned = nl_string_concat(values[0], values[1]);
        break;
    case NANO_BSTR_SUBSTRING: {
        int64_t start = args[1].as.i64, length = args[2].as.i64;
        if (start < 0 || length < 0) owned = nl_string_new("");
        else {
            size_t offset = (uint64_t)start > values[0]->length ? values[0]->length : (size_t)start;
            size_t available = values[0]->length - offset;
            size_t selected = (uint64_t)length > available ? available : (size_t)length;
            owned = nl_string_substring(values[0], offset, selected);
        }
        break;
    }
    case NANO_BSTR_EQUALS:
        *result = val_bool(nl_string_equals(values[0], values[1]));
        return true;
    case NANO_BSTR_BYTE: {
        int64_t offset = args[1].as.i64;
        *result = val_int(offset < 0 || (uint64_t)offset >= values[0]->length ? 0 :
                          (unsigned char)nl_string_byte_at(values[0], (size_t)offset));
        return true;
    }
    case NANO_BSTR_VALIDATE:
        *result = val_bool(nl_string_validate_utf8(values[0]));
        return true;
    case NANO_BSTR_UTF8_LENGTH:
        *result = val_int(nl_string_utf8_length(values[0]));
        return true;
    case NANO_BSTR_UTF8_CHAR:
        *result = val_int(args[1].as.i64 < 0 ? -1 :
                         nl_string_utf8_char_at(values[0], (size_t)args[1].as.i64));
        return true;
    case NANO_BSTR_CSTR: {
        size_t before = values[0]->capacity;
        const char *text = nl_string_to_cstr(values[0]);
        if (values[0]->capacity > before) heap->stats.allocated += values[0]->capacity - before;
        if (!text || !values[0]->null_terminated) {
            snprintf(error, error_size, "I could not terminate my binary-string conversion");
            return false;
        }
        size_t length = strlen(text);
        VmString *copy = text && length <= UINT32_MAX ? vm_string_new(heap, text, (uint32_t)length) : NULL;
        if (!copy) break;
        *result = val_string(copy);
        return true;
    }
    case NANO_BSTR_FREE:
        vm_bstring_clear(heap, args[0].as.obj);
        return true;
    }
    if (owned) {
        VmBString *value = vm_bstring_adopt(heap, owned);
        if (value) { *result = val_bstring(value); return true; }
    }
    snprintf(error, error_size, "I could not allocate a binary-string result");
    return false;
}

/* I cannot transfer heap-owned binary strings through the raw COP ABI. */
static bool vm_bstring_cop_refused(const NvmModule *module, uint32_t index,
                                  const NanoValue *args, int count,
                                  char *error, size_t size) {
    bool owned = vm_bstring_import(module, index) != NULL;
    if (module && index < module->import_count) {
        const NvmImportEntry *import = &module->imports[index];
        owned |= import->return_type == TAG_BSTRING;
        const uint8_t *types = module->import_param_types ? module->import_param_types[index] : NULL;
        for (uint16_t p = 0; types && p < import->param_count; ++p)
            owned |= types[p] == TAG_BSTRING;
    }
    for (int p = 0; args && p < count; ++p) owned |= args[p].tag == TAG_BSTRING;
    if (owned) snprintf(error, size, "I require owner-heap dispatch for binary strings; I cannot serialize their pointers");
    return owned;
}
