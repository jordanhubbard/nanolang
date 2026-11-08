/* I keep managed owner shells alive until interpreter GC shutdown, like my
 * existing opaque handles. Explicit free clears only their shared payload. */
#include "../bstring_contracts.h"
#include "../runtime/nl_string.h"

typedef struct {
    nl_string_t *payload;
} EvalBString;

static void eval_bstring_finalize(void *pointer) {
    EvalBString *owner = pointer;
    nl_string_free(owner->payload);
    owner->payload = NULL;
}

static Value eval_bstring_borrow(void *pointer) {
    Value value = create_void();
    value.type = VAL_BSTRING;
    value.as.bstring_val = pointer;
    return value;
}

static Value eval_bstring_error(ASTNode *node, const char *message) {
    fprintf(stderr, "%s\n", message);
    if (!g_in_shadow_tests) exit(1);
    g_shadow_current_fail_count++;
    if (!g_shadow_current_first_line) {
        g_shadow_current_first_line = node->line;
        g_shadow_current_first_column = node->column;
    }
    return create_void();
}

static Value eval_bstring_intrinsic(ASTNode *node, const NanoBstringContract *contract,
                                   Value *args) {
    if (node->as.call.arg_count != contract->argc || (!args && contract->argc))
        return eval_bstring_error(node, "I require the declared binary-string builtin arity.");
    nl_string_t *values[3] = {0};
    for (int i = 0; i < contract->argc; ++i) {
        uint8_t type = contract->parameters[i];
        bool valid = false;
        if (type == TAG_STRING) valid = args[i].type == VAL_STRING && args[i].as.string_val;
        if (type == TAG_INT) valid = args[i].type == VAL_INT;
        if (type == TAG_ARRAY) valid = (args[i].type == VAL_ARRAY && args[i].as.array_val) ||
                                      (args[i].type == VAL_DYN_ARRAY && args[i].as.dyn_array_val);
        if (type == TAG_BSTRING && args[i].type == VAL_BSTRING &&
            args[i].as.bstring_val && gc_is_managed(args[i].as.bstring_val)) {
            GCHeader *header = gc_get_header(args[i].as.bstring_val);
            if (header->type == GC_TYPE_OPAQUE && header->finalizer == eval_bstring_finalize) {
                EvalBString *owner = args[i].as.bstring_val;
                values[i] = owner->payload;
                valid = values[i] != NULL;
            }
        }
        if (!valid) return eval_bstring_error(node, "I require live binary-string owners and exact operand types.");
    }
    nl_string_t *owned = NULL;
    switch (contract->operation) {
    case NANO_BSTR_NEW:
        owned = nl_string_new(args[0].as.string_val);
        break;
    case NANO_BSTR_BINARY: {
        size_t length = 0;
        bool bytes = false;
        Array *array = args[0].type == VAL_ARRAY ? args[0].as.array_val : NULL;
        DynArray *dynamic = args[0].type == VAL_DYN_ARRAY ? args[0].as.dyn_array_val : NULL;
        if (array) {
            if (array->length < 0 || array->length > array->capacity)
                return eval_bstring_error(node, "I require valid byte-array bounds.");
            length = (size_t)array->length;
            bytes = array->element_type == VAL_INT && array->is_byte_array;
        } else {
            int64_t size = dyn_array_length(dynamic);
            if (size < 0) return eval_bstring_error(node, "I require valid byte-array bounds.");
            length = (size_t)size;
            bytes = dyn_array_get_elem_type(dynamic) == ELEM_U8;
        }
        if (!bytes || !length) { owned = nl_string_new_binary("", 0); break; }
        if (array && !array->data) return eval_bstring_error(node, "I require present byte-array storage.");
        unsigned char *copy = malloc(length);
        if (!copy) break;
        for (size_t i = 0; i < length; ++i)
            copy[i] = array ? (unsigned char)((long long *)array->data)[i] : dyn_array_get_u8(dynamic, (int64_t)i);
        owned = nl_string_new_binary(copy, length);
        free(copy);
        break;
    }
    case NANO_BSTR_LENGTH:
        if (values[0]->length > INT64_MAX) return eval_bstring_error(node, "I cannot represent this binary-string length as INT.");
        return create_int((int64_t)values[0]->length);
    case NANO_BSTR_CONCAT:
        if (values[1]->length >= SIZE_MAX - values[0]->length)
            return eval_bstring_error(node, "I cannot overflow a binary-string concatenation.");
        owned = nl_string_concat(values[0], values[1]);
        break;
    case NANO_BSTR_SUBSTRING: {
        int64_t start = args[1].as.int_val, length = args[2].as.int_val;
        if (start < 0 || length < 0) owned = nl_string_new("");
        else {
            size_t offset = (uint64_t)start > values[0]->length ? values[0]->length : (size_t)start;
            size_t available = values[0]->length - offset;
            size_t selected = (uint64_t)length > available ? available : (size_t)length;
            owned = nl_string_substring(values[0], offset, selected);
        }
        break;
    }
    case NANO_BSTR_EQUALS: return create_bool(nl_string_equals(values[0], values[1]));
    case NANO_BSTR_BYTE: {
        int64_t offset = args[1].as.int_val;
        return create_int(offset < 0 || (uint64_t)offset >= values[0]->length ? 0 :
                          (unsigned char)nl_string_byte_at(values[0], (size_t)offset));
    }
    case NANO_BSTR_VALIDATE: return create_bool(nl_string_validate_utf8(values[0]));
    case NANO_BSTR_UTF8_LENGTH: return create_int(nl_string_utf8_length(values[0]));
    case NANO_BSTR_UTF8_CHAR:
        return create_int(args[1].as.int_val < 0 ? -1 : nl_string_utf8_char_at(values[0], (size_t)args[1].as.int_val));
    case NANO_BSTR_CSTR: {
        const char *text = nl_string_to_cstr(values[0]);
        if (!text || !values[0]->null_terminated)
            return eval_bstring_error(node, "I could not terminate my binary-string conversion.");
        return create_string(text);
    }
    case NANO_BSTR_FREE:
        eval_bstring_finalize(args[0].as.bstring_val);
        return create_void();
    }
    if (owned) {
        EvalBString *owner = gc_alloc_opaque(sizeof *owner, eval_bstring_finalize);
        if (owner) { owner->payload = owned; return eval_bstring_borrow(owner); }
        nl_string_free(owned);
    }
    return eval_bstring_error(node, "I could not allocate my binary-string result.");
}
