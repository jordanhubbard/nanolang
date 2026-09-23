#include "portable_host_plan.h"
#include "isa.h"
#include "service_bindings_module.h"
#include "verifier.h"
#include "../nanovm/vm_decode.h"
#include <stdlib.h>
#include <string.h>

struct NvmPortableReadPlan {
    NvmPortableReadCounts counts;
    NvmPortableReadImport imports[NVM_PORTABLE_READ_MAX_IMPORTS];
};

static NvmPortableReadResult result(NvmPortableReadStatus status,
        uint32_t function, uint32_t pc, uint32_t import, const char *message) {
    return (NvmPortableReadResult){status, function, pc, import, message};
}
#define NONE NVM_PORTABLE_READ_NO_INDEX
#define STOP(status, message) return result(status, NONE, NONE, NONE, message)

static bool charge(size_t *used, size_t count, size_t width) {
    if (*used > NVM_PORTABLE_READ_MAX_BYTES ||
        (width && count > (NVM_PORTABLE_READ_MAX_BYTES - *used) / width))
        return false;
    *used += count * width;
    return true;
}

static bool scalar_string(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_U8 || tag == TAG_FLOAT ||
           tag == TAG_BOOL || tag == TAG_STRING;
}

static bool unsupported_opcode(uint8_t op) {
    switch (op) {
    case OP_CALL_MODULE: case OP_CALL_INDIRECT: case OP_CLOSURE_NEW:
    case OP_LOAD_UPVALUE: case OP_STORE_UPVALUE:
    case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
    case OP_OWN_PACK: case OP_OWN_UNPACK_LOCAL: case OP_OWN_UNPACK_VARIANT: case OP_CALL_REF:
    case OP_REGION_BEGIN: case OP_REGION_END:
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE:
    case OP_REF_GET: case OP_REF_SET:
    case OP_BORROW_PATH_SHARED: case OP_BORROW_PATH_EXCLUSIVE:
    case OP_REBORROW_SHARED: case OP_REBORROW_EXCLUSIVE:
        return true;
    default: return false;
    }
}

static bool read_name(const NvmModule *m, uint32_t index) {
    static const char *const names[] = {
        "file_read", "vm_file_read", "nl_os_file_read"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        size_t length = strlen(names[i]);
        if (m->string_lengths[index] == length &&
            !memcmp(m->strings[index], names[i], length)) return true;
    }
    return false;
}

NvmPortableReadResult nvm_portable_read_plan(const NvmModule *m,
                                            NvmPortableReadPlan **out) {
    if (!out || !m)
        STOP(NVM_PORTABLE_READ_INVALID, "I require a module and plan output.");
    if (m->function_count > NVM_PORTABLE_READ_MAX_FUNCTIONS ||
        m->import_count > NVM_PORTABLE_READ_MAX_IMPORTS ||
        m->string_count > NVM_MAX_STRINGS)
        STOP(NVM_PORTABLE_READ_LIMIT, "I exceed my private table limits.");
    if ((m->function_count && !m->functions) ||
        (m->import_count && (!m->imports || !m->import_param_types)) ||
        (m->string_count && (!m->strings || !m->string_lengths)) ||
        (m->code_size && !m->code) ||
        (m->metadata_count && !m->metadata) ||
        (m->debug_count && !m->debug_entries))
        STOP(NVM_PORTABLE_READ_INVALID, "I require complete table storage.");

    size_t bytes = sizeof(*m);
    if (!charge(&bytes, m->code_size, 1) ||
        !charge(&bytes, m->function_count, sizeof(*m->functions) + sizeof(uint8_t *)) ||
        !charge(&bytes, m->import_count, sizeof(*m->imports) + sizeof(uint8_t *)) ||
        !charge(&bytes, m->string_count, sizeof(char *) + sizeof(uint32_t)) ||
        !charge(&bytes, m->metadata_count, sizeof(*m->metadata)) ||
        !charge(&bytes, m->debug_count, sizeof(*m->debug_entries)))
        STOP(NVM_PORTABLE_READ_LIMIT, "I exceed my represented module byte limit.");
    /* I establish safe scan ranges before the service-priority predicate. */
    for (uint32_t f = 0; f < m->function_count; f++) {
        const NvmFunctionEntry *fn = &m->functions[f];
        if (fn->code_offset > m->code_size ||
            fn->code_length > m->code_size - fn->code_offset)
            return result(NVM_PORTABLE_READ_INVALID, f, NONE, NONE,
                          "I require an in-range function body.");
        if (fn->code_length) for (uint32_t earlier = 0; earlier < f; earlier++) {
            const NvmFunctionEntry *other = &m->functions[earlier];
            if (other->code_length && fn->code_offset < other->code_offset + other->code_length &&
                other->code_offset < fn->code_offset + fn->code_length)
                return result(NVM_PORTABLE_READ_INVALID, f, NONE, NONE,
                              "I require disjoint function bodies before scanning.");
        }
    }
    if ((nvm_capture_bindings_present(m) || nvm_service_execution_pending(m)))
        STOP(NVM_PORTABLE_READ_UNSUPPORTED, "I require service and capture admission before portable execution.");
    if (m->ownership_data || m->ownership_size || m->layout_data || m->layout_size ||
        m->passive_data || m->passive_size || m->module_ref_count ||
        m->callback_contract_count ||
        m->struct_count || m->enum_count || m->union_count)
        STOP(NVM_PORTABLE_READ_UNSUPPORTED, "I require a non-nominal unlinked declaration envelope.");
    if (!nvm_validate_header(&m->header) || m->section_count > NVM_MAX_SECTIONS ||
        (m->header.flags & ~(NVM_FLAG_HAS_MAIN | NVM_FLAG_NEEDS_EXTERN | NVM_FLAG_DEBUG_INFO)) ||
        !(m->header.flags & NVM_FLAG_HAS_MAIN) ||
        m->header.entry_point >= m->function_count)
        STOP(NVM_PORTABLE_READ_INVALID, "I require a valid header and explicit entry.");
    for (uint32_t s = 0; s < m->section_count; s++) {
        const NvmSectionEntry *section = &m->sections[s];
        if (section->type < NVM_SECTION_CODE || section->type > NVM_SECTION_MODULE_REFS ||
            section->size > UINT32_MAX - section->offset)
            STOP(NVM_PORTABLE_READ_INVALID, "I require a valid retained section directory.");
        for (uint32_t earlier = 0; earlier < s; earlier++)
            if (m->sections[earlier].type == section->type)
                STOP(NVM_PORTABLE_READ_INVALID, "I refuse duplicate retained section entries.");
    }
    for (uint32_t s = 0; s < m->string_count; s++) {
        if (!charge(&bytes, m->string_lengths[s], 1) || !charge(&bytes, 1, 1))
            STOP(NVM_PORTABLE_READ_LIMIT, "I exceed my represented string byte limit.");
        if (!m->strings[s] || m->strings[s][m->string_lengths[s]] != '\0')
            STOP(NVM_PORTABLE_READ_INVALID, "I require length-delimited terminated string storage.");
    }
    if (!nvm_metadata_valid(m) ||
        (m->source_file_idx && m->source_file_idx >= m->string_count))
        STOP(NVM_PORTABLE_READ_INVALID, "I require valid advisory metadata indices.");
    for (uint32_t d = 0; d < m->debug_count; d++)
        if (m->debug_entries[d].bytecode_offset > m->code_size)
            STOP(NVM_PORTABLE_READ_INVALID, "I require in-range debug offsets.");

    for (uint32_t i = 0; i < m->import_count; i++) {
        const NvmImportEntry *imp = &m->imports[i];
        if (imp->module_name_idx >= m->string_count ||
            imp->function_name_idx >= m->string_count ||
            imp->kind > NVM_IMPORT_SERVICE || imp->return_type >= TAG_COUNT ||
            imp->param_count > NANO_MAX_FFI_ARGS ||
            (imp->param_count && !m->import_param_types[i]))
            return result(NVM_PORTABLE_READ_INVALID, NONE, NONE, i,
                          "I require a complete import signature and name indices.");
        if (!charge(&bytes, imp->param_count, 1))
            STOP(NVM_PORTABLE_READ_LIMIT, "I exceed my import signature byte limit.");
        for (uint16_t p = 0; p < imp->param_count; p++)
            if (m->import_param_types[i][p] >= TAG_COUNT)
                return result(NVM_PORTABLE_READ_INVALID, NONE, NONE, i,
                              "I require valid import parameter tags.");
        if (memchr(m->strings[imp->module_name_idx], 0, m->string_lengths[imp->module_name_idx]) ||
            memchr(m->strings[imp->function_name_idx], 0, m->string_lengths[imp->function_name_idx]))
            return result(NVM_PORTABLE_READ_INVALID, NONE, NONE, i,
                          "I refuse embedded NUL in import identifiers.");
        if (imp->kind != NVM_IMPORT_FFI || m->string_lengths[imp->module_name_idx] ||
            !read_name(m, imp->function_name_idx) || imp->param_count != 1 ||
            imp->return_type != TAG_STRING || m->import_param_types[i][0] != TAG_STRING)
            return result(NVM_PORTABLE_READ_UNSUPPORTED, NONE, NONE, i,
                          "I require an exact empty-namespace STRING read-text declaration.");
    }

    uint32_t instructions = 0;
    size_t allocation = sizeof(NvmPortableReadPlan);
    for (uint32_t f = 0; f < m->function_count; f++) {
        const NvmFunctionEntry *fn = &m->functions[f];
        if (!fn->code_length || fn->name_idx >= m->string_count ||
            fn->local_count < fn->arity || fn->result_tag >= TAG_COUNT ||
            ((fn->result_count == 0) != (fn->result_tag == TAG_VOID)) ||
            (fn->arity && (!m->function_param_types || !m->function_param_types[f])))
            return result(NVM_PORTABLE_READ_INVALID, f, NONE, NONE,
                          "I require complete nonempty function declarations.");
        if (fn->upvalue_count || fn->result_count > 1 ||
            (fn->result_count && !scalar_string(fn->result_tag)))
            return result(NVM_PORTABLE_READ_UNSUPPORTED, f, NONE, NONE,
                          "I require scalar or STRING functions without captures.");
        if (!charge(&bytes, fn->arity, 1))
            STOP(NVM_PORTABLE_READ_LIMIT, "I exceed my function signature byte limit.");
        for (uint16_t p = 0; p < fn->arity; p++) {
            uint8_t tag = m->function_param_types[f][p];
            if (tag >= TAG_COUNT)
                return result(NVM_PORTABLE_READ_INVALID, f, NONE, NONE,
                              "I require valid function parameter tags.");
            if (!scalar_string(tag))
                return result(NVM_PORTABLE_READ_UNSUPPORTED, f, NONE, NONE,
                              "I require explicit scalar or STRING parameter facts.");
        }
        uint32_t count = 0, pc = 0;
        while (pc < fn->code_length) {
            DecodedInstruction ins;
            uint32_t size = isa_decode(m->code + fn->code_offset + pc,
                                       fn->code_length - pc, &ins);
            if (!size || size > fn->code_length - pc)
                return result(NVM_PORTABLE_READ_INVALID, f, fn->code_offset + pc, NONE,
                              "I require complete decoded instructions.");
            if (unsupported_opcode(ins.opcode))
                return result(NVM_PORTABLE_READ_UNSUPPORTED, f, fn->code_offset + pc, NONE,
                              "I keep linked, indirect and affine operations outside this query.");
            if (instructions == NVM_PORTABLE_READ_MAX_INSTRUCTIONS)
                STOP(NVM_PORTABLE_READ_LIMIT, "I exceed my decoded instruction limit.");
            instructions++; count++; pc += size;
        }
        /* Decoder realloc requests form 16,32,...,capacity. Twice capacity
         * bounds their sum and also old+new during a moving realloc. Stack
         * and advisory type arrays are charged together conservatively. */
        size_t capacity = 16;
        while (capacity < count) capacity *= 2;
        if (!charge(&allocation, capacity, 2 * sizeof(VmDecodedInstruction)) ||
            !charge(&allocation, (size_t)fn->code_length + 1, 1 + sizeof(uint32_t)) ||
            !charge(&allocation, (size_t)count + 1,
                    3 * sizeof(uint32_t) + 256 + sizeof(uint16_t) +
                    2 * sizeof(bool) + sizeof(uint32_t)))
            STOP(NVM_PORTABLE_READ_LIMIT, "I exceed my conservative query allocation budget.");
    }
    if (m->functions[m->header.entry_point].arity)
        STOP(NVM_PORTABLE_READ_UNSUPPORTED, "I require a zero-argument explicit entry.");
    /* Metadata-free preflight excludes composed/owned branches. All functions
     * still pass common operand and stack validation, including unused ones.
     * No callback into this private query exists in the public verifier. */
    NvmVerifyResult verified = nvm_verify(m);
    if (!verified.ok)
        STOP(NVM_PORTABLE_READ_INVALID, "I could not complete common structural verification.");
    if (!m->import_count)
        STOP(NVM_PORTABLE_READ_NOT_SELECTED, "I found no read-text import declaration.");
    NvmPortableReadPlan *plan = calloc(1, sizeof(*plan));
    if (!plan)
        STOP(NVM_PORTABLE_READ_MEMORY, "I could not allocate my owned declaration report.");
    plan->counts = (NvmPortableReadCounts){m->header.entry_point, m->function_count,
        m->import_count, instructions, bytes, allocation};
    for (uint32_t i = 0; i < m->import_count; i++) {
        const NvmImportEntry *imp = &m->imports[i];
        NvmPortableReadImport *row = &plan->imports[i];
        row->import_index = i;
        row->namespace_string_index = imp->module_name_idx;
        row->symbol_string_index = imp->function_name_idx;
        row->operation = NVM_PORTABLE_HOST_READ_TEXT; row->revision = 1;
        row->argument_ownership = NVM_PORTABLE_HOST_BORROW_ROOTED_ARGUMENT;
        row->result_ownership = NVM_PORTABLE_HOST_COPY_MANAGED_RESULT;
        row->parameter_count = 1; row->parameter_tag = TAG_STRING;
        row->result_count = 1; row->result_tag = TAG_STRING;
        row->import_kind = NVM_IMPORT_FFI;
        row->symbol_length = m->string_lengths[imp->function_name_idx];
        memcpy(row->symbol_bytes, m->strings[imp->function_name_idx], row->symbol_length);
    }
    *out = plan;
    STOP(NVM_PORTABLE_READ_PREPARED, "I copied checked read-text declarations without host authority.");
}

void nvm_portable_read_plan_free(NvmPortableReadPlan *plan) { free(plan); }
bool nvm_portable_read_plan_counts(const NvmPortableReadPlan *plan,
                                   NvmPortableReadCounts *out) {
    if (!plan || !out) return false;
    *out = plan->counts; return true;
}
bool nvm_portable_read_plan_import(const NvmPortableReadPlan *plan, uint32_t index,
                                   NvmPortableReadImport *out) {
    if (!plan || !out || index >= plan->counts.import_count) return false;
    *out = plan->imports[index]; return true;
}
#undef STOP
#undef NONE
