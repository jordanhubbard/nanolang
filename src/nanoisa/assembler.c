/*
 * NanoISA Text Assembler
 *
 * Two-pass assembler:
 *   Pass 1: Collect labels and their byte offsets
 *   Pass 2: Encode instructions, resolve labels to relative jumps
 */

#include "assembler.h"
#include "isa.h"
#include "verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

/* ========================================================================
 * Label Table
 * ======================================================================== */

#define MAX_LABELS 1024
#define MAX_SYMBOLS 2048

typedef enum {
    SYMBOL_FUNCTION,
    SYMBOL_IMPORT,
    SYMBOL_FIELD,
    SYMBOL_TYPE,
    SYMBOL_CONSTANT
} SymbolKind;

typedef struct {
    char name[128];
    uint32_t value;
    SymbolKind kind;
} Symbol;

typedef struct {
    char name[128];
    uint32_t offset;    /* Byte offset in current function's code */
    uint32_t function;  /* Which function this label belongs to */
    bool defined;
} Label;

/* ========================================================================
 * Patch List (forward references to resolve in pass 2)
 * ======================================================================== */

#define MAX_PATCHES 2048

typedef struct {
    char label[128];
    uint32_t code_offset; /* Where in the code buffer the i32 offset lives */
    uint32_t instr_start; /* Start of the instruction (for relative offset calc) */
    uint32_t function;    /* Which function */
} Patch;

/* ========================================================================
 * Assembler State
 * ======================================================================== */

typedef struct {
    NvmModule *mod;

    Label labels[MAX_LABELS];
    uint32_t label_count;

    Symbol symbols[MAX_SYMBOLS];
    uint32_t symbol_count;

    Patch patches[MAX_PATCHES];
    uint32_t patch_count;

    /* Current function being assembled */
    bool in_function;
    uint32_t current_function;
    uint32_t function_code_start; /* Byte offset where current function's code begins */

    /* Temporary code buffer for current function */
    uint8_t *fn_code;
    uint32_t fn_code_size;
    uint32_t fn_code_capacity;

    /* Line tracking */
    uint32_t line;
} AsmState;

static void asm_state_init(AsmState *state) {
    memset(state, 0, sizeof(*state));
    state->fn_code_capacity = 4096;
    state->fn_code = malloc(state->fn_code_capacity);
}

static void asm_state_cleanup(AsmState *state) {
    free(state->fn_code);
}

static void fn_emit(AsmState *state, const uint8_t *data, uint32_t size) {
    while (state->fn_code_size + size > state->fn_code_capacity) {
        state->fn_code_capacity *= 2;
        state->fn_code = realloc(state->fn_code, state->fn_code_capacity);
    }
    memcpy(state->fn_code + state->fn_code_size, data, size);
    state->fn_code_size += size;
}

/* ========================================================================
 * Parsing Helpers
 * ======================================================================== */

static void skip_whitespace(const char **p) {
    while (**p == ' ' || **p == '\t') (*p)++;
}

static bool require_line_end(const char *p, AsmResult *result) {
    skip_whitespace(&p);
    if (*p == '\0') return true;

    result->error = ASM_ERR_SYNTAX;
    snprintf(result->message, sizeof(result->message),
             "Unexpected trailing input: %s", p);
    return false;
}

static bool parse_identifier(const char **p, char *out, size_t out_size) {
    skip_whitespace(p);
    size_t i = 0;
    while ((**p >= 'A' && **p <= 'Z') || (**p >= 'a' && **p <= 'z') ||
           (**p >= '0' && **p <= '9') || **p == '_') {
        if (i + 1 >= out_size) return false;
        out[i++] = *(*p)++;
    }
    out[i] = '\0';
    return i > 0;
}

static bool at_line_end(const char *p) {
    skip_whitespace(&p);
    return *p == '\0';
}

static bool parse_int64(const char **p, int64_t *val) {
    skip_whitespace(p);
    char *end;
    errno = 0;
    long long v = strtoll(*p, &end, 0);
    if (end == *p || errno != 0) return false;
    *val = (int64_t)v;
    *p = end;
    return true;
}

static bool parse_uint32(const char **p, uint32_t *val) {
    int64_t v;
    if (!parse_int64(p, &v)) return false;
    if (v < 0 || v > UINT32_MAX) return false;
    *val = (uint32_t)v;
    return true;
}

static bool parse_uint16(const char **p, uint16_t *val) {
    int64_t v;
    if (!parse_int64(p, &v)) return false;
    if (v < 0 || v > UINT16_MAX) return false;
    *val = (uint16_t)v;
    return true;
}

static bool parse_uint8(const char **p, uint8_t *val) {
    int64_t v;
    if (!parse_int64(p, &v)) return false;
    if (v < 0 || v > 255) return false;
    *val = (uint8_t)v;
    return true;
}

static bool parse_result_tag(const char **p, uint8_t *tag) {
    char name[32];
    if (!parse_identifier(p, name, sizeof(name))) return false;
    for (uint8_t candidate = 0; candidate < TAG_COUNT; candidate++) {
        if (strcmp(name, isa_tag_name(candidate)) == 0) {
            *tag = candidate;
            return true;
        }
    }
    return false;
}

static bool parse_double(const char **p, double *val) {
    skip_whitespace(p);
    char *end;
    errno = 0;
    double v = strtod(*p, &end);
    if (end == *p || errno != 0) return false;
    *val = v;
    *p = end;
    return true;
}

static bool parse_int32(const char **p, int32_t *val) {
    int64_t v;
    if (!parse_int64(p, &v)) return false;
    if (v < INT32_MIN || v > INT32_MAX) return false;
    *val = (int32_t)v;
    return true;
}

static bool parse_symbol_kind(const char **p, SymbolKind *kind) {
    char name[32];
    if (!parse_identifier(p, name, sizeof(name))) return false;
    if (strcmp(name, "function") == 0) *kind = SYMBOL_FUNCTION;
    else if (strcmp(name, "import") == 0) *kind = SYMBOL_IMPORT;
    else if (strcmp(name, "field") == 0) *kind = SYMBOL_FIELD;
    else if (strcmp(name, "type") == 0) *kind = SYMBOL_TYPE;
    else if (strcmp(name, "constant") == 0) *kind = SYMBOL_CONSTANT;
    else return false;
    return true;
}

static const char *symbol_kind_name(SymbolKind kind) {
    static const char *names[] = { "function", "import", "field", "type", "constant" };
    return names[kind];
}

static int find_symbol(const AsmState *state, SymbolKind kind, const char *name) {
    for (uint32_t i = 0; i < state->symbol_count; i++) {
        if (state->symbols[i].kind == kind && strcmp(state->symbols[i].name, name) == 0)
            return (int)i;
    }
    return -1;
}

static bool add_symbol(AsmState *state, SymbolKind kind, const char *name, uint32_t value) {
    if (state->symbol_count >= MAX_SYMBOLS || find_symbol(state, kind, name) >= 0) return false;
    Symbol *symbol = &state->symbols[state->symbol_count++];
    snprintf(symbol->name, sizeof(symbol->name), "%s", name);
    symbol->kind = kind;
    symbol->value = value;
    return true;
}

static bool operand_symbol_kind(uint8_t opcode, int operand_index, SymbolKind *kind) {
    if (operand_index == 0) {
        switch (opcode) {
            case OP_CALL: case OP_TAIL_CALL: case OP_FUNCREF: case OP_CLOSURE_NEW:
                *kind = SYMBOL_FUNCTION; return true;
            case OP_CALL_EXTERN:
                *kind = SYMBOL_IMPORT; return true;
            case OP_PUSH_STR:
                *kind = SYMBOL_CONSTANT; return true;
            case OP_STRUCT_NEW: case OP_STRUCT_LITERAL: case OP_UNION_CONSTRUCT: case OP_ENUM_VAL:
                *kind = SYMBOL_TYPE; return true;
            case OP_STRUCT_GET: case OP_STRUCT_SET: case OP_UNION_FIELD:
            case OP_AGG_GET: case OP_AGG_SET:
                *kind = SYMBOL_FIELD; return true;
            default: break;
        }
    }
    if (opcode == OP_AGG_PACK && operand_index == 1) {
        *kind = SYMBOL_TYPE;
        return true;
    }
    return false;
}
/* Parse one hex digit, returning 0-15 or -1 if not a hex digit. */
static int hex_digit_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Parse a quoted string: "hello world" -> hello world
 * Supports escape sequences: \n, \r, \t, \0, \\, \", and \xHH for
 * arbitrary bytes so binary strings round-trip losslessly. */
static bool parse_quoted_string(const char **p, char *out, size_t out_size, uint32_t *out_len) {
    skip_whitespace(p);
    if (**p != '"') return false;
    (*p)++;

    uint32_t i = 0;
    while (**p != '"' && **p != '\0') {
        if (i + 1 >= out_size) return false;
        if (**p == '\\') {
            (*p)++;
            switch (**p) {
                case 'n':  out[i++] = '\n'; break;
                case 'r':  out[i++] = '\r'; break;
                case 't':  out[i++] = '\t'; break;
                case '\\': out[i++] = '\\'; break;
                case '"':  out[i++] = '"';  break;
                case '0':  out[i++] = '\0'; break;
                case 'x': {
                    int hi = hex_digit_value((*p)[1]);
                    int lo = hi < 0 ? -1 : hex_digit_value((*p)[2]);
                    if (lo < 0) return false; /* \x must be followed by two hex digits */
                    out[i++] = (char)((hi << 4) | lo);
                    (*p) += 2;
                    break;
                }
                default:   out[i++] = **p;  break;
            }
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    if (**p != '"') return false;
    (*p)++;

    out[i] = '\0';
    *out_len = i;
    return true;
}

/* ========================================================================
 * Label Management
 * ======================================================================== */

static int find_label(AsmState *state, const char *name, uint32_t function) {
    for (uint32_t i = 0; i < state->label_count; i++) {
        if (state->labels[i].function == function &&
            strcmp(state->labels[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static bool add_label(AsmState *state, const char *name, uint32_t offset) {
    if (state->label_count >= MAX_LABELS) return false;
    int existing = find_label(state, name, state->current_function);
    if (existing >= 0 && state->labels[existing].defined) return false; /* duplicate */

    if (existing >= 0) {
        state->labels[existing].offset = offset;
        state->labels[existing].defined = true;
        return true;
    }

    Label *l = &state->labels[state->label_count++];
    snprintf(l->name, sizeof(l->name), "%s", name);
    l->offset = offset;
    l->function = state->current_function;
    l->defined = true;
    return true;
}

static void add_patch(AsmState *state, const char *label, uint32_t code_offset, uint32_t instr_start) {
    if (state->patch_count >= MAX_PATCHES) return;
    Patch *p = &state->patches[state->patch_count++];
    snprintf(p->label, sizeof(p->label), "%s", label);
    p->code_offset = code_offset;
    p->instr_start = instr_start;
    p->function = state->current_function;
}

/* ========================================================================
 * Instruction Assembly
 * ======================================================================== */

/* Encode a single operand value based on type, returning bytes written */
static uint32_t encode_operand(uint8_t *buf, OperandType type, uint8_t opcode,
                               int operand_index,
                               const char **line_ptr, AsmState *state,
                               uint32_t instr_start, AsmResult *result) {
    SymbolKind symbol_kind;
    if ((type == OPERAND_U16 || type == OPERAND_U32) &&
        operand_symbol_kind(opcode, operand_index, &symbol_kind)) {
        const char *saved = *line_ptr;
        char name[128];
        skip_whitespace(line_ptr);
        if (((**line_ptr >= 'A' && **line_ptr <= 'Z') ||
             (**line_ptr >= 'a' && **line_ptr <= 'z') || **line_ptr == '_') &&
            parse_identifier(line_ptr, name, sizeof(name))) {
            int found = find_symbol(state, symbol_kind, name);
            if (found < 0) {
                result->error = ASM_ERR_UNDEFINED_SYMBOL;
                snprintf(result->message, sizeof(result->message),
                         "Undefined %s symbol: %s", symbol_kind_name(symbol_kind), name);
                return 0;
            }
            uint32_t value = state->symbols[found].value;
            if (type == OPERAND_U16 && value > UINT16_MAX) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "%s symbol out of u16 range: %s", symbol_kind_name(symbol_kind), name);
                return 0;
            }
            buf[0] = (uint8_t)(value & 0xff);
            buf[1] = (uint8_t)((value >> 8) & 0xff);
            if (type == OPERAND_U32) {
                buf[2] = (uint8_t)((value >> 16) & 0xff);
                buf[3] = (uint8_t)((value >> 24) & 0xff);
                return 4;
            }
            return 2;
        }
        *line_ptr = saved;
    }
    switch (type) {
        case OPERAND_U8: {
            uint8_t v;
            if (!parse_uint8(line_ptr, &v)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "Expected u8 operand");
                return 0;
            }
            buf[0] = v;
            return 1;
        }
        case OPERAND_U16: {
            uint16_t v;
            if (!parse_uint16(line_ptr, &v)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "Expected u16 operand");
                return 0;
            }
            buf[0] = (uint8_t)(v & 0xFF);
            buf[1] = (uint8_t)((v >> 8) & 0xFF);
            return 2;
        }
        case OPERAND_U32: {
            uint32_t v;
            if (!parse_uint32(line_ptr, &v)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "Expected u32 operand");
                return 0;
            }
            buf[0] = (uint8_t)(v & 0xFF);
            buf[1] = (uint8_t)((v >> 8) & 0xFF);
            buf[2] = (uint8_t)((v >> 16) & 0xFF);
            buf[3] = (uint8_t)((v >> 24) & 0xFF);
            return 4;
        }
        case OPERAND_I32: {
            /* Could be a label reference or numeric literal */
            skip_whitespace(line_ptr);
            if ((**line_ptr >= 'A' && **line_ptr <= 'Z') ||
                (**line_ptr >= 'a' && **line_ptr <= 'z') ||
                **line_ptr == '_') {
                /* Label reference - emit placeholder, add patch */
                char label[128];
                if (!parse_identifier(line_ptr, label, sizeof(label))) {
                    result->error = ASM_ERR_BAD_OPERAND;
                    snprintf(result->message, sizeof(result->message),
                             "Expected label or i32 operand");
                    return 0;
                }
                add_patch(state, label, state->fn_code_size + (uint32_t)(buf - (state->fn_code + state->fn_code_size)),
                          instr_start);
                /* Placeholder - will be patched */
                memset(buf, 0, 4);
                return 4;
            }
            int32_t v;
            if (!parse_int32(line_ptr, &v)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "Expected i32 operand");
                return 0;
            }
            uint32_t uv = (uint32_t)v;
            buf[0] = (uint8_t)(uv & 0xFF);
            buf[1] = (uint8_t)((uv >> 8) & 0xFF);
            buf[2] = (uint8_t)((uv >> 16) & 0xFF);
            buf[3] = (uint8_t)((uv >> 24) & 0xFF);
            return 4;
        }
        case OPERAND_I64: {
            int64_t v;
            if (!parse_int64(line_ptr, &v)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "Expected i64 operand");
                return 0;
            }
            uint64_t uv = (uint64_t)v;
            for (int i = 0; i < 8; i++) {
                buf[i] = (uint8_t)(uv & 0xFF);
                uv >>= 8;
            }
            return 8;
        }
        case OPERAND_F64: {
            double v;
            if (!parse_double(line_ptr, &v)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "Expected f64 operand");
                return 0;
            }
            uint64_t bits;
            memcpy(&bits, &v, sizeof(bits));
            for (int i = 0; i < 8; i++) {
                buf[i] = (uint8_t)(bits & 0xFF);
                bits >>= 8;
            }
            return 8;
        }
        case OPERAND_NONE:
            return 0;
    }
    return 0;
}

static bool assemble_instruction(AsmState *state, const char *mnemonic,
                                 const char **rest, AsmResult *result) {
    int opcode = isa_opcode_by_name(mnemonic);
    if (opcode < 0) {
        result->error = ASM_ERR_UNKNOWN_OPCODE;
        snprintf(result->message, sizeof(result->message),
                 "Unknown opcode: %s", mnemonic);
        return false;
    }

    const InstructionInfo *info = isa_get_info((uint8_t)opcode);
    if (!info) {
        result->error = ASM_ERR_UNKNOWN_OPCODE;
        return false;
    }

    uint32_t instr_start = state->fn_code_size;

    /* Emit opcode byte */
    uint8_t op = (uint8_t)opcode;
    fn_emit(state, &op, 1);

    /* Emit operands */
    for (int i = 0; i < info->operand_count; i++) {
        uint8_t operand_buf[8];
        /* For I32 label patches, we need the offset into fn_code where the operand will land */
        uint32_t patch_offset = state->fn_code_size;
        uint32_t patches_before = state->patch_count;
        uint32_t nbytes = encode_operand(operand_buf, info->operands[i], (uint8_t)opcode, i,
                                           rest, state, instr_start, result);
        if (result->error != ASM_OK) return false;

        /* Fix up patch offset: if a patch was added, update its code_offset */
        if (info->operands[i] == OPERAND_I32 && state->patch_count > patches_before) {
            Patch *last = &state->patches[state->patch_count - 1];
            if (last->code_offset != patch_offset) {
                last->code_offset = patch_offset;
            }
        }

        fn_emit(state, operand_buf, nbytes);
    }

    return require_line_end(*rest, result);
}

/* ========================================================================
 * Line Processing
 * ======================================================================== */

static bool process_line(AsmState *state, const char *line, AsmResult *result) {
    const char *p = line;
    skip_whitespace(&p);

    /* Empty line or comment */
    if (*p == '\0' || *p == ';' || *p == '#') return true;

    /* Directive: .string, .symbol, .function, .end, .entry, .flag */
    if (*p == '.') {
        p++;
        char directive[64];
        if (!parse_identifier(&p, directive, sizeof(directive))) {
            result->error = ASM_ERR_SYNTAX;
            snprintf(result->message, sizeof(result->message),
                     "Invalid directive");
            return false;
        }

        if (strcmp(directive, "string") == 0) {
            char buf[4096];
            char name[128];
            uint32_t len;
            const char *before_name = p;
            bool named = parse_identifier(&p, name, sizeof(name));
            if (!named) p = before_name;
            if (!parse_quoted_string(&p, buf, sizeof(buf), &len)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected quoted string after .string");
                return false;
            }
            uint32_t index = nvm_add_string(state->mod, buf, len);
            if (named && !add_symbol(state, SYMBOL_CONSTANT, name, index)) {
                result->error = ASM_ERR_DUPLICATE_SYMBOL;
                snprintf(result->message, sizeof(result->message),
                         "Duplicate constant symbol: %s", name);
                return false;
            }
            return require_line_end(p, result);
        }

        if (strcmp(directive, "symbol") == 0) {
            SymbolKind kind;
            char name[128];
            uint32_t value;
            if (!parse_symbol_kind(&p, &kind) ||
                !parse_identifier(&p, name, sizeof(name)) ||
                !parse_uint32(&p, &value) || !at_line_end(p)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected: .symbol function|import|field|type|constant name index");
                return false;
            }
            if (!add_symbol(state, kind, name, value)) {
                result->error = ASM_ERR_DUPLICATE_SYMBOL;
                snprintf(result->message, sizeof(result->message),
                         "Duplicate %s symbol: %s", symbol_kind_name(kind), name);
                return false;
            }
            return true;
        }

        if (strcmp(directive, "function") == 0) {
            if (state->in_function) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Nested .function not allowed");
                return false;
            }

            /* .function name arity locals upvalues result-tag result-count */
            char name[256];
            if (!parse_identifier(&p, name, sizeof(name))) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected function name after .function");
                return false;
            }

            uint32_t arity_val, locals_val, upvalues_val;
            uint8_t result_tag, result_count;
            if (!parse_uint32(&p, &arity_val) ||
                !parse_uint32(&p, &locals_val) ||
                !parse_uint32(&p, &upvalues_val) ||
                !parse_result_tag(&p, &result_tag) ||
                !parse_uint8(&p, &result_count) ||
                ((result_count == 0) != (result_tag == TAG_VOID))) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected: .function name arity locals upvalues result-tag result-count");
                return false;
            }

            state->in_function = true;
            state->fn_code_size = 0;

            NvmFunctionEntry fn = {0};
            fn.name_idx = nvm_add_string(state->mod, name, (uint32_t)strlen(name));
            fn.arity = (uint16_t)arity_val;
            fn.local_count = (uint16_t)locals_val;
            fn.upvalue_count = (uint16_t)upvalues_val;
            fn.result_tag = result_tag;
            fn.result_count = result_count;

            state->current_function = nvm_add_function(state->mod, &fn);
            int symbol = find_symbol(state, SYMBOL_FUNCTION, name);
            if (symbol < 0 || state->symbols[symbol].value != state->current_function) {
                result->error = ASM_ERR_DUPLICATE_SYMBOL;
                snprintf(result->message, sizeof(result->message),
                         "Duplicate function symbol: %s", name);
                return false;
            }
            return require_line_end(p, result);
        }

        if (strcmp(directive, "end") == 0) {
            if (!state->in_function) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         ".end without matching .function");
                return false;
            }

            /* Resolve patches for this function */
            for (uint32_t i = 0; i < state->patch_count; i++) {
                Patch *patch = &state->patches[i];
                if (patch->function != state->current_function) continue;

                int lbl = find_label(state, patch->label, state->current_function);
                if (lbl < 0 || !state->labels[lbl].defined) {
                    result->error = ASM_ERR_UNDEFINED_LABEL;
                    snprintf(result->message, sizeof(result->message),
                             "Undefined label: %s", patch->label);
                    return false;
                }

                /* Relative offset from instruction start to label */
                int32_t rel = (int32_t)state->labels[lbl].offset - (int32_t)patch->instr_start;
                uint32_t urel = (uint32_t)rel;
                state->fn_code[patch->code_offset + 0] = (uint8_t)(urel & 0xFF);
                state->fn_code[patch->code_offset + 1] = (uint8_t)((urel >> 8) & 0xFF);
                state->fn_code[patch->code_offset + 2] = (uint8_t)((urel >> 16) & 0xFF);
                state->fn_code[patch->code_offset + 3] = (uint8_t)((urel >> 24) & 0xFF);
            }

            /* Flush function code to module */
            uint32_t code_off = nvm_append_code(state->mod, state->fn_code, state->fn_code_size);
            state->mod->functions[state->current_function].code_offset = code_off;
            state->mod->functions[state->current_function].code_length = state->fn_code_size;

            state->in_function = false;

            /* Clear patches for this function */
            uint32_t new_count = 0;
            for (uint32_t i = 0; i < state->patch_count; i++) {
                if (state->patches[i].function != state->current_function) {
                    state->patches[new_count++] = state->patches[i];
                }
            }
            state->patch_count = new_count;

            return require_line_end(p, result);
        }

        if (strcmp(directive, "entry") == 0) {
            uint32_t v;
            const char *before_symbol = p;
            char name[128];
            skip_whitespace(&p);
            if (((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || *p == '_') &&
                parse_identifier(&p, name, sizeof(name))) {
                int found = find_symbol(state, SYMBOL_FUNCTION, name);
                if (found < 0) {
                    result->error = ASM_ERR_UNDEFINED_SYMBOL;
                    snprintf(result->message, sizeof(result->message),
                             "Undefined function symbol: %s", name);
                    return false;
                }
                v = state->symbols[found].value;
            } else {
                p = before_symbol;
                if (!parse_uint32(&p, &v)) {
                    result->error = ASM_ERR_SYNTAX;
                    snprintf(result->message, sizeof(result->message),
                             "Expected function name or index after .entry");
                    return false;
                }
            }
            if (!at_line_end(p)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Unexpected text after .entry operand");
                return false;
            }
            state->mod->header.entry_point = v;
            state->mod->header.flags |= NVM_FLAG_HAS_MAIN;
            return require_line_end(p, result);
        }

        if (strcmp(directive, "flag") == 0) {
            char flag_name[64];
            if (!parse_identifier(&p, flag_name, sizeof(flag_name))) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected flag name after .flag");
                return false;
            }
            if (strcmp(flag_name, "has_main") == 0) {
                state->mod->header.flags |= NVM_FLAG_HAS_MAIN;
            } else if (strcmp(flag_name, "needs_extern") == 0) {
                state->mod->header.flags |= NVM_FLAG_NEEDS_EXTERN;
            } else if (strcmp(flag_name, "debug_info") == 0) {
                state->mod->header.flags |= NVM_FLAG_DEBUG_INFO;
            }
            return require_line_end(p, result);
        }

        result->error = ASM_ERR_SYNTAX;
        snprintf(result->message, sizeof(result->message),
                 "Unknown directive: .%s", directive);
        return false;
    }

    /* Label: identifier followed by ':' */
    {
        const char *saved = p;
        char ident[128];
        if (parse_identifier(&p, ident, sizeof(ident))) {
            skip_whitespace(&p);
            if (*p == ':') {
                p++;
                if (!state->in_function) {
                    result->error = ASM_ERR_NO_FUNCTION;
                    snprintf(result->message, sizeof(result->message),
                             "Label outside .function/.end block");
                    return false;
                }
                if (!add_label(state, ident, state->fn_code_size)) {
                    result->error = ASM_ERR_DUPLICATE_LABEL;
                    snprintf(result->message, sizeof(result->message),
                             "Duplicate label: %s", ident);
                    return false;
                }
                /* Check if there's an instruction on the same line after the label */
                skip_whitespace(&p);
                if (*p != '\0' && *p != ';' && *p != '#') {
                    char mnemonic[64];
                    if (!parse_identifier(&p, mnemonic, sizeof(mnemonic))) {
                        result->error = ASM_ERR_SYNTAX;
                        return false;
                    }
                    return assemble_instruction(state, mnemonic, &p, result);
                }
                return true;
            }
            /* Not a label - reset and try as instruction */
            p = saved;
        } else {
            p = saved;
        }
    }

    /* Instruction */
    if (!state->in_function) {
        result->error = ASM_ERR_NO_FUNCTION;
        snprintf(result->message, sizeof(result->message),
                 "Instruction outside .function/.end block");
        return false;
    }

    char mnemonic[64];
    if (!parse_identifier(&p, mnemonic, sizeof(mnemonic))) {
        result->error = ASM_ERR_SYNTAX;
        snprintf(result->message, sizeof(result->message),
                 "Expected instruction mnemonic");
        return false;
    }

    return assemble_instruction(state, mnemonic, &p, result);
}

/* ========================================================================
 * Public API
 * ======================================================================== */

static bool collect_function_symbols(AsmState *state, const char *source, AsmResult *result) {
    const char *p = source;
    uint32_t line = 0;
    uint32_t function_index = 0;
    while (*p) {
        line++;
        const char *line_start = p;
        while (*p && *p != '\n') p++;
        size_t line_len = (size_t)(p - line_start);
        char *line_buf = malloc(line_len + 1);
        if (!line_buf) {
            result->error = ASM_ERR_MEMORY;
            result->line = line;
            snprintf(result->message, sizeof(result->message), "Out of memory");
            return false;
        }
        memcpy(line_buf, line_start, line_len);
        line_buf[line_len] = '\0';

        const char *cursor = line_buf;
        skip_whitespace(&cursor);
        if (strncmp(cursor, ".function", 9) == 0 &&
            (cursor[9] == ' ' || cursor[9] == '\t')) {
            cursor += 9;
            char name[128];
            if (parse_identifier(&cursor, name, sizeof(name)) &&
                !add_symbol(state, SYMBOL_FUNCTION, name, function_index++)) {
                result->error = ASM_ERR_DUPLICATE_SYMBOL;
                result->line = line;
                snprintf(result->message, sizeof(result->message),
                         "Duplicate function symbol: %s", name);
                free(line_buf);
                return false;
            }
        }
        free(line_buf);
        if (*p == '\n') p++;
    }
    return true;
}

static NvmModule *asm_assemble_impl(const char *source, AsmResult *result,
                                    bool verify) {
    memset(result, 0, sizeof(*result));

    AsmState state;
    asm_state_init(&state);
    state.mod = nvm_module_new();
    if (!state.mod || !state.fn_code) {
        result->error = ASM_ERR_MEMORY;
        snprintf(result->message, sizeof(result->message), "Out of memory");
        asm_state_cleanup(&state);
        return NULL;
    }
    if (!collect_function_symbols(&state, source, result)) {
        nvm_module_free(state.mod);
        asm_state_cleanup(&state);
        return NULL;
    }

    /* Process line by line */
    const char *p = source;
    state.line = 0;

    while (*p) {
        state.line++;

        /* Extract line */
        const char *line_start = p;
        while (*p && *p != '\n') p++;

        size_t line_len = (size_t)(p - line_start);
        char *line_buf = malloc(line_len + 1);
        if (!line_buf) {
            result->error = ASM_ERR_MEMORY;
            result->line = state.line;
            nvm_module_free(state.mod);
            asm_state_cleanup(&state);
            return NULL;
        }
        memcpy(line_buf, line_start, line_len);
        line_buf[line_len] = '\0';

        /* Strip trailing comment */
        char *comment = strchr(line_buf, ';');
        if (comment) *comment = '\0';
        comment = strchr(line_buf, '#');
        if (comment) *comment = '\0';

        /* Strip trailing whitespace */
        size_t len = strlen(line_buf);
        while (len > 0 && (line_buf[len - 1] == ' ' || line_buf[len - 1] == '\t' ||
                           line_buf[len - 1] == '\r')) {
            line_buf[--len] = '\0';
        }

        if (!process_line(&state, line_buf, result)) {
            result->line = state.line;
            free(line_buf);
            nvm_module_free(state.mod);
            asm_state_cleanup(&state);
            return NULL;
        }

        free(line_buf);
        if (*p == '\n') p++;
    }

    if (state.in_function) {
        result->error = ASM_ERR_SYNTAX;
        result->line = state.line;
        snprintf(result->message, sizeof(result->message),
                 "Unterminated .function (missing .end)");
        nvm_module_free(state.mod);
        asm_state_cleanup(&state);
        return NULL;
    }

    NvmModule *mod = state.mod;
    asm_state_cleanup(&state);

    if (verify) {
        NvmVerifyResult verdict = nvm_verify(mod);
        if (!verdict.ok) {
            result->error = ASM_ERR_VERIFY;
            snprintf(result->message, sizeof(result->message),
                     "Assembled module failed verification: %.210s", verdict.error_msg);
            nvm_module_free(mod);
            return NULL;
        }
    }
    return mod;
}

NvmModule *asm_assemble(const char *source, AsmResult *result) {
    return asm_assemble_impl(source, result, true);
}

NvmModule *asm_assemble_unverified(const char *source, AsmResult *result) {
    return asm_assemble_impl(source, result, false);
}

NvmModule *asm_assemble_file(const char *path, AsmResult *result) {
    FILE *f = fopen(path, "r");
    if (!f) {
        result->error = ASM_ERR_IO;
        snprintf(result->message, sizeof(result->message),
                 "Cannot open file: %s", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize < 0 || fsize > 10 * 1024 * 1024) { /* 10MB max */
        result->error = ASM_ERR_IO;
        snprintf(result->message, sizeof(result->message),
                 "File too large or unreadable: %s", path);
        fclose(f);
        return NULL;
    }

    char *source = malloc((size_t)fsize + 1);
    if (!source) {
        result->error = ASM_ERR_MEMORY;
        fclose(f);
        return NULL;
    }

    size_t nread = fread(source, 1, (size_t)fsize, f);
    fclose(f);
    source[nread] = '\0';

    NvmModule *mod = asm_assemble(source, result);
    free(source);
    return mod;
}
