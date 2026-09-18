/*
 * NanoISA Text Assembler
 *
 * Two-pass assembler:
 *   Pass 1: Collect labels and their byte offsets
 *   Pass 2: Encode instructions, resolve labels to relative jumps
 */

#include "assembler.h"
#include "local_bindings.h"
#include "isa.h"
#include "verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>

/* ========================================================================
 * Label Table
 * ======================================================================== */


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


typedef struct {
    char label[128];
    uint32_t code_offset; /* Where in the code buffer the i32 offset lives */
    uint32_t instr_start; /* Start of the instruction (for relative offset calc) */
    uint32_t function;    /* Which function */
} Patch;

/* ========================================================================
 * Assembler State
 * ======================================================================== */

typedef struct { uint32_t offset, length; } FlowMarker;

typedef struct LocalMarker {
    uint16_t slot;
    uint32_t begin,end,length;
    char *name;
    bool closed;
    struct LocalMarker *next;
} LocalMarker;

typedef struct {
    NvmModule *mod;
    LocalMarker *local_markers, *local_tail;

    Label *labels;
    uint32_t label_count, label_capacity;

    Symbol *symbols;
    uint32_t symbol_count, symbol_capacity;

    Patch *patches;
    uint32_t patch_count, patch_capacity;

    /* Current function being assembled */
    bool in_function;
    uint32_t current_function;
    uint32_t function_code_start; /* Byte offset where current function's code begins */

    /* Temporary code buffer for current function */
    uint8_t *fn_code;
    uint32_t fn_code_size;
    uint32_t fn_code_capacity;

    /* Structured producer markers resolve to the ordinary passive payload. */
    bool passive_structured, par_active, par_node_active;
    uint32_t passive_capacity, par_block_offset, par_node_offset;
    uint32_t passive_blocks, par_nodes;
    FlowMarker *flow_nodes;
    uint32_t flow_count;

    /* Line tracking */
    uint32_t line;
} AsmState;

static void asm_state_init(AsmState *state) {
    memset(state, 0, sizeof(*state));
    state->fn_code_capacity = 4096;
    state->fn_code = malloc(state->fn_code_capacity);
}

static void asm_state_cleanup(AsmState *state) {
    while (state->local_markers) {
        LocalMarker *next=state->local_markers->next;
        free(state->local_markers->name);free(state->local_markers);state->local_markers=next;
    }

    free(state->fn_code);
    free(state->labels);
    free(state->symbols);
    free(state->patches);
    free(state->flow_nodes);
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

/* Lookup helpers return signed indices; reject overflow before allocation. */
static void *reserve_table(void *table, uint32_t *capacity, uint32_t count,
                           size_t element_size, AsmResult *result) {
    if (count < *capacity) return table;
    if (count >= INT_MAX) {
        result->error = ASM_ERR_BAD_OPERAND;
        snprintf(result->message, sizeof(result->message), "I reached my assembler table index limit");
        return NULL;
    }
    uint32_t next = *capacity ? (*capacity > INT_MAX / 2 ? INT_MAX : *capacity * 2) : 64;
    if (next > SIZE_MAX / element_size) {
        result->error = ASM_ERR_BAD_OPERAND;
        snprintf(result->message, sizeof(result->message), "I cannot represent this assembler table size");
        return NULL;
    }
    void *grown = realloc(table, (size_t)next * element_size);
    if (!grown) {
        result->error = ASM_ERR_MEMORY;
        snprintf(result->message, sizeof(result->message), "I cannot grow this assembler table");
        return NULL;
    }
    *capacity = next;
    return grown;
}

static bool add_symbol(AsmState *state, SymbolKind kind, const char *name, uint32_t value, AsmResult *result) {
    if (find_symbol(state, kind, name) >= 0) { result->error = ASM_ERR_DUPLICATE_SYMBOL; return false; }
    Symbol *grown = reserve_table(state->symbols, &state->symbol_capacity,
                                 state->symbol_count, sizeof(Symbol), result);
    if (!grown) return false;
    state->symbols = grown;
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
/* Comment markers only terminate unquoted assembly text. */
static void strip_trailing_comment(char *line) {
    bool quoted = false;
    for (char *p = line; *p; p++) {
        if (quoted && *p == '\\' && p[1]) { p++; continue; }
        if (*p == '"') { quoted = !quoted; continue; }
        if (!quoted && (*p == ';' || *p == '#')) { *p = '\0'; return; }
    }
}

static bool parse_quoted_string(const char **p, char *out, size_t out_size, uint32_t *out_len) {
    skip_whitespace(p);
    if (**p != '"') return false;
    (*p)++;

    uint32_t i = 0;
    while (**p != '"' && **p != '\0') {
        if (i + 1 >= out_size) return false;
        if (**p == '\\') {
            (*p)++;
            /* A backslash immediately before the terminator is an unterminated
             * escape. Without this the default arm below consumes the NUL and
             * the loop's (*p)++ steps past the end of the buffer, so the next
             * iteration reads out of bounds. */
            if (**p == '\0') return false;
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

static bool add_label(AsmState *state, const char *name, uint32_t offset, AsmResult *result) {
    int existing = find_label(state, name, state->current_function);
    if (existing >= 0 && state->labels[existing].defined) { result->error = ASM_ERR_DUPLICATE_LABEL; return false; }

    if (existing >= 0) {
        state->labels[existing].offset = offset;
        state->labels[existing].defined = true;
        return true;
    }

    Label *grown = reserve_table(state->labels, &state->label_capacity,
                                state->label_count, sizeof(Label), result);
    if (!grown) return false;
    state->labels = grown;
    Label *l = &state->labels[state->label_count++];
    snprintf(l->name, sizeof(l->name), "%s", name);
    l->offset = offset;
    l->function = state->current_function;
    l->defined = true;
    return true;
}

static bool add_patch(AsmState *state, const char *label, uint32_t code_offset, uint32_t instr_start, AsmResult *result) {
    Patch *grown = reserve_table(state->patches, &state->patch_capacity,
                                state->patch_count, sizeof(Patch), result);
    if (!grown) return false;
    state->patches = grown;
    Patch *p = &state->patches[state->patch_count++];
    snprintf(p->label, sizeof(p->label), "%s", label);
    p->code_offset = code_offset;
    p->instr_start = instr_start;
    p->function = state->current_function;
    return true;
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
                /* assemble_instruction sets the final operand offset. */
                if (!add_patch(state, label, 0, instr_start, result)) return 0;
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

static bool par_error(AsmResult *result, const char *message) {
    result->error = ASM_ERR_SYNTAX;
    snprintf(result->message, sizeof(result->message), "%s", message);
    return false;
}

static void passive_patch(AsmState *state, uint32_t offset, uint32_t word) {
    for (unsigned i = 0; i < 4; ++i)
        state->mod->passive_data[offset + i] = (uint8_t)(word >> (8 * i));
}

static bool passive_word(AsmState *state, uint32_t word, AsmResult *result) {
    uint32_t size = state->mod->passive_size;
    if (size > UINT32_MAX - 4) {
        result->error = ASM_ERR_MEMORY;
        snprintf(result->message, sizeof(result->message), "I cannot grow the passive record.");
        return false;
    }
    uint32_t needed = size + 4;
    if (needed > state->passive_capacity) {
        uint32_t capacity = state->passive_capacity ? state->passive_capacity : 128;
        while (capacity < needed)
            capacity = capacity > UINT32_MAX / 2 ? UINT32_MAX : capacity * 2;
        uint8_t *data = realloc(state->mod->passive_data, capacity);
        if (!data) {
            result->error = ASM_ERR_MEMORY;
            snprintf(result->message, sizeof(result->message), "I cannot allocate the passive record.");
            return false;
        }
        state->mod->passive_data = data;
        state->passive_capacity = capacity;
    }
    passive_patch(state, size, word);
    state->mod->passive_size = needed;
    return true;
}

static bool par_directive(AsmState *state, const char *directive,
                          const char *p, AsmResult *result) {
    if (state->flow_nodes)
        return par_error(result, "I cannot mix par markers into a flow block.");
    if (!state->in_function)
        return par_error(result, "I require passive producer markers inside a function.");
    if (state->mod->code_size > UINT32_MAX - state->fn_code_size)
        return par_error(result, "I cannot represent the passive instruction offset.");
    uint32_t offset = state->mod->code_size + state->fn_code_size;
    if (strcmp(directive, "par_begin") == 0) {
        if (!require_line_end(p, result)) return false;
        if (state->par_active)
            return par_error(result, "I cannot nest passive producer blocks.");
        if (!state->passive_structured) {
            if (state->mod->passive_size)
                return par_error(result, "I cannot mix raw passive chunks and producer markers.");
            if (!passive_word(state, 2, result) || !passive_word(state, 0, result)) return false;
            state->passive_structured = true;
        }
        state->par_block_offset = state->mod->passive_size;
        uint32_t fields[] = {1, state->current_function, offset, 0, 0};
        for (unsigned i = 0; i < 5; ++i)
            if (!passive_word(state, fields[i], result)) return false;
        state->par_active = true;
        state->par_node_active = false;
        state->par_nodes = 0;
        return true;
    }
    if (!state->par_active)
        return par_error(result, "I require .par_begin before a passive node or end marker.");
    if (strcmp(directive, "par_end") == 0) {
        if (!require_line_end(p, result)) return false;
        if (!state->par_node_active)
            return par_error(result, "I require at least one node in a passive producer block.");
        passive_patch(state, state->par_node_offset + 4, offset);
        passive_patch(state, state->par_block_offset + 12, offset);
        passive_patch(state, state->par_block_offset + 16, state->par_nodes);
        passive_patch(state, 4, ++state->passive_blocks);
        state->par_active = state->par_node_active = false;
        return true;
    }
    uint32_t local;
    if (!parse_uint32(&p, &local) || local > UINT16_MAX)
        return par_error(result, "I require a local index after .par_node.");
    if (state->par_node_active) passive_patch(state, state->par_node_offset + 4, offset);
    state->par_node_offset = state->mod->passive_size;
    uint32_t fields[] = {offset, 0, local, 0, 0, 0, 0};
    for (unsigned i = 0; i < 7; ++i)
        if (!passive_word(state, fields[i], result)) return false;
    uint32_t count = 0;
    for (;;) {
        skip_whitespace(&p);
        if (!*p) break;
        uint32_t input;
        if (!parse_uint32(&p, &input) || input > UINT16_MAX)
            return par_error(result, "I require parameter indices after the passive result local.");
        if (!passive_word(state, input, result)) return false;
        ++count;
    }
    passive_patch(state, state->par_node_offset + 16, count);
    ++state->par_nodes;
    state->par_node_active = true;
    return true;
}

static bool flow_directive(AsmState *state, const char *directive,
                           const char *p, AsmResult *result) {
    if (!state->in_function)
        return par_error(result, "I require flow producer markers inside a function.");
    if (state->mod->code_size > UINT32_MAX - state->fn_code_size)
        return par_error(result, "I cannot represent the flow instruction offset.");
    uint32_t offset = state->mod->code_size + state->fn_code_size;
    if (strcmp(directive, "flow_begin") == 0) {
        uint32_t count;
        if (!parse_uint32(&p, &count) || !count ||
            count > state->mod->functions[state->current_function].local_count)
            return par_error(result, "I require a positive flow node count within the function locals.");
        if (!require_line_end(p, result)) return false;
        if (!par_directive(state, "par_begin", "", result)) return false;
        state->flow_nodes = calloc(count, sizeof(FlowMarker));
        if (!state->flow_nodes) {
            result->error = ASM_ERR_MEMORY;
            snprintf(result->message, sizeof(result->message), "I cannot allocate the flow marker index.");
            return false;
        }
        state->flow_count = count;
        passive_patch(state, state->par_block_offset, 2);
        return true;
    }
    if (!state->flow_nodes)
        return par_error(result, "I require .flow_begin before a flow node or end marker.");
    if (strcmp(directive, "flow_end") == 0) {
        if (!require_line_end(p, result)) return false;
        if (state->par_nodes != state->flow_count)
            return par_error(result, "I require every source node exactly once before .flow_end.");
        passive_patch(state, state->par_node_offset + 4, offset);
        uint32_t start = state->par_block_offset + 20;
        uint32_t size = state->mod->passive_size - start;
        uint8_t *ordered = malloc(size);
        if (!ordered) {
            result->error = ASM_ERR_MEMORY;
            snprintf(result->message, sizeof(result->message), "I cannot order the flow records.");
            return false;
        }
        uint32_t cursor = 0;
        for (uint32_t i = 0; i < state->flow_count; ++i) {
            FlowMarker node = state->flow_nodes[i];
            memcpy(ordered + cursor, state->mod->passive_data + node.offset, node.length);
            cursor += node.length;
        }
        memcpy(state->mod->passive_data + start, ordered, size);
        free(ordered);
        free(state->flow_nodes);
        state->flow_nodes = NULL;
        passive_patch(state, state->par_block_offset + 12, offset);
        passive_patch(state, state->par_block_offset + 16, state->flow_count);
        passive_patch(state, 4, ++state->passive_blocks);
        state->par_active = state->par_node_active = false;
        state->flow_count = 0;
        return true;
    }
    uint32_t id, local, dependencies;
    if (!parse_uint32(&p, &id) || id >= state->flow_count ||
        !parse_uint32(&p, &local) || local > UINT16_MAX ||
        !parse_uint32(&p, &dependencies) || dependencies > state->flow_count)
        return par_error(result, "I require a source ID, result local and dependency count after .flow_node.");
    if (state->flow_nodes[id].length)
        return par_error(result, "I require distinct flow source IDs.");
    if (state->par_node_active) passive_patch(state, state->par_node_offset + 4, offset);
    uint32_t start = state->mod->passive_size;
    uint32_t fields[] = {offset, 0, local, dependencies, 0, 0, 0};
    for (unsigned i = 0; i < 7; ++i)
        if (!passive_word(state, fields[i], result)) return false;
    uint32_t previous = 0;
    for (uint32_t i = 0; i < dependencies; ++i) {
        uint32_t dependency;
        if (!parse_uint32(&p, &dependency) || dependency >= state->flow_count ||
            (i && dependency <= previous))
            return par_error(result, "I require sorted distinct source IDs for flow dependencies.");
        if (!passive_word(state, dependency, result)) return false;
        previous = dependency;
    }
    uint32_t reads;
    if (!parse_uint32(&p, &reads) || reads > state->mod->functions[state->current_function].arity)
        return par_error(result, "I require a parameter-read count after flow dependencies.");
    passive_patch(state, start + 16, reads);
    previous = 0;
    for (uint32_t i = 0; i < reads; ++i) {
        uint32_t input;
        if (!parse_uint32(&p, &input) || input >= state->mod->functions[state->current_function].arity ||
            (i && input <= previous))
            return par_error(result, "I require sorted distinct parameter indices for flow reads.");
        if (!passive_word(state, input, result)) return false;
        previous = input;
    }
    if (!require_line_end(p, result)) return false;
    state->flow_nodes[id].offset = start;
    state->flow_nodes[id].length = state->mod->passive_size - start;
    state->par_node_offset = start;
    state->par_node_active = true;
    ++state->par_nodes;
    return true;
}

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

        if (strcmp(directive,"local_begin")==0 || strcmp(directive,"local_end")==0) {
            uint16_t slot;
            if(!state->in_function || !parse_uint16(&p,&slot) ||
               slot>=state->mod->functions[state->current_function].local_count)
                return par_error(result,"I require a declared local slot inside a function.");
            LocalMarker *active=NULL;
            for(LocalMarker *m=state->local_markers;m;m=m->next)
                if(m->slot==slot && !m->closed)active=m;
            if(strcmp(directive,"local_end")==0) {
                if(!active || !require_line_end(p,result))
                    return par_error(result,"I require an open local-name marker before its end.");
                active->end=state->fn_code_size;active->closed=true;return true;
            }
            if(active)return par_error(result,"I require disjoint local-name markers for one slot.");
            size_t available=strlen(p)+1;uint32_t length=0;
            char *name=malloc(available);
            if(!name)return par_error(result,"I cannot allocate a local-name marker.");
            if(!parse_quoted_string(&p,name,available,&length) || !length || !require_line_end(p,result)) {
                free(name);return par_error(result,"I require one nonempty quoted local name.");
            }
            LocalMarker *m=calloc(1,sizeof *m);
            if(!m){free(name);return par_error(result,"I cannot allocate a local-name marker.");}
            m->slot=slot;m->begin=state->fn_code_size;m->name=name;m->length=length;
            if(state->local_tail)state->local_tail->next=m;else state->local_markers=m;
            state->local_tail=m;return true;
        }

        if (strcmp(directive, "metadata") == 0) {
            uint32_t key, value;
            if (state->in_function || !parse_uint32(&p, &key) ||
                !parse_uint32(&p, &value) || key >= state->mod->string_count ||
                value >= state->mod->string_count || !require_line_end(p, result)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof result->message,
                         "I require two declared string indices outside functions for metadata");
                return false;
            }
            if (!nvm_add_metadata(state->mod, key, value)) {
                result->error = ASM_ERR_MEMORY;
                snprintf(result->message, sizeof result->message,
                         "I cannot retain this advisory metadata entry");
                return false;
            }
            return true;
        }

        /* I append lossless metadata chunks; normal verification checks the graph. */
        if (strcmp(directive, "par_begin") == 0 || strcmp(directive, "par_node") == 0 ||
            strcmp(directive, "par_end") == 0)
            return par_directive(state, directive, p, result);
        if (strcmp(directive, "flow_begin") == 0 || strcmp(directive, "flow_node") == 0 ||
            strcmp(directive, "flow_end") == 0)
            return flow_directive(state, directive, p, result);

        if (strcmp(directive, "passive") == 0 || strcmp(directive, "layouts") == 0 ||
            strcmp(directive, "ownership") == 0) {
            bool layouts = strcmp(directive, "layouts") == 0;
            bool ownership = strcmp(directive, "ownership") == 0;
            uint8_t **payload = ownership ? &state->mod->ownership_data :
                layouts ? &state->mod->layout_data : &state->mod->passive_data;
            uint32_t *payload_size = ownership ? &state->mod->ownership_size :
                layouts ? &state->mod->layout_size : &state->mod->passive_size;
            if (!layouts && !ownership && state->passive_structured)
                return par_error(result, "I cannot mix raw passive chunks and producer markers.");
            char hex[4096];
            uint32_t length;
            if (state->in_function || !parse_quoted_string(&p, hex, sizeof(hex), &length) ||
                !length || length % 2 || !require_line_end(p, result)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "I expect a nonempty quoted hexadecimal metadata chunk outside functions");
                return false;
            }
            for (uint32_t i = 0; i < length; ++i) {
                if (!isxdigit((unsigned char)hex[i])) {
                    result->error = ASM_ERR_SYNTAX;
                    snprintf(result->message, sizeof(result->message), "I require hexadecimal metadata bytes");
                    return false;
                }
            }
            uint32_t bytes = length / 2;
            if (bytes > UINT32_MAX - *payload_size) {
                result->error = ASM_ERR_MEMORY;
                return false;
            }
            uint8_t *data = realloc(*payload, *payload_size + bytes);
            if (!data) { result->error = ASM_ERR_MEMORY; return false; }
            for (uint32_t i = 0; i < bytes; ++i) {
                unsigned char a = (unsigned char)tolower((unsigned char)hex[i * 2]);
                unsigned char b = (unsigned char)tolower((unsigned char)hex[i * 2 + 1]);
                unsigned high = a <= '9' ? a - '0' : a - 'a' + 10;
                unsigned low = b <= '9' ? b - '0' : b - 'a' + 10;
                data[*payload_size + i] = (uint8_t)(high * 16 + low);
            }
            *payload = data;
            *payload_size += bytes;
            return true;
        }

        if (strcmp(directive, "string") == 0) {
            char name[128];
            uint32_t len;
            const char *before_name = p;
            bool named = parse_identifier(&p, name, sizeof(name));
            if (!named) p = before_name;
            size_t source_length = strlen(p);
            if (source_length >= UINT32_MAX) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message),
                         "I require a string literal within the u32 length limit");
                return false;
            }
            char *buf = malloc(source_length + 1);
            if (!buf) {
                result->error = ASM_ERR_MEMORY;
                snprintf(result->message, sizeof(result->message),
                         "I cannot allocate this string literal");
                return false;
            }
            if (!parse_quoted_string(&p, buf, source_length + 1, &len)) {
                free(buf);
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected quoted string after .string");
                return false;
            }
            uint32_t index = nvm_add_string(state->mod, buf, len);
            free(buf);
            if (index == UINT32_MAX) {
                result->error = ASM_ERR_MEMORY;
                snprintf(result->message, sizeof(result->message),
                         "I cannot retain this string literal");
                return false;
            }
            if (named && !add_symbol(state, SYMBOL_CONSTANT, name, index, result)) {
                if (result->error != ASM_ERR_DUPLICATE_SYMBOL) return false;
                snprintf(result->message, sizeof(result->message),
                         "Duplicate constant symbol: %.200s", name);
                return false;
            }
            return require_line_end(p, result);
        }

        /* .import "module" "symbol" <return-tag> [param-tags...]
         *
         * The import table had no textual form, so a module with an import
         * could be disassembled but not reassembled: CALL_EXTERN referred to
         * a table the text never declared. Names are string literals rather
         * than pool indices because nvm_add_string deduplicates, so a name
         * the pool already holds resolves to the same index it had -- which is
         * what keeps a disassemble/reassemble cycle byte-identical. */
        if (strcmp(directive, "import") == 0) {
            char module_name[256], symbol_name[256];
            uint32_t mlen = 0, slen = 0;
            uint8_t return_tag;
            if (!parse_quoted_string(&p, module_name, sizeof(module_name), &mlen) ||
                !parse_quoted_string(&p, symbol_name, sizeof(symbol_name), &slen) ||
                !parse_result_tag(&p, &return_tag)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected: .import \"module\" \"symbol\" return-tag [param-tags...]");
                return false;
            }
            uint8_t params[NANO_MAX_FFI_ARGS];
            uint16_t param_count = 0;
            for (;;) {
                skip_whitespace(&p);
                if (at_line_end(p)) break;
                if (param_count >= NANO_MAX_FFI_ARGS) {
                    result->error = ASM_ERR_SYNTAX;
                    snprintf(result->message, sizeof(result->message),
                             ".import takes at most %d parameter tags",
                             NANO_MAX_FFI_ARGS);
                    return false;
                }
                if (!parse_result_tag(&p, &params[param_count])) {
                    result->error = ASM_ERR_SYNTAX;
                    snprintf(result->message, sizeof(result->message),
                             "Expected a type tag in .import parameter list");
                    return false;
                }
                param_count++;
            }
            uint32_t midx = nvm_add_string(state->mod, module_name, mlen);
            uint32_t sidx = nvm_add_string(state->mod, symbol_name, slen);
            nvm_add_import(state->mod, midx, sidx, param_count, return_tag,
                           param_count ? params : NULL);
            return require_line_end(p, result);
        }

        if (strcmp(directive, "callback") == 0) {
            NvmCallbackContract c = {0};
            uint32_t parameter, abi, length;
            char adapter[4096], execution[32];
            if (!parse_uint32(&p, &c.import_idx) || !parse_uint32(&p, &parameter) ||
                parameter > UINT16_MAX ||
                !parse_quoted_string(&p, adapter, sizeof(adapter), &length) ||
                !parse_uint32(&p, &abi) || abi != NVM_CALLBACK_ABI_RETAINED_V1 ||
                !parse_identifier(&p, execution, sizeof(execution)) ||
                (strcmp(execution, "owner") && strcmp(execution, "worker")) ||
                !parse_result_tag(&p, &c.return_tag)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "I expect .callback import parameter \"adapter\" 1 owner|worker result [parameters]");
                return false;
            }
            c.parameter_idx = (uint16_t)parameter;
            c.abi_version = (uint8_t)abi;
            c.execution = strcmp(execution, "worker") == 0 ? NVM_FOREIGN_WORKER_THREAD : NVM_FOREIGN_OWNER_THREAD;
            while (!at_line_end(p)) {
                if (c.param_count >= NANO_MAX_FFI_ARGS || !parse_result_tag(&p, &c.param_tags[c.param_count])) {
                    result->error = ASM_ERR_SYNTAX;
                    snprintf(result->message, sizeof(result->message), "I need at most 16 callback parameter tags");
                    return false;
                }
                c.param_count++;
            }
            c.adapter_name_idx = nvm_add_string(state->mod, adapter, length);
            if (c.adapter_name_idx == UINT32_MAX || !nvm_add_callback_contract(state->mod, &c)) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message), "I could not add this callback contract");
                return false;
            }
            return true;
        }

        if (strcmp(directive, "parameters") == 0) {
            uint32_t index = UINT32_MAX;
            const char *start = p;
            if (!parse_uint32(&p, &index)) {
                char name[128];
                p = start;
                if (parse_identifier(&p, name, sizeof(name))) {
                    int found = find_symbol(state, SYMBOL_FUNCTION, name);
                    if (found >= 0) index = state->symbols[found].value;
                }
            }
            if (index >= state->mod->function_count) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message), "I need an existing function name or index for .parameters");
                return false;
            }
            uint16_t arity = state->mod->functions[index].arity;
            uint8_t *tags = arity ? malloc(arity) : NULL;
            bool ok = !arity || tags;
            for (uint16_t i = 0; ok && i < arity; i++) ok = parse_result_tag(&p, &tags[i]);
            ok = ok && at_line_end(p) && nvm_set_function_param_types(state->mod, index, tags, arity);
            free(tags);
            if (!ok) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message), "I need exact-arity tags for .parameters");
            }
            return ok;
        }

        if (strcmp(directive, "import_kind") == 0) {
            uint32_t index;
            char kind[32];
            if (!parse_uint32(&p, &index) || index >= state->mod->import_count ||
                !parse_identifier(&p, kind, sizeof(kind)) || !at_line_end(p) ||
                (strcmp(kind, "ffi") && strcmp(kind, "coprocess") && strcmp(kind, "artifact"))) {
                result->error = ASM_ERR_BAD_OPERAND;
                snprintf(result->message, sizeof(result->message), "I expect .import_kind index ffi|coprocess|artifact");
                return false;
            }
            state->mod->imports[index].kind = !strcmp(kind, "artifact") ? NVM_IMPORT_ARTIFACT :
                !strcmp(kind, "coprocess") ? NVM_IMPORT_COPROCESS : NVM_IMPORT_FFI;
            return true;
        }

        /* .module_ref "name" -- an ordered linked-module dependency.
         * OP_CALL_MODULE's first operand indexes this table. */
        if (strcmp(directive, "module_ref") == 0) {
            char name[256];
            uint32_t len = 0;
            if (!parse_quoted_string(&p, name, sizeof(name), &len)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected: .module_ref \"name\"");
                return false;
            }
            nvm_add_module_ref(state->mod, nvm_add_string(state->mod, name, len));
            return require_line_end(p, result);
        }

        /* .types <structs> <enums> <unions> -- the counts the verifier bounds
         * AGG_* and STRUCT_NEW operands against. Only counts, because that is
         * all a v1 module records. */
        if (strcmp(directive, "types") == 0) {
            uint32_t structs, enums, unions;
            if (!parse_uint32(&p, &structs) || !parse_uint32(&p, &enums) ||
                !parse_uint32(&p, &unions)) {
                result->error = ASM_ERR_SYNTAX;
                snprintf(result->message, sizeof(result->message),
                         "Expected: .types <structs> <enums> <unions>");
                return false;
            }
            state->mod->struct_count = structs;
            state->mod->enum_count = enums;
            state->mod->union_count = unions;
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
            if (!add_symbol(state, kind, name, value, result)) {
                if (result->error != ASM_ERR_DUPLICATE_SYMBOL) return false;
                snprintf(result->message, sizeof(result->message),
                         "Duplicate %s symbol: %.200s", symbol_kind_name(kind), name);
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

            NvmFunctionEntry fn = {0};
            fn.name_idx = nvm_add_string(state->mod, name, (uint32_t)strlen(name));
            if (fn.name_idx == UINT32_MAX) {
                result->error = ASM_ERR_MEMORY;
                snprintf(result->message, sizeof(result->message),
                         "I cannot allocate the function name.");
                return false;
            }
            fn.arity = (uint16_t)arity_val;
            fn.local_count = (uint16_t)locals_val;
            fn.upvalue_count = (uint16_t)upvalues_val;
            fn.result_tag = result_tag;
            fn.result_count = result_count;

            state->current_function = nvm_add_function(state->mod, &fn);
            if (state->current_function == UINT32_MAX) {
                result->error = ASM_ERR_MEMORY;
                snprintf(result->message, sizeof(result->message),
                         "I cannot allocate the function entry.");
                return false;
            }
            int symbol = find_symbol(state, SYMBOL_FUNCTION, name);
            if (symbol < 0 || state->symbols[symbol].value != state->current_function) {
                result->error = ASM_ERR_DUPLICATE_SYMBOL;
                snprintf(result->message, sizeof(result->message),
                         "Duplicate function symbol: %.200s", name);
                return false;
            }
            state->in_function = true;
            state->fn_code_size = 0;
            return require_line_end(p, result);
        }

        if (strcmp(directive, "end") == 0) {
            if (state->par_active)
                return par_error(result, "I require the passive block's end marker before ending the function.");
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

            while(state->local_markers) {
                LocalMarker *m=state->local_markers;
                NvmLocalBinding binding={.function=state->current_function,.slot=m->slot,
                    .begin=m->begin,.end=m->closed?m->end:state->fn_code_size,
                    .name=(const uint8_t *)m->name,.name_size=m->length};
                if(!nvm_add_local_binding(state->mod,&binding))
                    return par_error(result,"I cannot retain this lexical local name.");
                state->local_markers=m->next;free(m->name);free(m);
            }
            state->local_tail=NULL;
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
                if (!add_label(state, ident, state->fn_code_size, result)) {
                    if (result->error != ASM_ERR_DUPLICATE_LABEL) return false;
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
                !add_symbol(state, SYMBOL_FUNCTION, name, function_index++, result)) {
                result->line = line;
                if (result->error == ASM_ERR_DUPLICATE_SYMBOL)
                    snprintf(result->message, sizeof(result->message),
                             "Duplicate function symbol: %.200s", name);
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
        nvm_module_free(state.mod);
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
        strip_trailing_comment(line_buf);

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
