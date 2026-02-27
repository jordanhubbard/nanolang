/*
 * NanoISA - Instruction metadata, encoding, and decoding
 */

#include "isa.h"
#include <string.h>

/* ========================================================================
 * Value Tag Names
 * ======================================================================== */

static const char *tag_names[] = {
    [TAG_VOID]     = "void",
    [TAG_INT]      = "int",
    [TAG_U8]       = "u8",
    [TAG_FLOAT]    = "float",
    [TAG_BOOL]     = "bool",
    [TAG_STRING]   = "string",
    [TAG_BSTRING]  = "bstring",
    [TAG_ARRAY]    = "array",
    [TAG_STRUCT]   = "struct",
    [TAG_ENUM]     = "enum",
    [TAG_UNION]    = "union",
    [TAG_FUNCTION] = "function",
    [TAG_TUPLE]    = "tuple",
    [TAG_HASHMAP]  = "hashmap",
    [TAG_OPAQUE]   = "opaque",
};

const char *isa_tag_name(uint8_t tag) {
    if (tag < TAG_COUNT) {
        return tag_names[tag];
    }
    return "UNKNOWN";
}

/* ========================================================================
 * Instruction Metadata Table
 *
 * Indexed by opcode value. Invalid opcodes have name=NULL.
 * We use a flat array up to OP_COUNT for O(1) lookup.
 * ======================================================================== */

/* Helper macros for concise table entries */
#define INSTR0(op, nm) \
    [op] = { .name = nm, .opcode = op, .operand_count = 0, .operands = {0} }

#define INSTR1(op, nm, t1) \
    [op] = { .name = nm, .opcode = op, .operand_count = 1, \
             .operands = { t1 } }

#define INSTR2(op, nm, t1, t2) \
    [op] = { .name = nm, .opcode = op, .operand_count = 2, \
             .operands = { t1, t2 } }

#define INSTR3(op, nm, t1, t2, t3) \
    [op] = { .name = nm, .opcode = op, .operand_count = 3, \
             .operands = { t1, t2, t3 } }

static const InstructionInfo instruction_table[256] = {
    /* Stack & Constants */
    INSTR0(OP_NOP,       "NOP"),
    INSTR1(OP_PUSH_I64,  "PUSH_I64",  OPERAND_I64),
    INSTR1(OP_PUSH_F64,  "PUSH_F64",  OPERAND_F64),
    INSTR1(OP_PUSH_BOOL, "PUSH_BOOL", OPERAND_U8),
    INSTR1(OP_PUSH_STR,  "PUSH_STR",  OPERAND_U32),
    INSTR0(OP_PUSH_VOID, "PUSH_VOID"),
    INSTR1(OP_PUSH_U8,   "PUSH_U8",   OPERAND_U8),
    INSTR0(OP_DUP,       "DUP"),
    INSTR0(OP_POP,       "POP"),
    INSTR0(OP_SWAP,      "SWAP"),
    INSTR0(OP_ROT3,      "ROT3"),

    /* Variable Access */
    INSTR1(OP_LOAD_LOCAL,    "LOAD_LOCAL",    OPERAND_U16),
    INSTR1(OP_STORE_LOCAL,   "STORE_LOCAL",   OPERAND_U16),
    INSTR1(OP_LOAD_GLOBAL,   "LOAD_GLOBAL",   OPERAND_U32),
    INSTR1(OP_STORE_GLOBAL,  "STORE_GLOBAL",  OPERAND_U32),
    INSTR2(OP_LOAD_UPVALUE,  "LOAD_UPVALUE",  OPERAND_U16, OPERAND_U16),
    INSTR2(OP_STORE_UPVALUE, "STORE_UPVALUE", OPERAND_U16, OPERAND_U16),

    /* Arithmetic */
    INSTR0(OP_ADD, "ADD"),
    INSTR0(OP_SUB, "SUB"),
    INSTR0(OP_MUL, "MUL"),
    INSTR0(OP_DIV, "DIV"),
    INSTR0(OP_MOD, "MOD"),
    INSTR0(OP_NEG, "NEG"),

    /* Comparison */
    INSTR0(OP_EQ, "EQ"),
    INSTR0(OP_NE, "NE"),
    INSTR0(OP_LT, "LT"),
    INSTR0(OP_LE, "LE"),
    INSTR0(OP_GT, "GT"),
    INSTR0(OP_GE, "GE"),

    /* Logic */
    INSTR0(OP_AND, "AND"),
    INSTR0(OP_OR,  "OR"),
    INSTR0(OP_NOT, "NOT"),

    /* Control Flow */
    INSTR1(OP_JMP,           "JMP",           OPERAND_I32),
    INSTR1(OP_JMP_TRUE,      "JMP_TRUE",      OPERAND_I32),
    INSTR1(OP_JMP_FALSE,     "JMP_FALSE",     OPERAND_I32),
    INSTR1(OP_CALL,          "CALL",          OPERAND_U32),
    INSTR0(OP_CALL_INDIRECT, "CALL_INDIRECT"),
    INSTR0(OP_RET,           "RET"),
    INSTR1(OP_CALL_EXTERN,   "CALL_EXTERN",   OPERAND_U32),
    INSTR2(OP_CALL_MODULE,   "CALL_MODULE",   OPERAND_U32, OPERAND_U32),

    /* String Ops */
    INSTR0(OP_STR_LEN,        "STR_LEN"),
    INSTR0(OP_STR_CONCAT,     "STR_CONCAT"),
    INSTR0(OP_STR_SUBSTR,     "STR_SUBSTR"),
    INSTR0(OP_STR_CONTAINS,   "STR_CONTAINS"),
    INSTR0(OP_STR_EQ,         "STR_EQ"),
    INSTR0(OP_STR_CHAR_AT,    "STR_CHAR_AT"),
    INSTR0(OP_STR_FROM_INT,   "STR_FROM_INT"),
    INSTR0(OP_STR_FROM_FLOAT, "STR_FROM_FLOAT"),

    /* Array Ops */
    INSTR1(OP_ARR_NEW,     "ARR_NEW",     OPERAND_U8),
    INSTR0(OP_ARR_PUSH,    "ARR_PUSH"),
    INSTR0(OP_ARR_POP,     "ARR_POP"),
    INSTR0(OP_ARR_GET,     "ARR_GET"),
    INSTR0(OP_ARR_SET,     "ARR_SET"),
    INSTR0(OP_ARR_LEN,     "ARR_LEN"),
    INSTR0(OP_ARR_SLICE,   "ARR_SLICE"),
    INSTR0(OP_ARR_REMOVE,  "ARR_REMOVE"),
    INSTR2(OP_ARR_LITERAL, "ARR_LITERAL", OPERAND_U8, OPERAND_U16),

    /* Struct Ops */
    INSTR1(OP_STRUCT_NEW,     "STRUCT_NEW",     OPERAND_U32),
    INSTR1(OP_STRUCT_GET,     "STRUCT_GET",     OPERAND_U16),
    INSTR1(OP_STRUCT_SET,     "STRUCT_SET",     OPERAND_U16),
    INSTR2(OP_STRUCT_LITERAL, "STRUCT_LITERAL", OPERAND_U32, OPERAND_U16),

    /* Union/Enum Ops */
    INSTR3(OP_UNION_CONSTRUCT, "UNION_CONSTRUCT", OPERAND_U32, OPERAND_U16, OPERAND_U16),
    INSTR0(OP_UNION_TAG,       "UNION_TAG"),
    INSTR1(OP_UNION_FIELD,     "UNION_FIELD",     OPERAND_U16),
    INSTR2(OP_MATCH_TAG,       "MATCH_TAG",       OPERAND_U16, OPERAND_I32),
    INSTR2(OP_ENUM_VAL,        "ENUM_VAL",        OPERAND_U32, OPERAND_U16),

    /* Tuple Ops */
    INSTR1(OP_TUPLE_NEW, "TUPLE_NEW", OPERAND_U16),
    INSTR1(OP_TUPLE_GET, "TUPLE_GET", OPERAND_U16),

    /* Hashmap Ops */
    INSTR2(OP_HM_NEW,    "HM_NEW",    OPERAND_U8, OPERAND_U8),
    INSTR0(OP_HM_GET,    "HM_GET"),
    INSTR0(OP_HM_SET,    "HM_SET"),
    INSTR0(OP_HM_HAS,    "HM_HAS"),
    INSTR0(OP_HM_DELETE, "HM_DELETE"),
    INSTR0(OP_HM_KEYS,   "HM_KEYS"),
    INSTR0(OP_HM_VALUES, "HM_VALUES"),
    INSTR0(OP_HM_LEN,    "HM_LEN"),

    /* GC/Memory */
    INSTR0(OP_GC_RETAIN,      "GC_RETAIN"),
    INSTR0(OP_GC_RELEASE,     "GC_RELEASE"),
    INSTR0(OP_GC_SCOPE_ENTER, "GC_SCOPE_ENTER"),
    INSTR0(OP_GC_SCOPE_EXIT,  "GC_SCOPE_EXIT"),

    /* Type Casts */
    INSTR0(OP_CAST_INT,    "CAST_INT"),
    INSTR0(OP_CAST_FLOAT,  "CAST_FLOAT"),
    INSTR0(OP_CAST_BOOL,   "CAST_BOOL"),
    INSTR0(OP_CAST_STRING, "CAST_STRING"),
    INSTR1(OP_TYPE_CHECK,  "TYPE_CHECK", OPERAND_U8),

    /* Closures */
    INSTR2(OP_CLOSURE_NEW,  "CLOSURE_NEW",  OPERAND_U32, OPERAND_U16),
    INSTR0(OP_CLOSURE_CALL, "CLOSURE_CALL"),

    /* I/O & Debug */
    INSTR0(OP_PRINT,      "PRINT"),
    INSTR0(OP_ASSERT,     "ASSERT"),
    INSTR1(OP_DEBUG_LINE, "DEBUG_LINE", OPERAND_U32),
    INSTR0(OP_HALT,       "HALT"),
    INSTR0(OP_PRINTLN,    "PRINTLN"),

    /* Opaque Proxy */
    INSTR0(OP_OPAQUE_NULL,  "OPAQUE_NULL"),
    INSTR0(OP_OPAQUE_VALID, "OPAQUE_VALID"),
};

/* ========================================================================
 * API Implementation
 * ======================================================================== */

const InstructionInfo *isa_get_info(uint8_t opcode) {
    const InstructionInfo *info = &instruction_table[opcode];
    if (info->name == NULL) {
        return NULL;
    }
    return info;
}

uint32_t isa_operand_size(OperandType type) {
    switch (type) {
        case OPERAND_NONE: return 0;
        case OPERAND_U8:   return 1;
        case OPERAND_U16:  return 2;
        case OPERAND_U32:  return 4;
        case OPERAND_I32:  return 4;
        case OPERAND_I64:  return 8;
        case OPERAND_F64:  return 8;
    }
    return 0;
}

/* ---- Little-endian encoding helpers ---- */

static void write_u16(uint8_t *buf, uint16_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
}

static void write_u32(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
}

static void write_i32(uint8_t *buf, int32_t val) {
    write_u32(buf, (uint32_t)val);
}

static void write_i64(uint8_t *buf, int64_t val) {
    uint64_t uval = (uint64_t)val;
    for (int i = 0; i < 8; i++) {
        buf[i] = (uint8_t)(uval & 0xFF);
        uval >>= 8;
    }
}

static void write_f64(uint8_t *buf, double val) {
    uint64_t bits;
    memcpy(&bits, &val, sizeof(bits));
    for (int i = 0; i < 8; i++) {
        buf[i] = (uint8_t)(bits & 0xFF);
        bits >>= 8;
    }
}

/* ---- Little-endian decoding helpers ---- */

static uint16_t read_u16(const uint8_t *buf) {
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

static uint32_t read_u32(const uint8_t *buf) {
    return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}

static int32_t read_i32(const uint8_t *buf) {
    return (int32_t)read_u32(buf);
}

static int64_t read_i64(const uint8_t *buf) {
    uint64_t val = 0;
    for (int i = 7; i >= 0; i--) {
        val = (val << 8) | buf[i];
    }
    return (int64_t)val;
}

static double read_f64(const uint8_t *buf) {
    uint64_t bits = 0;
    for (int i = 7; i >= 0; i--) {
        bits = (bits << 8) | buf[i];
    }
    double val;
    memcpy(&val, &bits, sizeof(val));
    return val;
}

/* ---- Encode ---- */

uint32_t isa_encode(const DecodedInstruction *instr, uint8_t *buf, size_t buf_size) {
    const InstructionInfo *info = isa_get_info(instr->opcode);
    if (info == NULL) {
        return 0;
    }

    /* Calculate total size */
    uint32_t total = 1; /* opcode byte */
    for (int i = 0; i < info->operand_count; i++) {
        total += isa_operand_size(info->operands[i]);
    }
    if (total > buf_size) {
        return 0;
    }

    /* Write opcode */
    buf[0] = instr->opcode;
    uint32_t pos = 1;

    /* Write operands */
    for (int i = 0; i < info->operand_count; i++) {
        switch (info->operands[i]) {
            case OPERAND_NONE:
                break;
            case OPERAND_U8:
                buf[pos] = instr->operands[i].u8;
                pos += 1;
                break;
            case OPERAND_U16:
                write_u16(&buf[pos], instr->operands[i].u16);
                pos += 2;
                break;
            case OPERAND_U32:
                write_u32(&buf[pos], instr->operands[i].u32);
                pos += 4;
                break;
            case OPERAND_I32:
                write_i32(&buf[pos], instr->operands[i].i32);
                pos += 4;
                break;
            case OPERAND_I64:
                write_i64(&buf[pos], instr->operands[i].i64);
                pos += 8;
                break;
            case OPERAND_F64:
                write_f64(&buf[pos], instr->operands[i].f64);
                pos += 8;
                break;
        }
    }

    return pos;
}

/* ---- Decode ---- */

uint32_t isa_decode(const uint8_t *buf, size_t buf_size, DecodedInstruction *out) {
    if (buf_size < 1) {
        return 0;
    }

    uint8_t opcode = buf[0];
    const InstructionInfo *info = isa_get_info(opcode);
    if (info == NULL) {
        return 0;
    }

    memset(out, 0, sizeof(*out));
    out->opcode = opcode;
    out->operand_count = info->operand_count;

    uint32_t pos = 1;
    for (int i = 0; i < info->operand_count; i++) {
        uint32_t sz = isa_operand_size(info->operands[i]);
        if (pos + sz > buf_size) {
            return 0; /* truncated */
        }
        out->operand_types[i] = info->operands[i];
        switch (info->operands[i]) {
            case OPERAND_NONE:
                break;
            case OPERAND_U8:
                out->operands[i].u8 = buf[pos];
                break;
            case OPERAND_U16:
                out->operands[i].u16 = read_u16(&buf[pos]);
                break;
            case OPERAND_U32:
                out->operands[i].u32 = read_u32(&buf[pos]);
                break;
            case OPERAND_I32:
                out->operands[i].i32 = read_i32(&buf[pos]);
                break;
            case OPERAND_I64:
                out->operands[i].i64 = read_i64(&buf[pos]);
                break;
            case OPERAND_F64:
                out->operands[i].f64 = read_f64(&buf[pos]);
                break;
        }
        pos += sz;
    }

    out->byte_length = pos;
    return pos;
}

/* ---- Lookup by name ---- */

int isa_opcode_by_name(const char *name) {
    for (int i = 0; i < 256; i++) {
        if (instruction_table[i].name != NULL &&
            strcmp(instruction_table[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}
