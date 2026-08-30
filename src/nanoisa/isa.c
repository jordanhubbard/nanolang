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

static InstructionInfo instruction_table[256];
static bool instruction_table_ready;

static void init_instruction_table(void) {
    if (instruction_table_ready) return;
    for (size_t i = 0; i < NANOISA_LEGACY_OPCODE_COUNT; i++) {
        const NanoisaSchemaOpcode *source = &nanoisa_schema_opcodes[i];
        InstructionInfo *target = &instruction_table[source->opcode];
        target->name = source->name;
        target->opcode = source->opcode;
        target->operand_count = source->operand_count;
        memcpy(target->operands, source->operands, sizeof(target->operands));
    }
    instruction_table_ready = true;
}

/* ========================================================================
 * API Implementation
 * ======================================================================== */

const InstructionInfo *isa_get_info(uint8_t opcode) {
    init_instruction_table();
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
