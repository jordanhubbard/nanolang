/*
 * NanoISA Test Suite
 *
 * Tests: ISA encode/decode, NVM format serialize/deserialize,
 *        assembler, disassembler, and round-trip.
 */

#include "isa.h"
#include "nvm_format.h"
#include "assembler.h"
#include "disassembler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Test Framework (minimal)
 * ======================================================================== */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ_INT(a, b, msg) do { \
    tests_run++; \
    long long _a = (long long)(a), _b = (long long)(b); \
    if (_a != _b) { \
        printf("  FAIL [%s:%d]: %s (expected %lld, got %lld)\n", \
               __FILE__, __LINE__, msg, _b, _a); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ_STR(a, b, msg) do { \
    tests_run++; \
    if (strcmp((a), (b)) != 0) { \
        printf("  FAIL [%s:%d]: %s (expected \"%s\", got \"%s\")\n", \
               __FILE__, __LINE__, msg, (b), (a)); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    printf("  %s...\n", #fn); \
    fn(); \
} while(0)

/* ========================================================================
 * ISA Tests
 * ======================================================================== */

static void test_tag_names(void) {
    ASSERT_EQ_STR(isa_tag_name(TAG_VOID), "void", "TAG_VOID name");
    ASSERT_EQ_STR(isa_tag_name(TAG_INT), "int", "TAG_INT name");
    ASSERT_EQ_STR(isa_tag_name(TAG_FLOAT), "float", "TAG_FLOAT name");
    ASSERT_EQ_STR(isa_tag_name(TAG_BOOL), "bool", "TAG_BOOL name");
    ASSERT_EQ_STR(isa_tag_name(TAG_STRING), "string", "TAG_STRING name");
    ASSERT_EQ_STR(isa_tag_name(TAG_ARRAY), "array", "TAG_ARRAY name");
    ASSERT_EQ_STR(isa_tag_name(TAG_STRUCT), "struct", "TAG_STRUCT name");
    ASSERT_EQ_STR(isa_tag_name(TAG_ENUM), "enum", "TAG_ENUM name");
    ASSERT_EQ_STR(isa_tag_name(TAG_UNION), "union", "TAG_UNION name");
    ASSERT_EQ_STR(isa_tag_name(TAG_FUNCTION), "function", "TAG_FUNCTION name");
    ASSERT_EQ_STR(isa_tag_name(TAG_TUPLE), "tuple", "TAG_TUPLE name");
    ASSERT_EQ_STR(isa_tag_name(TAG_HASHMAP), "hashmap", "TAG_HASHMAP name");
    ASSERT_EQ_STR(isa_tag_name(TAG_OPAQUE), "opaque", "TAG_OPAQUE name");
    ASSERT_EQ_STR(isa_tag_name(0xFF), "UNKNOWN", "Invalid tag name");
}

static void test_instruction_info(void) {
    const InstructionInfo *info;

    /* NOP - no operands */
    info = isa_get_info(OP_NOP);
    ASSERT(info != NULL, "NOP info exists");
    ASSERT_EQ_STR(info->name, "NOP", "NOP name");
    ASSERT_EQ_INT(info->operand_count, 0, "NOP operand count");

    /* PUSH_I64 - one i64 operand */
    info = isa_get_info(OP_PUSH_I64);
    ASSERT(info != NULL, "PUSH_I64 info exists");
    ASSERT_EQ_INT(info->operand_count, 1, "PUSH_I64 operand count");
    ASSERT_EQ_INT(info->operands[0], OPERAND_I64, "PUSH_I64 operand type");

    /* PUSH_F64 - one f64 operand */
    info = isa_get_info(OP_PUSH_F64);
    ASSERT(info != NULL, "PUSH_F64 info exists");
    ASSERT_EQ_INT(info->operands[0], OPERAND_F64, "PUSH_F64 operand type");

    /* LOAD_LOCAL - one u16 operand */
    info = isa_get_info(OP_LOAD_LOCAL);
    ASSERT(info != NULL, "LOAD_LOCAL info exists");
    ASSERT_EQ_INT(info->operand_count, 1, "LOAD_LOCAL operand count");
    ASSERT_EQ_INT(info->operands[0], OPERAND_U16, "LOAD_LOCAL operand type");

    /* LOAD_UPVALUE - two u16 operands */
    info = isa_get_info(OP_LOAD_UPVALUE);
    ASSERT(info != NULL, "LOAD_UPVALUE info exists");
    ASSERT_EQ_INT(info->operand_count, 2, "LOAD_UPVALUE operand count");
    ASSERT_EQ_INT(info->operands[0], OPERAND_U16, "LOAD_UPVALUE op0 type");
    ASSERT_EQ_INT(info->operands[1], OPERAND_U16, "LOAD_UPVALUE op1 type");

    /* CALL_MODULE - two u32 operands */
    info = isa_get_info(OP_CALL_MODULE);
    ASSERT(info != NULL, "CALL_MODULE info exists");
    ASSERT_EQ_INT(info->operand_count, 2, "CALL_MODULE operand count");

    /* UNION_CONSTRUCT - three operands */
    info = isa_get_info(OP_UNION_CONSTRUCT);
    ASSERT(info != NULL, "UNION_CONSTRUCT info exists");
    ASSERT_EQ_INT(info->operand_count, 3, "UNION_CONSTRUCT operand count");

    /* Invalid opcode */
    info = isa_get_info(0xFF);
    ASSERT(info == NULL, "Invalid opcode returns NULL");
}

static void test_opcode_by_name(void) {
    ASSERT_EQ_INT(isa_opcode_by_name("NOP"), OP_NOP, "NOP by name");
    ASSERT_EQ_INT(isa_opcode_by_name("PUSH_I64"), OP_PUSH_I64, "PUSH_I64 by name");
    ASSERT_EQ_INT(isa_opcode_by_name("ADD"), OP_ADD, "ADD by name");
    ASSERT_EQ_INT(isa_opcode_by_name("JMP_FALSE"), OP_JMP_FALSE, "JMP_FALSE by name");
    ASSERT_EQ_INT(isa_opcode_by_name("HALT"), OP_HALT, "HALT by name");
    ASSERT_EQ_INT(isa_opcode_by_name("NONEXISTENT"), -1, "Unknown opcode");
}

static void test_encode_decode_no_operands(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_ADD;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 1, "ADD encodes to 1 byte");
    ASSERT_EQ_INT(buf[0], OP_ADD, "ADD opcode byte");

    DecodedInstruction decoded;
    uint32_t consumed = isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(consumed, 1, "ADD decodes from 1 byte");
    ASSERT_EQ_INT(decoded.opcode, OP_ADD, "ADD decoded opcode");
    ASSERT_EQ_INT(decoded.operand_count, 0, "ADD decoded operand count");
}

static void test_encode_decode_i64(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_PUSH_I64;
    instr.operands[0].i64 = -42;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 9, "PUSH_I64 encodes to 9 bytes");

    DecodedInstruction decoded;
    uint32_t consumed = isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(consumed, 9, "PUSH_I64 decodes from 9 bytes");
    ASSERT_EQ_INT(decoded.operands[0].i64, -42, "PUSH_I64 value preserved");
}

static void test_encode_decode_i64_large(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_PUSH_I64;
    instr.operands[0].i64 = 0x7FFFFFFFFFFFFFFFLL;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    DecodedInstruction decoded;
    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].i64, 0x7FFFFFFFFFFFFFFFLL, "PUSH_I64 max i64");

    instr.operands[0].i64 = (int64_t)0x8000000000000000LL; /* INT64_MIN */
    encoded = isa_encode(&instr, buf, sizeof(buf));
    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].i64, (int64_t)0x8000000000000000LL, "PUSH_I64 min i64");
}

static void test_encode_decode_f64(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_PUSH_F64;
    instr.operands[0].f64 = 3.14159265358979;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 9, "PUSH_F64 encodes to 9 bytes");

    DecodedInstruction decoded;
    isa_decode(buf, encoded, &decoded);
    ASSERT(fabs(decoded.operands[0].f64 - 3.14159265358979) < 1e-15,
           "PUSH_F64 value preserved");
}

static void test_encode_decode_u8(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_PUSH_BOOL;
    instr.operands[0].u8 = 1;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 2, "PUSH_BOOL encodes to 2 bytes");

    DecodedInstruction decoded;
    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].u8, 1, "PUSH_BOOL value preserved");
}

static void test_encode_decode_u16(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_LOAD_LOCAL;
    instr.operands[0].u16 = 0x1234;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 3, "LOAD_LOCAL encodes to 3 bytes");

    DecodedInstruction decoded;
    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].u16, 0x1234, "LOAD_LOCAL value preserved");
}

static void test_encode_decode_u32(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_PUSH_STR;
    instr.operands[0].u32 = 0xDEADBEEF;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 5, "PUSH_STR encodes to 5 bytes");

    DecodedInstruction decoded;
    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].u32, 0xDEADBEEF, "PUSH_STR value preserved");
}

static void test_encode_decode_i32(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};
    instr.opcode = OP_JMP;
    instr.operands[0].i32 = -100;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 5, "JMP encodes to 5 bytes");

    DecodedInstruction decoded;
    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].i32, -100, "JMP negative offset preserved");
}

static void test_encode_decode_multi_operand(void) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr = {0};

    /* LOAD_UPVALUE: u16, u16 */
    instr.opcode = OP_LOAD_UPVALUE;
    instr.operands[0].u16 = 2;
    instr.operands[1].u16 = 5;

    uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 5, "LOAD_UPVALUE encodes to 5 bytes");

    DecodedInstruction decoded;
    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].u16, 2, "LOAD_UPVALUE depth");
    ASSERT_EQ_INT(decoded.operands[1].u16, 5, "LOAD_UPVALUE index");

    /* UNION_CONSTRUCT: u32, u16, u16 */
    memset(&instr, 0, sizeof(instr));
    instr.opcode = OP_UNION_CONSTRUCT;
    instr.operands[0].u32 = 10;
    instr.operands[1].u16 = 3;
    instr.operands[2].u16 = 2;

    encoded = isa_encode(&instr, buf, sizeof(buf));
    ASSERT_EQ_INT(encoded, 9, "UNION_CONSTRUCT encodes to 9 bytes");

    isa_decode(buf, encoded, &decoded);
    ASSERT_EQ_INT(decoded.operands[0].u32, 10, "UNION_CONSTRUCT def_idx");
    ASSERT_EQ_INT(decoded.operands[1].u16, 3, "UNION_CONSTRUCT variant");
    ASSERT_EQ_INT(decoded.operands[2].u16, 2, "UNION_CONSTRUCT fields");
}

static void test_decode_truncated(void) {
    uint8_t buf[1] = { OP_PUSH_I64 };
    DecodedInstruction decoded;
    uint32_t consumed = isa_decode(buf, 1, &decoded);
    ASSERT_EQ_INT(consumed, 0, "Truncated PUSH_I64 returns 0");
}

static void test_decode_invalid_opcode(void) {
    uint8_t buf[1] = { 0xFF };
    DecodedInstruction decoded;
    uint32_t consumed = isa_decode(buf, 1, &decoded);
    ASSERT_EQ_INT(consumed, 0, "Invalid opcode returns 0");
}

static void test_encode_all_categories(void) {
    /* Verify every opcode category encodes without error */
    uint8_t opcodes[] = {
        OP_NOP, OP_PUSH_I64, OP_PUSH_F64, OP_PUSH_BOOL, OP_PUSH_STR,
        OP_PUSH_VOID, OP_PUSH_U8, OP_DUP, OP_POP, OP_SWAP, OP_ROT3,
        OP_LOAD_LOCAL, OP_STORE_LOCAL, OP_LOAD_GLOBAL, OP_STORE_GLOBAL,
        OP_LOAD_UPVALUE, OP_STORE_UPVALUE,
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_NEG,
        OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE,
        OP_AND, OP_OR, OP_NOT,
        OP_JMP, OP_JMP_TRUE, OP_JMP_FALSE, OP_CALL, OP_CALL_INDIRECT, OP_RET,
        OP_CALL_EXTERN, OP_CALL_MODULE,
        OP_STR_LEN, OP_STR_CONCAT, OP_STR_SUBSTR, OP_STR_CONTAINS,
        OP_STR_EQ, OP_STR_CHAR_AT, OP_STR_FROM_INT, OP_STR_FROM_FLOAT,
        OP_ARR_NEW, OP_ARR_PUSH, OP_ARR_POP, OP_ARR_GET, OP_ARR_SET,
        OP_ARR_LEN, OP_ARR_SLICE, OP_ARR_REMOVE, OP_ARR_LITERAL,
        OP_STRUCT_NEW, OP_STRUCT_GET, OP_STRUCT_SET, OP_STRUCT_LITERAL,
        OP_UNION_CONSTRUCT, OP_UNION_TAG, OP_UNION_FIELD, OP_MATCH_TAG, OP_ENUM_VAL,
        OP_TUPLE_NEW, OP_TUPLE_GET,
        OP_HM_NEW, OP_HM_GET, OP_HM_SET, OP_HM_HAS, OP_HM_DELETE,
        OP_HM_KEYS, OP_HM_VALUES, OP_HM_LEN,
        OP_GC_RETAIN, OP_GC_RELEASE, OP_GC_SCOPE_ENTER, OP_GC_SCOPE_EXIT,
        OP_CAST_INT, OP_CAST_FLOAT, OP_CAST_BOOL, OP_CAST_STRING, OP_TYPE_CHECK,
        OP_CLOSURE_NEW, OP_CLOSURE_CALL,
        OP_PRINT, OP_ASSERT, OP_DEBUG_LINE, OP_HALT,
        OP_OPAQUE_NULL, OP_OPAQUE_VALID,
    };
    int count = sizeof(opcodes) / sizeof(opcodes[0]);

    for (int i = 0; i < count; i++) {
        uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
        DecodedInstruction instr = {0};
        instr.opcode = opcodes[i];
        /* Set dummy operand values */
        instr.operands[0].i64 = 1;
        instr.operands[1].u32 = 2;
        instr.operands[2].u16 = 3;

        uint32_t encoded = isa_encode(&instr, buf, sizeof(buf));
        ASSERT(encoded > 0, "Opcode encodes successfully");

        DecodedInstruction decoded;
        uint32_t consumed = isa_decode(buf, encoded, &decoded);
        ASSERT_EQ_INT(consumed, encoded, "Decoded same number of bytes");
        ASSERT_EQ_INT(decoded.opcode, opcodes[i], "Opcode round-trips");
    }
}

/* ========================================================================
 * NVM Format Tests
 * ======================================================================== */

static void test_crc32(void) {
    const uint8_t data[] = "Hello, World!";
    uint32_t crc = nvm_crc32(data, 13);
    /* Known CRC32 value for "Hello, World!" */
    ASSERT_EQ_INT(crc, 0xEC4AC3D0, "CRC32 of 'Hello, World!'");
}

static void test_module_new_free(void) {
    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL, "Module created");
    ASSERT_EQ_INT(mod->header.magic[0], 'N', "Magic byte 0");
    ASSERT_EQ_INT(mod->header.magic[1], 'V', "Magic byte 1");
    ASSERT_EQ_INT(mod->header.magic[2], 'M', "Magic byte 2");
    ASSERT_EQ_INT(mod->header.magic[3], 0x01, "Magic byte 3");
    ASSERT_EQ_INT(mod->string_count, 0, "No strings initially");
    ASSERT_EQ_INT(mod->function_count, 0, "No functions initially");
    nvm_module_free(mod);
}

static void test_string_pool(void) {
    NvmModule *mod = nvm_module_new();

    uint32_t idx0 = nvm_add_string(mod, "hello", 5);
    uint32_t idx1 = nvm_add_string(mod, "world", 5);
    uint32_t idx2 = nvm_add_string(mod, "hello", 5); /* duplicate */

    ASSERT_EQ_INT(idx0, 0, "First string at index 0");
    ASSERT_EQ_INT(idx1, 1, "Second string at index 1");
    ASSERT_EQ_INT(idx2, 0, "Duplicate returns existing index");
    ASSERT_EQ_INT(mod->string_count, 2, "Two unique strings");
    ASSERT_EQ_STR(nvm_get_string(mod, 0), "hello", "String 0 content");
    ASSERT_EQ_STR(nvm_get_string(mod, 1), "world", "String 1 content");
    ASSERT(nvm_get_string(mod, 99) == NULL, "Out of range returns NULL");

    nvm_module_free(mod);
}

static void test_function_table(void) {
    NvmModule *mod = nvm_module_new();

    NvmFunctionEntry fn1 = { .name_idx = 0, .arity = 2, .code_offset = 0,
                             .code_length = 10, .local_count = 3, .upvalue_count = 0 };
    NvmFunctionEntry fn2 = { .name_idx = 1, .arity = 0, .code_offset = 10,
                             .code_length = 20, .local_count = 5, .upvalue_count = 1 };

    uint32_t i0 = nvm_add_function(mod, &fn1);
    uint32_t i1 = nvm_add_function(mod, &fn2);

    ASSERT_EQ_INT(i0, 0, "First function at index 0");
    ASSERT_EQ_INT(i1, 1, "Second function at index 1");
    ASSERT_EQ_INT(mod->function_count, 2, "Two functions");
    ASSERT_EQ_INT(mod->functions[0].arity, 2, "Function 0 arity");
    ASSERT_EQ_INT(mod->functions[1].local_count, 5, "Function 1 locals");

    nvm_module_free(mod);
}

static void test_code_append(void) {
    NvmModule *mod = nvm_module_new();

    uint8_t code1[] = { OP_PUSH_I64, 0x2A, 0, 0, 0, 0, 0, 0, 0 };
    uint8_t code2[] = { OP_PRINT, OP_HALT };

    uint32_t off1 = nvm_append_code(mod, code1, sizeof(code1));
    uint32_t off2 = nvm_append_code(mod, code2, sizeof(code2));

    ASSERT_EQ_INT(off1, 0, "First code at offset 0");
    ASSERT_EQ_INT(off2, 9, "Second code after first");
    ASSERT_EQ_INT(mod->code_size, 11, "Total code size");
    ASSERT_EQ_INT(mod->code[0], OP_PUSH_I64, "First opcode");
    ASSERT_EQ_INT(mod->code[9], OP_PRINT, "Second opcode");

    nvm_module_free(mod);
}

static void test_serialize_deserialize(void) {
    NvmModule *mod = nvm_module_new();
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = 0;

    nvm_add_string(mod, "main", 4);
    nvm_add_string(mod, "hello", 5);

    NvmFunctionEntry fn = { .name_idx = 0, .arity = 0, .code_offset = 0,
                            .code_length = 0, .local_count = 1, .upvalue_count = 0 };
    nvm_add_function(mod, &fn);

    /* Some bytecode */
    uint8_t code[] = { OP_PUSH_STR, 1, 0, 0, 0, OP_PRINT, OP_HALT };
    uint32_t off = nvm_append_code(mod, code, sizeof(code));
    mod->functions[0].code_offset = off;
    mod->functions[0].code_length = sizeof(code);

    /* Debug info */
    nvm_add_debug_entry(mod, 0, 1);
    nvm_add_debug_entry(mod, 5, 2);

    /* Serialize */
    uint32_t out_size;
    uint8_t *data = nvm_serialize(mod, &out_size);
    ASSERT(data != NULL, "Serialization succeeded");
    ASSERT(out_size > NVM_HEADER_SIZE, "Output larger than header");

    /* Deserialize */
    NvmModule *mod2 = nvm_deserialize(data, out_size);
    ASSERT(mod2 != NULL, "Deserialization succeeded");

    /* Verify header */
    ASSERT_EQ_INT(mod2->header.format_version, NVM_FORMAT_VERSION, "Format version");
    ASSERT_EQ_INT(mod2->header.flags, NVM_FLAG_HAS_MAIN, "Flags");
    ASSERT_EQ_INT(mod2->header.entry_point, 0, "Entry point");

    /* Verify strings */
    ASSERT_EQ_INT(mod2->string_count, 2, "String count");
    ASSERT_EQ_STR(nvm_get_string(mod2, 0), "main", "String 0");
    ASSERT_EQ_STR(nvm_get_string(mod2, 1), "hello", "String 1");

    /* Verify functions */
    ASSERT_EQ_INT(mod2->function_count, 1, "Function count");
    ASSERT_EQ_INT(mod2->functions[0].arity, 0, "Function arity");
    ASSERT_EQ_INT(mod2->functions[0].local_count, 1, "Function locals");
    ASSERT_EQ_INT(mod2->functions[0].code_length, sizeof(code), "Function code length");

    /* Verify code */
    ASSERT_EQ_INT(mod2->code_size, sizeof(code), "Code size");
    ASSERT_EQ_INT(mod2->code[0], OP_PUSH_STR, "Code byte 0");
    ASSERT_EQ_INT(mod2->code[5], OP_PRINT, "Code byte 5");
    ASSERT_EQ_INT(mod2->code[6], OP_HALT, "Code byte 6");

    /* Verify debug */
    ASSERT_EQ_INT(mod2->debug_count, 2, "Debug entry count");
    ASSERT_EQ_INT(mod2->debug_entries[0].bytecode_offset, 0, "Debug 0 offset");
    ASSERT_EQ_INT(mod2->debug_entries[0].source_line, 1, "Debug 0 line");
    ASSERT_EQ_INT(mod2->debug_entries[1].bytecode_offset, 5, "Debug 1 offset");
    ASSERT_EQ_INT(mod2->debug_entries[1].source_line, 2, "Debug 1 line");

    free(data);
    nvm_module_free(mod);
    nvm_module_free(mod2);
}

static void test_validate_header(void) {
    NvmHeader header;
    header.magic[0] = 'N';
    header.magic[1] = 'V';
    header.magic[2] = 'M';
    header.magic[3] = 0x01;
    header.format_version = NVM_FORMAT_VERSION;
    header.section_count = 3;

    ASSERT(nvm_validate_header(&header), "Valid header");

    header.magic[0] = 'X';
    ASSERT(!nvm_validate_header(&header), "Bad magic rejected");

    header.magic[0] = 'N';
    header.format_version = 999;
    ASSERT(!nvm_validate_header(&header), "Bad version rejected");

    header.format_version = NVM_FORMAT_VERSION;
    header.section_count = NVM_MAX_SECTIONS + 1;
    ASSERT(!nvm_validate_header(&header), "Too many sections rejected");
}

static void test_corrupt_checksum(void) {
    NvmModule *mod = nvm_module_new();
    nvm_add_string(mod, "test", 4);

    uint32_t out_size;
    uint8_t *data = nvm_serialize(mod, &out_size);
    ASSERT(data != NULL, "Serialization succeeded");

    /* Corrupt a data byte */
    if (out_size > NVM_HEADER_SIZE + 5) {
        data[NVM_HEADER_SIZE + 5] ^= 0xFF;
    }

    NvmModule *mod2 = nvm_deserialize(data, out_size);
    ASSERT(mod2 == NULL, "Corrupt data rejected");

    free(data);
    nvm_module_free(mod);
}

/* ========================================================================
 * Assembler Tests
 * ======================================================================== */

static void test_asm_simple_program(void) {
    const char *src =
        ".string \"hello\"\n"
        ".entry 0\n"
        ".function main 0 1 0\n"
        "  PUSH_STR 0\n"
        "  PRINT\n"
        "  HALT\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);

    ASSERT(mod != NULL, "Assembly succeeded");
    ASSERT_EQ_INT(result.error, ASM_OK, "No error");
    ASSERT_EQ_INT(mod->string_count, 2, "2 strings (hello + main)");
    ASSERT_EQ_STR(nvm_get_string(mod, 0), "hello", "String 0");
    ASSERT_EQ_STR(nvm_get_string(mod, 1), "main", "String 1 (function name)");
    ASSERT_EQ_INT(mod->function_count, 1, "1 function");
    ASSERT_EQ_INT(mod->functions[0].arity, 0, "main arity 0");
    ASSERT_EQ_INT(mod->header.flags & NVM_FLAG_HAS_MAIN, NVM_FLAG_HAS_MAIN, "has_main flag");

    /* Check bytecode */
    ASSERT(mod->functions[0].code_length > 0, "Function has code");
    const uint8_t *code = mod->code + mod->functions[0].code_offset;
    ASSERT_EQ_INT(code[0], OP_PUSH_STR, "First instruction PUSH_STR");

    nvm_module_free(mod);
}

static void test_asm_labels_and_jumps(void) {
    const char *src =
        ".function test_loop 0 2 0\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 0\n"
        "loop_top:\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 10\n"
        "  LT\n"
        "  JMP_FALSE loop_end\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  ADD\n"
        "  STORE_LOCAL 0\n"
        "  JMP loop_top\n"
        "loop_end:\n"
        "  RET\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);

    ASSERT(mod != NULL, "Assembly with labels succeeded");
    ASSERT_EQ_INT(result.error, ASM_OK, "No error");
    ASSERT_EQ_INT(mod->function_count, 1, "1 function");
    ASSERT(mod->functions[0].code_length > 0, "Function has code");

    /* Verify jumps are resolved to offsets (not zero placeholders) */
    /* The JMP_FALSE should target loop_end, JMP should target loop_top */
    const uint8_t *code = mod->code + mod->functions[0].code_offset;
    uint32_t code_size = mod->functions[0].code_length;

    /* Decode instructions to find JMP_FALSE */
    uint32_t pos = 0;
    bool found_jmp_false = false;
    bool found_jmp = false;
    while (pos < code_size) {
        DecodedInstruction instr;
        uint32_t consumed = isa_decode(code + pos, code_size - pos, &instr);
        if (consumed == 0) break;

        if (instr.opcode == OP_JMP_FALSE) {
            found_jmp_false = true;
            ASSERT(instr.operands[0].i32 > 0, "JMP_FALSE offset is forward");
        }
        if (instr.opcode == OP_JMP) {
            found_jmp = true;
            ASSERT(instr.operands[0].i32 < 0, "JMP offset is backward (loop)");
        }

        pos += consumed;
    }
    ASSERT(found_jmp_false, "Found JMP_FALSE instruction");
    ASSERT(found_jmp, "Found JMP instruction");

    nvm_module_free(mod);
}

static void test_asm_all_operand_types(void) {
    const char *src =
        ".string \"test string\"\n"
        ".function test_ops 0 5 0\n"
        "  PUSH_I64 -9999\n"
        "  PUSH_F64 2.718281828\n"
        "  PUSH_BOOL 1\n"
        "  PUSH_STR 0\n"
        "  PUSH_U8 255\n"
        "  PUSH_VOID\n"
        "  LOAD_LOCAL 42\n"
        "  STORE_LOCAL 42\n"
        "  LOAD_GLOBAL 1000\n"
        "  STORE_GLOBAL 1000\n"
        "  LOAD_UPVALUE 1 3\n"
        "  STORE_UPVALUE 2 4\n"
        "  ARR_NEW 1\n"
        "  ARR_LITERAL 1 5\n"
        "  STRUCT_LITERAL 0 3\n"
        "  UNION_CONSTRUCT 0 1 2\n"
        "  MATCH_TAG 0 5\n"
        "  ENUM_VAL 0 3\n"
        "  TUPLE_NEW 4\n"
        "  HM_NEW 1 5\n"
        "  CLOSURE_NEW 0 3\n"
        "  TYPE_CHECK 4\n"
        "  DEBUG_LINE 100\n"
        "  CALL 0\n"
        "  CALL_EXTERN 1\n"
        "  CALL_MODULE 0 1\n"
        "  HALT\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod != NULL, "All operand types assembly succeeded");
    if (result.error != ASM_OK) {
        printf("    Error at line %u: %s\n", result.line, result.message);
    }
    ASSERT_EQ_INT(result.error, ASM_OK, "No error");

    nvm_module_free(mod);
}

static void test_asm_error_unknown_opcode(void) {
    const char *src =
        ".function test 0 0 0\n"
        "  NONEXISTENT_OP\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod == NULL, "Unknown opcode returns NULL");
    ASSERT_EQ_INT(result.error, ASM_ERR_UNKNOWN_OPCODE, "Error type");
    ASSERT_EQ_INT(result.line, 2, "Error on line 2");
}

static void test_asm_error_undefined_label(void) {
    const char *src =
        ".function test 0 0 0\n"
        "  JMP nonexistent\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod == NULL, "Undefined label returns NULL");
    ASSERT_EQ_INT(result.error, ASM_ERR_UNDEFINED_LABEL, "Error type");
}

static void test_asm_error_missing_end(void) {
    const char *src =
        ".function test 0 0 0\n"
        "  HALT\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod == NULL, "Missing .end returns NULL");
    ASSERT_EQ_INT(result.error, ASM_ERR_SYNTAX, "Error type");
}

static void test_asm_error_instruction_outside_function(void) {
    const char *src =
        "PUSH_I64 42\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod == NULL, "Instruction outside function returns NULL");
    ASSERT_EQ_INT(result.error, ASM_ERR_NO_FUNCTION, "Error type");
}

static void test_asm_comments_and_whitespace(void) {
    const char *src =
        "; This is a comment\n"
        "# This is also a comment\n"
        "\n"
        "  \n"
        ".function main 0 0 0\n"
        "  NOP ; inline comment\n"
        "  HALT\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod != NULL, "Comments and whitespace handled");
    ASSERT_EQ_INT(result.error, ASM_OK, "No error");
    nvm_module_free(mod);
}

static void test_asm_string_escapes(void) {
    const char *src =
        ".string \"hello\\nworld\"\n"
        ".string \"tab\\there\"\n"
        ".string \"quote\\\"inside\"\n"
        ".string \"backslash\\\\\"\n"
        ".function main 0 0 0\n"
        "  HALT\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod != NULL, "String escapes assembly succeeded");

    /* .function adds "main" as string 4, but our explicit strings are first */
    ASSERT(mod->string_count >= 4, "At least 4 strings");
    ASSERT_EQ_STR(nvm_get_string(mod, 0), "hello\nworld", "Newline escape");
    ASSERT_EQ_STR(nvm_get_string(mod, 1), "tab\there", "Tab escape");
    ASSERT_EQ_STR(nvm_get_string(mod, 2), "quote\"inside", "Quote escape");
    ASSERT_EQ_STR(nvm_get_string(mod, 3), "backslash\\", "Backslash escape");

    nvm_module_free(mod);
}

static void test_asm_multiple_functions(void) {
    const char *src =
        ".function add 2 2 0\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  ADD\n"
        "  RET\n"
        ".end\n"
        ".function main 0 1 0\n"
        "  PUSH_I64 3\n"
        "  PUSH_I64 4\n"
        "  CALL 0\n"
        "  PRINT\n"
        "  HALT\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod != NULL, "Multiple functions assembly succeeded");
    ASSERT_EQ_INT(mod->function_count, 2, "2 functions");
    ASSERT_EQ_INT(mod->functions[0].arity, 2, "add arity");
    ASSERT_EQ_INT(mod->functions[1].arity, 0, "main arity");

    /* Functions should have distinct code regions */
    ASSERT(mod->functions[0].code_offset != mod->functions[1].code_offset,
           "Functions at different offsets");

    nvm_module_free(mod);
}

/* ========================================================================
 * Disassembler Tests
 * ======================================================================== */

static void test_disasm_basic(void) {
    NvmModule *mod = nvm_module_new();
    nvm_add_string(mod, "main", 4);
    nvm_add_string(mod, "hello", 5);

    NvmFunctionEntry fn = { .name_idx = 0, .arity = 0, .code_offset = 0,
                            .code_length = 0, .local_count = 1, .upvalue_count = 0 };
    nvm_add_function(mod, &fn);

    /* Build bytecode manually */
    uint8_t code[] = { OP_PUSH_STR, 1, 0, 0, 0, OP_PRINT, OP_HALT };
    uint32_t off = nvm_append_code(mod, code, sizeof(code));
    mod->functions[0].code_offset = off;
    mod->functions[0].code_length = sizeof(code);

    char *output = disasm_module(mod);
    ASSERT(output != NULL, "Disassembly produced output");
    ASSERT(strstr(output, "PUSH_STR") != NULL, "Contains PUSH_STR");
    ASSERT(strstr(output, "PRINT") != NULL, "Contains PRINT");
    ASSERT(strstr(output, "HALT") != NULL, "Contains HALT");
    ASSERT(strstr(output, ".function main") != NULL, "Contains function header");
    ASSERT(strstr(output, ".end") != NULL, "Contains .end");
    ASSERT(strstr(output, "hello") != NULL, "Contains string comment");

    free(output);
    nvm_module_free(mod);
}

static void test_disasm_labels(void) {
    /* Assemble a program with jumps, then disassemble and check for labels */
    const char *src =
        ".function test 0 1 0\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 0\n"
        "loop:\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 5\n"
        "  LT\n"
        "  JMP_FALSE done\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  ADD\n"
        "  STORE_LOCAL 0\n"
        "  JMP loop\n"
        "done:\n"
        "  RET\n"
        ".end\n";

    AsmResult result;
    NvmModule *mod = asm_assemble(src, &result);
    ASSERT(mod != NULL, "Assembly for disasm test succeeded");

    char *output = disasm_module(mod);
    ASSERT(output != NULL, "Disassembly produced output");

    /* Should contain reconstructed labels like L0, L1 */
    ASSERT(strstr(output, "L") != NULL, "Contains reconstructed labels");
    ASSERT(strstr(output, "JMP_FALSE") != NULL, "Contains JMP_FALSE");
    ASSERT(strstr(output, "JMP") != NULL, "Contains JMP");

    free(output);
    nvm_module_free(mod);
}

/* ========================================================================
 * Round-Trip Test
 * ======================================================================== */

static void test_roundtrip_assemble_serialize_deserialize_disassemble(void) {
    const char *src =
        ".string \"Hello, NanoISA!\"\n"
        ".entry 0\n"
        ".function main 0 2 0\n"
        "  PUSH_I64 42\n"
        "  STORE_LOCAL 0\n"
        "  PUSH_STR 0\n"
        "  PRINT\n"
        "  LOAD_LOCAL 0\n"
        "  PRINT\n"
        "  HALT\n"
        ".end\n";

    /* Step 1: Assemble */
    AsmResult result;
    NvmModule *mod1 = asm_assemble(src, &result);
    ASSERT(mod1 != NULL, "Round-trip: assembly succeeded");
    ASSERT_EQ_INT(result.error, ASM_OK, "No assembly error");

    /* Step 2: Serialize to binary */
    uint32_t bin_size;
    uint8_t *bin = nvm_serialize(mod1, &bin_size);
    ASSERT(bin != NULL, "Round-trip: serialization succeeded");

    /* Step 3: Deserialize from binary */
    NvmModule *mod2 = nvm_deserialize(bin, bin_size);
    ASSERT(mod2 != NULL, "Round-trip: deserialization succeeded");

    /* Step 4: Verify structural equality */
    ASSERT_EQ_INT(mod2->string_count, mod1->string_count, "String count matches");
    ASSERT_EQ_INT(mod2->function_count, mod1->function_count, "Function count matches");
    ASSERT_EQ_INT(mod2->code_size, mod1->code_size, "Code size matches");

    for (uint32_t i = 0; i < mod1->string_count; i++) {
        ASSERT_EQ_STR(nvm_get_string(mod2, i), nvm_get_string(mod1, i),
                       "String content matches");
    }

    ASSERT(memcmp(mod2->code, mod1->code, mod1->code_size) == 0,
           "Bytecode content matches");

    /* Step 5: Disassemble */
    char *disasm = disasm_module(mod2);
    ASSERT(disasm != NULL, "Round-trip: disassembly succeeded");
    ASSERT(strstr(disasm, "PUSH_I64") != NULL, "Disasm contains PUSH_I64");
    ASSERT(strstr(disasm, "PRINT") != NULL, "Disasm contains PRINT");
    ASSERT(strstr(disasm, "HALT") != NULL, "Disasm contains HALT");
    ASSERT(strstr(disasm, "Hello, NanoISA!") != NULL, "Disasm contains string");

    free(disasm);
    free(bin);
    nvm_module_free(mod1);
    nvm_module_free(mod2);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    printf("=== NanoISA Test Suite ===\n\n");

    printf("[ISA]\n");
    RUN_TEST(test_tag_names);
    RUN_TEST(test_instruction_info);
    RUN_TEST(test_opcode_by_name);
    RUN_TEST(test_encode_decode_no_operands);
    RUN_TEST(test_encode_decode_i64);
    RUN_TEST(test_encode_decode_i64_large);
    RUN_TEST(test_encode_decode_f64);
    RUN_TEST(test_encode_decode_u8);
    RUN_TEST(test_encode_decode_u16);
    RUN_TEST(test_encode_decode_u32);
    RUN_TEST(test_encode_decode_i32);
    RUN_TEST(test_encode_decode_multi_operand);
    RUN_TEST(test_decode_truncated);
    RUN_TEST(test_decode_invalid_opcode);
    RUN_TEST(test_encode_all_categories);

    printf("\n[NVM Format]\n");
    RUN_TEST(test_crc32);
    RUN_TEST(test_module_new_free);
    RUN_TEST(test_string_pool);
    RUN_TEST(test_function_table);
    RUN_TEST(test_code_append);
    RUN_TEST(test_serialize_deserialize);
    RUN_TEST(test_validate_header);
    RUN_TEST(test_corrupt_checksum);

    printf("\n[Assembler]\n");
    RUN_TEST(test_asm_simple_program);
    RUN_TEST(test_asm_labels_and_jumps);
    RUN_TEST(test_asm_all_operand_types);
    RUN_TEST(test_asm_error_unknown_opcode);
    RUN_TEST(test_asm_error_undefined_label);
    RUN_TEST(test_asm_error_missing_end);
    RUN_TEST(test_asm_error_instruction_outside_function);
    RUN_TEST(test_asm_comments_and_whitespace);
    RUN_TEST(test_asm_string_escapes);
    RUN_TEST(test_asm_multiple_functions);

    printf("\n[Disassembler]\n");
    RUN_TEST(test_disasm_basic);
    RUN_TEST(test_disasm_labels);

    printf("\n[Round-Trip]\n");
    RUN_TEST(test_roundtrip_assemble_serialize_deserialize_disassemble);

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}
