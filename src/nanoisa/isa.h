/*
 * NanoISA - Instruction Set Architecture for nanolang
 *
 * Stack machine with 5 registers (SP, FP, IP, R0/accumulator, R1/scratch).
 * Runtime-typed values (16 bytes: 1-byte tag + payload).
 * Variable-length instruction encoding (1-byte opcode + operands).
 * No undefined behavior.
 */

#ifndef NANOISA_ISA_H
#define NANOISA_ISA_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ========================================================================
 * Value Type Tags
 * Each value on the stack carries a 1-byte type tag.
 * ======================================================================== */

typedef enum {
    TAG_VOID     = 0x00,
    TAG_INT      = 0x01,  /* 64-bit signed integer */
    TAG_U8       = 0x02,  /* Unsigned byte */
    TAG_FLOAT    = 0x03,  /* 64-bit IEEE 754 double */
    TAG_BOOL     = 0x04,  /* true/false */
    TAG_STRING   = 0x05,  /* Heap ref, GC-managed, immutable */
    TAG_BSTRING  = 0x06,  /* Binary string with length */
    TAG_ARRAY    = 0x07,  /* Dynamic array, GC-managed */
    TAG_STRUCT   = 0x08,  /* Named struct with fields */
    TAG_ENUM     = 0x09,  /* Integer variant index */
    TAG_UNION    = 0x0A,  /* Tagged union */
    TAG_FUNCTION = 0x0B,  /* Direct function table reference */
    TAG_TUPLE    = 0x0C,  /* Fixed-size heterogeneous */
    TAG_HASHMAP  = 0x0D,  /* Key-value map */
    TAG_OPAQUE   = 0x0E,  /* RPC proxy ID to co-process handle */
    TAG_CLOSURE  = 0x0F,  /* Heap closure with captured values */

    TAG_COUNT    = 0x10   /* Number of valid tags (sentinel) */
} NanoValueTag;

/* ========================================================================
 * Opcodes
 * Variable-length encoding: 1-byte opcode + type-specific operands.
 * ======================================================================== */

typedef enum {
    /* Stack & Constants (0x00-0x0F) */
    OP_NOP          = 0x00,
    OP_PUSH_I64     = 0x01,  /* operand: i64 (8 bytes) */
    OP_PUSH_F64     = 0x02,  /* operand: f64 (8 bytes) */
    OP_PUSH_BOOL    = 0x03,  /* operand: u8 (1 byte) */
    OP_PUSH_STR     = 0x04,  /* operand: u32 string pool index */
    OP_PUSH_VOID    = 0x05,
    OP_PUSH_U8      = 0x06,  /* operand: u8 (1 byte) */
    OP_DUP          = 0x07,
    OP_POP          = 0x08,
    OP_SWAP         = 0x09,
    OP_ROT3         = 0x0A,

    /* Variable Access (0x10-0x1F) */
    OP_LOAD_LOCAL   = 0x10,  /* operand: u16 (FP-relative index) */
    OP_STORE_LOCAL  = 0x11,  /* operand: u16 */
    OP_LOAD_GLOBAL  = 0x12,  /* operand: u32 (global index) */
    OP_STORE_GLOBAL = 0x13,  /* operand: u32 */
    OP_LOAD_UPVALUE  = 0x14, /* operands: u16 depth, u16 index */
    OP_STORE_UPVALUE = 0x15, /* operands: u16 depth, u16 index */

    /* Arithmetic (0x20-0x27) */
    OP_ADD          = 0x20,  /* pop b, pop a, push a+b (also string concat) */
    OP_SUB          = 0x21,
    OP_MUL          = 0x22,
    OP_DIV          = 0x23,  /* div by zero = 0 */
    OP_MOD          = 0x24,
    OP_NEG          = 0x25,  /* unary negation */

    /* Comparison (0x28-0x2F) */
    OP_EQ           = 0x28,
    OP_NE           = 0x29,
    OP_LT           = 0x2A,
    OP_LE           = 0x2B,
    OP_GT           = 0x2C,
    OP_GE           = 0x2D,

    /* Logic (0x30-0x37) */
    OP_AND          = 0x30,
    OP_OR           = 0x31,
    OP_NOT          = 0x32,
    OP_TAIL_CALL    = 0x33,  /* operand: u32 function table index */

    /* Control Flow (0x38-0x3F) */
    OP_JMP          = 0x38,  /* operand: i32 relative offset */
    OP_JMP_TRUE     = 0x39,  /* operand: i32 relative offset */
    OP_JMP_FALSE    = 0x3A,  /* operand: i32 relative offset */
    OP_CALL         = 0x3B,  /* operand: u32 function table index */
    /* pop function value from stack; operands: u16 arity, u16 result count.
     * The shape is encoded because the callee is only known at run time, and
     * without it the stack effect is not statically determinable -- which is
     * what let stack verification stop at every indirect call (issue #212). */
    OP_CALL_INDIRECT = 0x3C,
    OP_RET          = 0x3D,
    OP_CALL_EXTERN  = 0x3E,  /* operand: u32 import table index (RPC) */
    OP_CALL_MODULE  = 0x3F,  /* operands: u32 module index, u32 function index */

    /* String Ops (0x40-0x4F) */
    OP_STR_LEN      = 0x40,
    OP_STR_CONCAT   = 0x41,
    OP_STR_SUBSTR   = 0x42,  /* pop len, pop start, pop str -> push substr */
    OP_STR_CONTAINS = 0x43,  /* pop needle, pop haystack -> push bool */
    OP_STR_EQ       = 0x44,
    OP_STR_CHAR_AT  = 0x45,  /* pop index, pop str -> push char (as string) */
    OP_STR_FROM_INT = 0x46,  /* pop int -> push string */
    OP_STR_FROM_FLOAT = 0x47, /* pop float -> push string */
    OP_STR_TRIM     = 0x48,  /* pop str -> push str with leading/trailing whitespace removed */
    OP_STR_TO_LOWER = 0x49,  /* pop str -> push lowercased str */
    OP_STR_TO_UPPER = 0x4A,  /* pop str -> push uppercased str */
    OP_STR_STARTS_WITH = 0x4B, /* pop prefix, pop str -> push bool */
    OP_STR_ENDS_WITH   = 0x4C, /* pop suffix, pop str -> push bool */
    OP_STR_SPLIT       = 0x4D, /* pop delim, pop str -> push array<string> */
    OP_STR_REPLACE     = 0x4E, /* pop new, pop old, pop str -> push str with all old replaced */

    /* Array Ops (0x50-0x5F) */
    OP_ARR_NEW      = 0x50,  /* operand: u8 element type tag */
    OP_ARR_PUSH     = 0x51,  /* pop value, pop array -> push array */
    OP_ARR_POP      = 0x52,  /* pop array -> push value, push array */
    OP_ARR_GET      = 0x53,  /* pop index, pop array -> push value */
    OP_ARR_SET      = 0x54,  /* pop value, pop index, pop array -> push array */
    OP_ARR_LEN      = 0x55,  /* pop array -> push int */
    OP_ARR_SLICE    = 0x56,  /* pop end, pop start, pop array -> push array */
    OP_ARR_REMOVE   = 0x57,  /* pop index, pop array -> push array */
    OP_ARR_LITERAL  = 0x58,  /* operands: u8 type tag, u16 count; pops count values */

    /* Struct Ops (0x60-0x67) */
    OP_STRUCT_NEW     = 0x60, /* operand: u32 struct def index */
    OP_STRUCT_GET     = 0x61, /* operand: u16 field index; pop struct -> push value */
    OP_STRUCT_SET     = 0x62, /* operand: u16 field index; pop value, pop struct -> push struct */
    OP_STRUCT_LITERAL = 0x63, /* operands: u32 def_idx, u16 field_count; pops field_count values */

    /* Union/Enum Ops (0x68-0x6F) */
    OP_UNION_CONSTRUCT = 0x68, /* operands: u32 def_idx, u16 variant, u16 fields */
    OP_UNION_TAG       = 0x69, /* pop union -> push int (variant index) */
    OP_UNION_FIELD     = 0x6A, /* operand: u16 field index; pop union -> push value */
    OP_MATCH_TAG       = 0x6B, /* operands: u16 variant, i32 jump_offset */
    OP_ENUM_VAL        = 0x6C, /* operands: u32 def_idx, u16 variant */

    /* Tuple Ops (0x70-0x77) */
    OP_TUPLE_NEW    = 0x70,  /* operand: u16 count; pops count values */
    OP_TUPLE_GET    = 0x71,  /* operand: u16 index; pop tuple -> push value */

    /* Hashmap Ops (0x78-0x7F) */
    OP_HM_NEW      = 0x78,  /* operands: u8 key type tag, u8 val type tag */
    OP_HM_GET      = 0x79,  /* pop key, pop map -> push value */
    OP_HM_SET      = 0x7A,  /* pop value, pop key, pop map -> push map */
    OP_HM_HAS      = 0x7B,  /* pop key, pop map -> push bool */
    OP_HM_DELETE   = 0x7C,  /* pop key, pop map -> push map */
    OP_HM_KEYS     = 0x7D,  /* pop map -> push array of keys */
    OP_HM_VALUES   = 0x7E,  /* pop map -> push array of values */
    OP_HM_LEN      = 0x7F,  /* pop map -> push int */

    /* GC/Memory (0x80-0x87) */
    OP_GC_RETAIN      = 0x80,
    OP_GC_RELEASE     = 0x81,

    /* Type Casts (0x88-0x8F) */
    OP_CAST_INT    = 0x88,   /* pop value -> push as int */
    OP_CAST_FLOAT  = 0x89,   /* pop value -> push as float */
    OP_CAST_BOOL   = 0x8A,   /* pop value -> push as bool */
    OP_CAST_STRING = 0x8B,   /* pop value -> push as string */
    OP_TYPE_CHECK  = 0x8C,   /* operand: u8 expected tag; pop value -> push bool */

    /* Closures (0x90-0x97) */
    OP_CLOSURE_NEW  = 0x90,  /* operands: u32 fn_idx, u16 capture_count */

    /* I/O & Debug (0xA0-0xAF) */
    OP_PRINT      = 0xA0,   /* pop value, print to stdout (no newline) */
    OP_ASSERT     = 0xA1,   /* pop bool, abort if false */
    OP_DEBUG_LINE = 0xA2,   /* operand: u32 source line number */
    OP_HALT       = 0xA3,   /* stop execution */
    OP_PRINTLN    = 0xA4,   /* pop value, print to stdout with newline */

    /* Opaque Proxy (0xB0-0xBF) */
    OP_OPAQUE_NULL  = 0xB0,  /* push null opaque proxy */
    OP_OPAQUE_VALID = 0xB1,  /* pop opaque -> push bool (is non-null) */
    OP_MEM_LOAD8    = 0xB2,  /* pop address -> push unsigned byte */
    OP_MEM_LOAD16   = 0xB3,  /* pop address -> push unsigned little-endian word */
    OP_MEM_LOAD32   = 0xB4,  /* pop address -> push unsigned little-endian dword */
    OP_MEM_LOAD64   = 0xB5,  /* pop address -> push little-endian cell */
    OP_MEM_STORE8   = 0xB6,  /* pop value, address */
    OP_MEM_STORE16  = 0xB7,
    OP_MEM_STORE32  = 0xB8,
    OP_MEM_STORE64  = 0xB9,

    /* Typed scalar operations (v2 migration) */
    OP_I64_ADD      = 0xC0,
    OP_I64_SUB      = 0xC1,
    OP_I64_MUL      = 0xC2,
    OP_I64_DIV_S    = 0xC3,
    OP_I64_REM_S    = 0xC4,
    OP_I64_NEG      = 0xC5,
    OP_F64_ADD      = 0xC8,
    OP_F64_SUB      = 0xC9,
    OP_F64_MUL      = 0xCA,
    OP_F64_DIV      = 0xCB,
    OP_F64_NEG      = 0xCC,

    OP_I64_EQ       = 0xD0,
    OP_I64_NE       = 0xD1,
    OP_I64_LT_S     = 0xD2,
    OP_I64_LE_S     = 0xD3,
    OP_I64_GT_S     = 0xD4,
    OP_I64_GE_S     = 0xD5,
    OP_F64_EQ       = 0xD8,
    OP_F64_NE       = 0xD9,
    OP_F64_LT       = 0xDA,
    OP_F64_LE       = 0xDB,
    OP_F64_GT       = 0xDC,
    OP_F64_GE       = 0xDD,
    OP_BOOL_AND     = 0xE0,
    OP_BOOL_OR      = 0xE1,
    OP_BOOL_NOT     = 0xE2,
    OP_I64_DIV_U    = 0xE3,
    OP_I64_REM_U    = 0xE4,
    OP_I64_LT_U     = 0xE5,
    OP_I64_LE_U     = 0xE6,
    OP_I64_GT_U     = 0xE7,
    OP_I64_GE_U     = 0xE8,
    OP_I64_AND      = 0xE9,
    OP_I64_OR       = 0xEA,
    OP_I64_XOR      = 0xEB,
    OP_I64_INVERT   = 0xEC,
    OP_I64_SHL      = 0xED,
    OP_I64_SHR_S    = 0xEE,
    OP_I64_SHR_U    = 0xEF,
    OP_I64_ADD_CARRY = 0xF0, /* a b carry -> low carry */
    OP_I64_SUB_BORROW = 0xF1, /* a b borrow -> low borrow */
    OP_I64_MUL_WIDE_S = 0xF2, /* a b -> low high */
    OP_I64_MUL_WIDE_U = 0xF3, /* a b -> low high */
    OP_PICK         = 0xF4,  /* operand: u16 depth */
    OP_ROLL         = 0xF5,  /* operand: u16 depth */
    OP_ARRAY_ADD    = 0xF6,
    OP_ARRAY_SUB    = 0xF7,
    OP_ARRAY_MUL    = 0xF8,
    OP_ARRAY_DIV    = 0xF9,
    OP_FUNCREF      = 0xFA,  /* operand: u32 function table index */
    OP_AGG_PACK     = 0xFB,  /* kind:u8 layout:u32 variant:u16 count:u16 */
    OP_AGG_GET      = 0xFC,  /* operand: u16 field/index */
    OP_AGG_SET      = 0xFD,  /* operand: u16 field/index */
    OP_AGG_TAG      = 0xFE,  /* pop aggregate -> push integer tag */

    /* 0xFF is deliberately NOT a primary opcode. It is the extension prefix:
     * when the decoder reads OP_EXTENSION_PREFIX, the next byte selects an
     * opcode from the extended plane. This keeps opcode values as pure
     * identifiers rather than an "instruction count". */
    OP_EXTENSION_PREFIX = 0xFF
} NanoOpcode;

/* Bounds and escape byte for the one-byte opcode space.
 *
 * The exclusive upper bound of the primary plane (identifiers 0x00..0xFE) and
 * the extension-prefix escape byte are defined in the generated schema header
 * as NANOISA_PRIMARY_OPCODE_LIMIT and NANOISA_EXTENSION_PREFIX. Both are range
 * / escape constants, not a running count of how many opcodes exist. A static
 * assertion below (after that header is included) keeps this enum in step. */

typedef enum {
    AGG_RECORD = 0,
    AGG_VARIANT = 1,
    AGG_TUPLE = 2
} NanoAggregateKind;

/* ========================================================================
 * Operand Types
 * ======================================================================== */

typedef enum {
    OPERAND_NONE,   /* No operand */
    OPERAND_U8,     /* 1-byte unsigned */
    OPERAND_U16,    /* 2-byte unsigned (little-endian) */
    OPERAND_U32,    /* 4-byte unsigned (little-endian) */
    OPERAND_I32,    /* 4-byte signed (little-endian) */
    OPERAND_I64,    /* 8-byte signed (little-endian) */
    OPERAND_F64     /* 8-byte IEEE 754 double (little-endian) */
} OperandType;

/* Maximum number of operands any instruction can have */
#define MAX_OPERANDS 4

/* ========================================================================
 * Instruction Metadata
 * ======================================================================== */

typedef struct {
    const char *name;          /* Mnemonic (e.g., "PUSH_I64") */
    uint8_t opcode;            /* Opcode byte */
    uint8_t operand_count;     /* Number of operands (0-4) */
    OperandType operands[MAX_OPERANDS]; /* Operand types */
    int8_t pop_count;          /* Fixed stack effect, or -1 if operand-dependent */
    int8_t push_count;
} InstructionInfo;

typedef struct {
    const char *name;
    uint8_t code;
} NanoisaSchemaTag;

typedef struct {
    const char *name;
    uint8_t opcode;
    uint8_t operand_count;
    OperandType operands[MAX_OPERANDS];
    int8_t pop_count;
    int8_t push_count;
} NanoisaSchemaOpcode;

typedef struct {
    const char *name;
    const char *meaning;       /* One comprehensible meaning per instruction */
    const char *ownership;
    const char *justification; /* Why this instruction belongs in the ISA rather
                                * than a runtime library reached through call.import */
    uint8_t operand_count;
    uint8_t pop_count;
    uint8_t push_count;
} NanoisaV2Family;

#include "generated_schema.h"

/* The extension prefix is exactly the byte reserved by the enum, and it sits
 * one past the last usable primary opcode. Opcode values are identifiers, so
 * the "limit" is an exclusive bound, never an instruction count. */
_Static_assert(NANOISA_EXTENSION_PREFIX == OP_EXTENSION_PREFIX,
               "extension prefix must match the reserved opcode byte");
_Static_assert(NANOISA_PRIMARY_OPCODE_LIMIT == NANOISA_EXTENSION_PREFIX,
               "primary opcode plane ends exactly at the extension prefix");
_Static_assert(OP_AGG_TAG < NANOISA_PRIMARY_OPCODE_LIMIT,
               "the last primary opcode must stay below the plane limit");

/* ========================================================================
 * Decoded Instruction (for disassembly / VM execution)
 * ======================================================================== */

typedef struct {
    uint8_t opcode;
    uint8_t operand_count;
    union {
        uint8_t  u8;
        uint16_t u16;
        uint32_t u32;
        int32_t  i32;
        int64_t  i64;
        double   f64;
    } operands[MAX_OPERANDS];
    OperandType operand_types[MAX_OPERANDS];
    uint32_t byte_length;      /* Total encoded length in bytes */
} DecodedInstruction;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Get metadata for an opcode. Returns NULL for invalid opcodes. */
const InstructionInfo *isa_get_info(uint8_t opcode);

/* Get the name of a value tag. Returns "UNKNOWN" for invalid tags. */
const char *isa_tag_name(uint8_t tag);

/* Encode a decoded instruction into a byte buffer.
 * Returns number of bytes written, or 0 on error.
 * buf must have at least ISA_MAX_INSTRUCTION_SIZE bytes available. */
uint32_t isa_encode(const DecodedInstruction *instr, uint8_t *buf, size_t buf_size);

/* Decode an instruction from a byte buffer.
 * Returns number of bytes consumed, or 0 on error. */
uint32_t isa_decode(const uint8_t *buf, size_t buf_size, DecodedInstruction *out);

/* Get the encoded size of an operand type in bytes. */
uint32_t isa_operand_size(OperandType type);

/* Maximum encoded instruction size (opcode + largest operand combination) */
#define ISA_MAX_INSTRUCTION_SIZE 32

/* Lookup opcode by mnemonic name. Returns -1 if not found. */
int isa_opcode_by_name(const char *name);

/* True when a code byte is the extension prefix rather than a primary opcode.
 * A decoder that sees this must read one more byte and resolve it through the
 * extended plane; it must not treat the prefix itself as an instruction. */
bool isa_is_extension_prefix(uint8_t opcode);

/* Metadata for an extended opcode (the byte that follows the prefix).
 * Returns NULL when no extended opcode with that code is defined. */
const InstructionInfo *isa_get_extended_info(uint8_t ext_opcode);

#endif /* NANOISA_ISA_H */
