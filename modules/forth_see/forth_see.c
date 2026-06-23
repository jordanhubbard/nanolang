/*
 * forth_see.c — NanoISA disassembler for the Forth interpreter's SEE word.
 *
 * Reads a compiled .nvm bytecode file, locates the exec_builtin function,
 * finds the dispatch branch for the requested Forth word, and returns a
 * human-readable NanoISA listing of its implementation.
 *
 * For user-defined words the caller handles the Forth-level display;
 * this module provides the ISA-level view for both built-ins and user words
 * (user words are listed as "interpreted — no compiled body").
 *
 * NVM format (little-endian):
 *   [Header 32 bytes]
 *   [Section directory: 12 * section_count]
 *   [Sections: CODE, STRINGS, FUNCTIONS, ...]
 */

#define _GNU_SOURCE
#include "forth_see.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── NVM constants (mirrored from src/nanoisa) ─────────────────────────── */

#define NVM_MAGIC_0 'N'
#define NVM_MAGIC_1 'V'
#define NVM_MAGIC_2 'M'
#define NVM_MAGIC_3 0x01

#define NVM_SECTION_CODE      0x0001
#define NVM_SECTION_STRINGS   0x0002
#define NVM_SECTION_FUNCTIONS 0x0003

/* Opcodes we care about */
#define OP_NOP          0x00
#define OP_PUSH_I64     0x01
#define OP_PUSH_F64     0x02
#define OP_PUSH_BOOL    0x03
#define OP_PUSH_STR     0x04
#define OP_PUSH_VOID    0x05
#define OP_PUSH_U8      0x06
#define OP_DUP_V        0x07  /* VM-level DUP (not Forth DUP) */
#define OP_POP          0x08
#define OP_SWAP         0x09
#define OP_ROT3         0x0A
#define OP_LOAD_LOCAL   0x10
#define OP_STORE_LOCAL  0x11
#define OP_LOAD_GLOBAL  0x12
#define OP_STORE_GLOBAL 0x13
#define OP_LOAD_UPVALUE 0x14
#define OP_STORE_UPVALUE 0x15
#define OP_ADD          0x20
#define OP_SUB          0x21
#define OP_MUL          0x22
#define OP_DIV          0x23
#define OP_MOD          0x24
#define OP_NEG          0x25
#define OP_EQ           0x28
#define OP_NE           0x29
#define OP_LT           0x2A
#define OP_LE           0x2B
#define OP_GT           0x2C
#define OP_GE           0x2D
#define OP_AND          0x30
#define OP_OR           0x31
#define OP_NOT          0x32
#define OP_JMP          0x38
#define OP_JMP_TRUE     0x39
#define OP_JMP_FALSE    0x3A
#define OP_CALL         0x3B
#define OP_CALL_INDIRECT 0x3C
#define OP_RET          0x3D
#define OP_CALL_EXTERN  0x3E
#define OP_CALL_MODULE  0x3F
#define OP_STR_LEN      0x40
#define OP_STR_CONCAT   0x41
#define OP_STR_SUBSTR   0x42
#define OP_STR_CONTAINS 0x43
#define OP_STR_EQ       0x44
#define OP_STR_CHAR_AT  0x45
#define OP_STR_FROM_INT 0x46
#define OP_STR_FROM_FLOAT 0x47
#define OP_ARR_NEW      0x50
#define OP_ARR_PUSH     0x51
#define OP_ARR_POP      0x52
#define OP_ARR_GET      0x53
#define OP_ARR_SET      0x54
#define OP_ARR_LEN      0x55
#define OP_CAST_INT     0x88
#define OP_CAST_FLOAT   0x89
#define OP_CAST_BOOL    0x8A
#define OP_CAST_STRING  0x8B
#define OP_PRINT        0xA0
#define OP_ASSERT       0xA1
#define OP_DEBUG_LINE   0xA2
#define OP_HALT         0xA3
#define OP_PRINTLN      0xA4

/* ── Minimal in-memory NVM representation ─────────────────────────────── */

typedef struct {
    uint32_t name_idx;
    uint32_t code_offset;
    uint32_t code_length;
    uint16_t local_count;
} NvmFuncEntry;

typedef struct {
    uint32_t function_name_idx;
} NvmImportEntry2;

typedef struct {
    char          **strings;
    uint32_t        string_count;
    NvmFuncEntry   *functions;
    uint32_t        function_count;
    NvmImportEntry2 *imports;
    uint32_t        import_count;
    uint8_t        *code;
    uint32_t        code_size;
} NvmMinimal;

/* ── Little-endian readers ─────────────────────────────────────────────── */

static uint16_t ru16(const uint8_t *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}
static uint32_t ru32(const uint8_t *p) {
    return p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static int32_t ri32(const uint8_t *p) { return (int32_t)ru32(p); }
static int64_t ri64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)p[i] << (8*i));
    return (int64_t)v;
}

/* ── NVM file loader ───────────────────────────────────────────────────── */

static NvmMinimal *nvm_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if ((long)fread(buf, 1, (size_t)sz, f) != sz) { free(buf); fclose(f); return NULL; }
    fclose(f);

    if (sz < 32 || buf[0]!=NVM_MAGIC_0 || buf[1]!=NVM_MAGIC_1 ||
        buf[2]!=NVM_MAGIC_2 || buf[3]!=NVM_MAGIC_3) {
        free(buf); return NULL;
    }

    uint32_t section_count = ru32(buf + 16);
    NvmMinimal *mod = (NvmMinimal *)calloc(1, sizeof(NvmMinimal));

    /* Section directory starts at byte 32 */
    for (uint32_t i = 0; i < section_count; i++) {
        uint32_t base = 32 + i * 12;
        if (base + 12 > (uint32_t)sz) break;
        uint32_t type   = ru32(buf + base);
        uint32_t offset = ru32(buf + base + 4);
        uint32_t size   = ru32(buf + base + 8);
        if (offset + size > (uint32_t)sz) continue;

        if (type == NVM_SECTION_CODE) {
            mod->code = (uint8_t *)malloc(size);
            if (mod->code) { memcpy(mod->code, buf + offset, size); mod->code_size = size; }
        } else if (type == NVM_SECTION_STRINGS) {
            /* String pool: [u32 len][bytes]... */
            uint32_t pos = 0;
            uint32_t cap = 64;
            mod->strings = (char **)malloc(cap * sizeof(char *));
            while (pos + 4 <= size && mod->string_count < 8192) {
                uint32_t slen = ru32(buf + offset + pos); pos += 4;
                if (pos + slen > size) break;
                if (mod->string_count >= cap) {
                    cap *= 2;
                    mod->strings = (char **)realloc(mod->strings, cap * sizeof(char *));
                }
                char *s = (char *)malloc(slen + 1);
                memcpy(s, buf + offset + pos, slen);
                s[slen] = '\0';
                mod->strings[mod->string_count++] = s;
                pos += slen;
            }
        } else if (type == NVM_SECTION_FUNCTIONS) {
            /* Function entries: 18 bytes each */
            uint32_t n = size / 18;
            mod->functions = (NvmFuncEntry *)malloc(n * sizeof(NvmFuncEntry));
            for (uint32_t j = 0; j < n; j++) {
                /* On-disk layout (18 bytes, packed, little-endian):
                 *  +0  u32  name_idx
                 *  +4  u16  arity
                 *  +6  u32  code_offset   ← NOT +8
                 *  +10 u32  code_length
                 *  +14 u16  local_count
                 *  +16 u16  upvalue_count */
                uint32_t base2 = offset + j * 18;
                mod->functions[j].name_idx    = ru32(buf + base2);
                mod->functions[j].code_offset = ru32(buf + base2 + 6);
                mod->functions[j].code_length = ru32(buf + base2 + 10);
                mod->functions[j].local_count = ru16(buf + base2 + 14);
            }
            mod->function_count = n;
        } else if (type == 0x0008) { /* IMPORTS section */
            /* Import entries: base size 11 bytes + param_count bytes */
            uint32_t pos = 0;
            uint32_t cap = 64;
            mod->imports = (NvmImportEntry2 *)malloc(cap * sizeof(NvmImportEntry2));
            while (pos + 11 <= size) {
                if (mod->import_count >= cap) {
                    cap *= 2;
                    mod->imports = (NvmImportEntry2 *)realloc(mod->imports, cap * sizeof(NvmImportEntry2));
                }
                mod->imports[mod->import_count].function_name_idx = ru32(buf + offset + pos + 4);
                uint16_t pc = ru16(buf + offset + pos + 9);
                mod->import_count++;
                pos += 11 + pc;
            }
        }
    }

    free(buf);
    return mod;
}

static void nvm_free(NvmMinimal *mod) {
    if (!mod) return;
    for (uint32_t i = 0; i < mod->string_count; i++) free(mod->strings[i]);
    free(mod->strings);
    free(mod->functions);
    free(mod->imports);
    free(mod->code);
    free(mod);
}

/* ── Instruction decoder ───────────────────────────────────────────────── */

typedef struct {
    uint8_t  op;
    uint32_t size;    /* bytes consumed */
    int64_t  imm;     /* immediate (i64/i32/u32/u16/u8) */
} Instr;

static Instr decode_instr(const uint8_t *p, uint32_t avail) {
    Instr r = {0};
    if (avail < 1) return r;
    r.op = p[0];
    switch (p[0]) {
    case OP_NOP: case OP_DUP_V: case OP_POP: case OP_SWAP: case OP_ROT3:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
    case OP_EQ:  case OP_NE:  case OP_LT:  case OP_LE:  case OP_GT:  case OP_GE:
    case OP_AND: case OP_OR:  case OP_NOT: case OP_RET: case OP_CALL_INDIRECT:
    case OP_STR_LEN: case OP_STR_CONCAT: case OP_STR_CONTAINS: case OP_STR_EQ:
    case OP_STR_CHAR_AT: case OP_STR_FROM_INT: case OP_STR_FROM_FLOAT:
    case OP_ARR_NEW: case OP_ARR_PUSH: case OP_ARR_POP: case OP_ARR_GET:
    case OP_ARR_SET: case OP_ARR_LEN:
    case OP_CAST_INT: case OP_CAST_FLOAT: case OP_CAST_BOOL: case OP_CAST_STRING:
    case OP_PRINT: case OP_PRINTLN: case OP_ASSERT: case OP_HALT: case OP_PUSH_VOID:
        r.size = 1; break;
    case OP_PUSH_U8: case OP_PUSH_BOOL:
        if (avail >= 2) { r.imm = p[1]; r.size = 2; } break;
    case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
        if (avail >= 3) { r.imm = (int64_t)ru16(p+1); r.size = 3; } break;
    case OP_LOAD_GLOBAL: case OP_STORE_GLOBAL:
    case OP_PUSH_STR: case OP_CALL: case OP_CALL_EXTERN: case OP_DEBUG_LINE:
        if (avail >= 5) { r.imm = (int64_t)ru32(p+1); r.size = 5; } break;
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:
        if (avail >= 5) { r.imm = (int64_t)ri32(p+1); r.size = 5; } break;
    case OP_PUSH_I64:
        if (avail >= 9) { r.imm = ri64(p+1); r.size = 9; } break;
    case OP_PUSH_F64:
        if (avail >= 9) { r.size = 9; } break;
    case OP_CALL_MODULE:
        if (avail >= 9) { r.imm = (int64_t)ru32(p+1); r.size = 9; } break;
    case OP_LOAD_UPVALUE: case OP_STORE_UPVALUE:
        if (avail >= 5) { r.imm = (int64_t)ru16(p+1); r.size = 5; } break;
    case 0x63: /* OP_STRUCT_LITERAL */
        if (avail >= 7) { r.imm = (int64_t)ru32(p+1); r.size = 7; } break;
    default:
        r.size = 1; break;
    }
    if (r.size == 0) r.size = 1;
    return r;
}

/* ── Opcode name ───────────────────────────────────────────────────────── */

static const char *opname(uint8_t op) {
    switch (op) {
    case OP_NOP:         return "NOP";
    case OP_PUSH_I64:    return "PUSH_I64";
    case OP_PUSH_F64:    return "PUSH_F64";
    case OP_PUSH_BOOL:   return "PUSH_BOOL";
    case OP_PUSH_STR:    return "PUSH_STR";
    case OP_PUSH_VOID:   return "PUSH_VOID";
    case OP_PUSH_U8:     return "PUSH_U8";
    case OP_DUP_V:       return "DUP";
    case OP_POP:         return "POP";
    case OP_SWAP:        return "SWAP";
    case OP_ROT3:        return "ROT3";
    case OP_LOAD_LOCAL:  return "LOAD_LOCAL";
    case OP_STORE_LOCAL: return "STORE_LOCAL";
    case OP_LOAD_GLOBAL: return "LOAD_GLOBAL";
    case OP_STORE_GLOBAL:return "STORE_GLOBAL";
    case OP_ADD:         return "ADD";
    case OP_SUB:         return "SUB";
    case OP_MUL:         return "MUL";
    case OP_DIV:         return "DIV";
    case OP_MOD:         return "MOD";
    case OP_NEG:         return "NEG";
    case OP_EQ:          return "EQ";
    case OP_NE:          return "NE";
    case OP_LT:          return "LT";
    case OP_LE:          return "LE";
    case OP_GT:          return "GT";
    case OP_GE:          return "GE";
    case OP_AND:         return "AND";
    case OP_OR:          return "OR";
    case OP_NOT:         return "NOT";
    case OP_JMP:         return "JMP";
    case OP_JMP_TRUE:    return "JMP_TRUE";
    case OP_JMP_FALSE:   return "JMP_FALSE";
    case OP_CALL:        return "CALL";
    case OP_CALL_INDIRECT: return "CALL_INDIRECT";
    case OP_RET:         return "RET";
    case OP_CALL_EXTERN: return "CALL_EXTERN";
    case OP_CALL_MODULE: return "CALL_MODULE";
    case OP_STR_LEN:     return "STR_LEN";
    case OP_STR_CONCAT:  return "STR_CONCAT";
    case OP_STR_SUBSTR:  return "STR_SUBSTR";
    case OP_STR_CONTAINS:return "STR_CONTAINS";
    case OP_STR_EQ:      return "STR_EQ";
    case OP_STR_CHAR_AT: return "STR_CHAR_AT";
    case OP_STR_FROM_INT:return "STR_FROM_INT";
    case OP_CAST_INT:    return "CAST_INT";
    case OP_CAST_FLOAT:  return "CAST_FLOAT";
    case OP_CAST_BOOL:   return "CAST_BOOL";
    case OP_CAST_STRING: return "CAST_STRING";
    case OP_PRINT:       return "PRINT";
    case OP_PRINTLN:     return "PRINTLN";
    case OP_ASSERT:      return "ASSERT";
    case OP_HALT:        return "HALT";
    case OP_ARR_GET:     return "ARR_GET";
    case OP_ARR_SET:     return "ARR_SET";
    case OP_ARR_LEN:     return "ARR_LEN";
    case OP_ARR_PUSH:    return "ARR_PUSH";
    case OP_ARR_POP:     return "ARR_POP";
    default: { static char b[12]; snprintf(b,sizeof(b),"0x%02X",op); return b; }
    }
}

/* ── Pattern scan ──────────────────────────────────────────────────────── */

/* The dispatch pattern in exec_builtin for each Forth word:
 *   LOAD_LOCAL 6       (3 bytes)  ← load the 'word' parameter
 *   PUSH_STR  <idx>    (5 bytes)  ← push word-name constant
 *   EQ                 (1 byte)   ← compare
 *   JMP_FALSE <off>    (5 bytes)  ← skip if not this word
 *                                 ← implementation starts here
 *
 * Total pattern: 14 bytes.  The implementation block ends just before
 * the label that JMP_FALSE points to.
 */

typedef struct { uint32_t start; uint32_t end; } Block;

static Block find_word_block(const NvmMinimal *mod,
                              uint32_t func_off, uint32_t func_len,
                              uint32_t word_str_idx) {
    Block none = {0, 0};
    if (!mod->code) return none;
    const uint8_t *base = mod->code + func_off;
    uint32_t pos = 0;

    while (pos + 14 <= func_len) {
        /* Match: LOAD_LOCAL 6, PUSH_STR word_str_idx, EQ, JMP_FALSE offset */
        if (base[pos] == OP_LOAD_LOCAL &&
            ru16(base + pos + 1) == 6 &&
            base[pos + 3] == OP_PUSH_STR &&
            ru32(base + pos + 4) == word_str_idx &&
            base[pos + 8] == OP_EQ &&
            base[pos + 9] == OP_JMP_FALSE) {

            int32_t off = ri32(base + pos + 10);
            uint32_t skip_pos = (uint32_t)((int32_t)pos + 9 + off); /* JMP_FALSE is at pos+9 */

            /* Implementation block: from pos+14 to skip_pos */
            if (skip_pos > pos + 14 && skip_pos <= func_len) {
                Block b;
                b.start = func_off + pos + 14;
                b.end   = func_off + skip_pos;
                return b;
            }
        }

        /* Advance by one instruction */
        Instr instr = decode_instr(base + pos, func_len - pos);
        pos += instr.size;
    }
    return none;
}

/* ── Block disassembler ────────────────────────────────────────────────── */

/* Collect all jump targets in the block to generate labels */
static void collect_labels(const uint8_t *code, uint32_t start, uint32_t end,
                             uint32_t *labels, int *nlabels, int max_labels) {
    *nlabels = 0;
    for (uint32_t pos = start; pos < end; ) {
        Instr in = decode_instr(code + pos, end - pos);
        if (in.op == OP_JMP || in.op == OP_JMP_TRUE || in.op == OP_JMP_FALSE) {
            /* offset relative to start of this instruction */
            uint32_t target = (uint32_t)((int32_t)pos + (int32_t)in.imm);
            if (target >= start && target <= end) {
                int found = 0;
                for (int i = 0; i < *nlabels; i++)
                    if (labels[i] == target) { found = 1; break; }
                if (!found && *nlabels < max_labels)
                    labels[(*nlabels)++] = target;
            }
        }
        pos += in.size;
    }
}

static int label_idx(const uint32_t *labels, int nlabels, uint32_t pos) {
    for (int i = 0; i < nlabels; i++) if (labels[i] == pos) return i;
    return -1;
}

#define OUT_CAP (1 << 16)  /* 64 KB output buffer */

static char *disasm_block(const NvmMinimal *mod,
                           uint32_t start, uint32_t end) {
    if (!mod->code || start >= end) return NULL;

    char *out = (char *)malloc(OUT_CAP);
    if (!out) return NULL;
    int olen = 0;

#define EMIT(...) do { \
    olen += snprintf(out + olen, OUT_CAP - olen, __VA_ARGS__); \
    if (olen >= OUT_CAP - 256) goto done; \
} while(0)

    uint32_t labels[64];
    int nlabels = 0;
    collect_labels(mod->code, start, end, labels, &nlabels, 64);

    /* Sort labels for stable output */
    for (int i = 0; i < nlabels - 1; i++)
        for (int j = i+1; j < nlabels; j++)
            if (labels[j] < labels[i]) {
                uint32_t t = labels[i]; labels[i] = labels[j]; labels[j] = t;
            }

    for (uint32_t pos = start; pos < end; ) {
        /* Emit label if this position is a jump target */
        int li = label_idx(labels, nlabels, pos);
        if (li >= 0) EMIT(".L%d:\n", li);

        Instr in = decode_instr(mod->code + pos, end - pos);
        EMIT("  %-16s", opname(in.op));

        switch (in.op) {
        case OP_PUSH_I64:
            EMIT(" %-8lld", (long long)in.imm);
            break;
        case OP_PUSH_BOOL:
            EMIT(" %s", in.imm ? "true" : "false");
            break;
        case OP_PUSH_U8:
            EMIT(" %lld", (long long)in.imm);
            break;
        case OP_PUSH_STR: {
            const char *s = (in.imm < (int64_t)mod->string_count)
                            ? mod->strings[in.imm] : "?";
            /* Truncate long strings */
            if (strlen(s) > 32) EMIT(" \"%.*s...\"", 30, s);
            else                EMIT(" \"%s\"", s);
            break;
        }
        case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
            EMIT(" %lld", (long long)in.imm);
            break;
        case OP_LOAD_GLOBAL: case OP_STORE_GLOBAL:
            EMIT(" G%lld", (long long)in.imm);
            break;
        case OP_CALL: {
            const char *nm = "?";
            if ((uint32_t)in.imm < mod->function_count) {
                uint32_t ni = mod->functions[(uint32_t)in.imm].name_idx;
                if (ni < mod->string_count) nm = mod->strings[ni];
            }
            EMIT(" %s", nm);
            break;
        }
        case OP_CALL_EXTERN: {
            const char *nm = "?";
            if ((uint32_t)in.imm < mod->import_count) {
                uint32_t ni = mod->imports[(uint32_t)in.imm].function_name_idx;
                if (ni < mod->string_count) nm = mod->strings[ni];
            }
            EMIT(" %s", nm);
            break;
        }
        case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: {
            uint32_t target = (uint32_t)((int32_t)pos + (int32_t)in.imm);
            int tli = label_idx(labels, nlabels, target);
            if (tli >= 0) EMIT(" .L%d", tli);
            else          EMIT(" %+d", (int)in.imm);
            break;
        }
        default:
            break;
        }
        EMIT("\n");
        pos += in.size;
    }
done:
    return out;
#undef EMIT
}

/* ── Public API ────────────────────────────────────────────────────────── */

static char nl_forth_see_buf[1 << 16];  /* 64 KB persistent output buffer */

const char *nl_forth_see(const char *word_name, const char *nvm_path) {
    nl_forth_see_buf[0] = '\0';
    int blen = 0;

#define BAPPEND(...) blen += snprintf(nl_forth_see_buf + blen, \
    (int)sizeof(nl_forth_see_buf) - blen, __VA_ARGS__)

    NvmMinimal *mod = nvm_load(nvm_path);
    if (!mod) {
        BAPPEND("SEE: cannot load %s\n", nvm_path);
        return nl_forth_see_buf;
    }

    /* Find "exec_builtin" function */
    uint32_t eb_idx = (uint32_t)-1;
    for (uint32_t i = 0; i < mod->function_count; i++) {
        uint32_t ni = mod->functions[i].name_idx;
        if (ni < mod->string_count &&
            strcmp(mod->strings[ni], "exec_builtin") == 0) {
            eb_idx = i; break;
        }
    }
    if (eb_idx == (uint32_t)-1) {
        BAPPEND("SEE: exec_builtin not found in %s\n", nvm_path);
        nvm_free(mod); return nl_forth_see_buf;
    }

    /* Find word name in string pool */
    uint32_t word_str_idx = (uint32_t)-1;
    for (uint32_t i = 0; i < mod->string_count; i++) {
        if (mod->strings[i] && strcmp(mod->strings[i], word_name) == 0) {
            word_str_idx = i; break;
        }
    }
    if (word_str_idx == (uint32_t)-1) {
        BAPPEND("; %s: not a built-in word\n", word_name);
        nvm_free(mod); return nl_forth_see_buf;
    }

    /* Find the implementation block */
    uint32_t fo = mod->functions[eb_idx].code_offset;
    uint32_t fl = mod->functions[eb_idx].code_length;
    Block blk = find_word_block(mod, fo, fl, word_str_idx);
    if (blk.start == 0 && blk.end == 0) {
        BAPPEND("; %s: control word — implemented in exec_tokens (not exec_builtin)\n",
                word_name);
        nvm_free(mod); return nl_forth_see_buf;
    }

    /* Header */
    BAPPEND("; ISA implementation of Forth word: %s\n", word_name);
    BAPPEND("; NanoISA block: bytes 0x%04X–0x%04X (%u bytes)\n",
            blk.start, blk.end, blk.end - blk.start);
    BAPPEND(";\n");

    /* Disassemble */
    char *dis = disasm_block(mod, blk.start, blk.end);
    if (dis) {
        int remaining = (int)sizeof(nl_forth_see_buf) - blen - 1;
        if (remaining > 0)
            strncat(nl_forth_see_buf + blen, dis, (size_t)remaining);
        free(dis);
    }

    nvm_free(mod);
    return nl_forth_see_buf;

#undef BAPPEND
}
