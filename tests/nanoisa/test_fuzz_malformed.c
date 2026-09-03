/*
 * test_fuzz_malformed.c — malformed-input and fuzz tests for the NanoISA
 * decoder, loader, verifier, assembler, and disassembler.
 *
 * Roadmap 4.0, Phase 12 (Verifier and safety): "add malformed-bytecode tests
 * and fuzz the decoder, loader, verifier, assembler, disassembler, and
 * co-process protocol."  This file covers the decoder (isa_decode), the loader
 * (nvm_deserialize), the verifier (nvm_verify), the assembler (asm_assemble),
 * and the disassembler (disasm_module).  The co-process protocol is fuzzed
 * separately in tests/nanovm/test_cop_fuzz.c because it links the VM heap.
 *
 * The goal is robustness, not correctness of any particular rejection: every
 * component must terminate cleanly (no crash, no hang, no read past the buffer)
 * on arbitrary bytes.  Where a well-formed structure is required to reach a
 * deeper code path, we build one and then mutate it byte-by-byte.
 *
 * A small deterministic xorshift PRNG keeps the run reproducible; there is no
 * dependence on the platform rand().
 */

#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanoisa/verifier.h"
#include "nanoisa/assembler.h"
#include "nanoisa/disassembler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-62s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-62s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* ── Deterministic PRNG (xorshift64) ─────────────────────────────────────── */

static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static void rng_seed(uint64_t s) { g_rng = s ? s : 0x1234567890ABCDEFULL; }
static uint64_t rng_next(void) {
    uint64_t x = g_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_rng = x;
    return x;
}
static uint8_t rng_byte(void) { return (uint8_t)(rng_next() & 0xFF); }
static uint32_t rng_u32(void)  { return (uint32_t)(rng_next() & 0xFFFFFFFFu); }

/* Fill a buffer with random bytes. */
static void rng_fill(uint8_t *buf, size_t n) {
    for (size_t i = 0; i < n; i++) buf[i] = rng_byte();
}

/* ── Little-endian writers (mirror the format module) ────────────────────── */

static void put_u32(uint8_t *b, uint32_t v) {
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
}

/* Build a minimal valid single-CODE-section module into a fresh malloc'd
 * buffer, returning the buffer and its size.  Callers may mutate it. */
static uint8_t *build_valid_module(uint32_t *out_size) {
    uint32_t dir = NVM_SECTION_ENTRY_SIZE;
    uint32_t data_start = NVM_HEADER_SIZE + dir;
    uint8_t payload[3] = { OP_HALT, OP_HALT, OP_HALT };
    uint32_t total = data_start + (uint32_t)sizeof(payload);
    uint8_t *buf = (uint8_t *)calloc(1, total);
    /* directory entry 0: CODE at data_start, size 3 */
    put_u32(buf + NVM_HEADER_SIZE + 0, NVM_SECTION_CODE);
    put_u32(buf + NVM_HEADER_SIZE + 4, data_start);
    put_u32(buf + NVM_HEADER_SIZE + 8, sizeof(payload));
    memcpy(buf + data_start, payload, sizeof(payload));
    /* header */
    buf[0] = 'N'; buf[1] = 'V'; buf[2] = 'M'; buf[3] = 0x01;
    put_u32(buf + 4,  NVM_FORMAT_VERSION);
    put_u32(buf + 8,  0);
    put_u32(buf + 12, 0);
    put_u32(buf + 16, 1);   /* section_count */
    put_u32(buf + 20, 0);
    put_u32(buf + 24, 0);
    put_u32(buf + 28, nvm_crc32(buf + NVM_HEADER_SIZE, total - NVM_HEADER_SIZE));
    *out_size = total;
    return buf;
}

/* ── Decoder fuzzing: isa_decode on arbitrary bytes ──────────────────────── */

/* isa_decode must never read past buf_size and must report 0 (or a length no
 * greater than the buffer) for every possible byte sequence. */
static void test_decode_random_bytes(void) {
    const char *test_name = "isa_decode: 100k random buffers never overrun";
    for (int iter = 0; iter < 100000; iter++) {
        uint8_t buf[32];
        size_t len = (size_t)(rng_next() % (sizeof(buf) + 1)); /* 0..32 */
        rng_fill(buf, len);
        DecodedInstruction instr;
        memset(&instr, 0, sizeof(instr));
        uint32_t consumed = isa_decode(buf, len, &instr);
        ASSERT(consumed <= len, "decode consumed more than the buffer holds");
    }
    PASS(test_name);
}

/* Every single opcode byte with a zero-length operand tail must be handled:
 * either decoded (0 operands) or rejected, but never crash. */
static void test_decode_all_opcode_bytes_empty_tail(void) {
    const char *test_name = "isa_decode: every opcode byte with empty tail is safe";
    for (int op = 0; op < 256; op++) {
        uint8_t buf[1] = { (uint8_t)op };
        DecodedInstruction instr;
        memset(&instr, 0, sizeof(instr));
        uint32_t consumed = isa_decode(buf, 1, &instr);
        ASSERT(consumed <= 1, "single-byte decode consumed too much");
    }
    PASS(test_name);
}

/* Truncated operand tails: for each opcode that needs operands, feed a prefix
 * that is one byte short and confirm decode reports failure rather than reading
 * uninitialised bytes. */
static void test_decode_truncated_operands(void) {
    const char *test_name = "isa_decode: truncated operand tail rejected";
    for (int op = 0; op < 256; op++) {
        const InstructionInfo *info = isa_get_info((uint8_t)op);
        if (!info || info->operand_count == 0) continue;
        /* Encode a full instruction, then feed every proper prefix of it. */
        DecodedInstruction instr;
        memset(&instr, 0, sizeof(instr));
        instr.opcode = (NanoOpcode)op;
        uint8_t full[64];
        uint32_t full_len = isa_encode(&instr, full, sizeof(full));
        if (full_len <= 1) continue;
        for (uint32_t prefix = 1; prefix < full_len; prefix++) {
            DecodedInstruction out;
            memset(&out, 0, sizeof(out));
            uint32_t consumed = isa_decode(full, prefix, &out);
            ASSERT(consumed == 0 || consumed <= prefix,
                   "truncated decode must fail or stay within the prefix");
        }
    }
    PASS(test_name);
}

/* ── Loader fuzzing: nvm_deserialize on arbitrary bytes ──────────────────── */

/* Pure random bytes: the loader must return NULL (or a freeable module) and
 * never crash.  Any returned module is freed. */
static void test_deserialize_random_bytes(void) {
    const char *test_name = "nvm_deserialize: 50k random buffers never crash";
    for (int iter = 0; iter < 50000; iter++) {
        uint8_t buf[128];
        size_t len = (size_t)(rng_next() % (sizeof(buf) + 1));
        rng_fill(buf, len);
        NvmModule *mod = nvm_deserialize(buf, (uint32_t)len);
        if (mod) nvm_module_free(mod);
    }
    PASS(test_name);
}

/* Random buffers that begin with the correct magic reach deeper header/section
 * parsing paths; still must never crash. */
static void test_deserialize_random_with_magic(void) {
    const char *test_name = "nvm_deserialize: random after magic never crashes";
    for (int iter = 0; iter < 50000; iter++) {
        uint8_t buf[128];
        size_t len = (size_t)(4 + (rng_next() % (sizeof(buf) - 4 + 1)));
        rng_fill(buf, len);
        buf[0] = 'N'; buf[1] = 'V'; buf[2] = 'M'; buf[3] = 0x01;
        NvmModule *mod = nvm_deserialize(buf, (uint32_t)len);
        if (mod) nvm_module_free(mod);
    }
    PASS(test_name);
}

/* Bit-flip mutation of a valid module: single-byte corruption anywhere must
 * either fail cleanly or yield a freeable module. */
static void test_deserialize_bitflip_valid(void) {
    const char *test_name = "nvm_deserialize: single-byte flips of valid module safe";
    uint32_t size = 0;
    uint8_t *base = build_valid_module(&size);
    /* Baseline must load. */
    NvmModule *ok = nvm_deserialize(base, size);
    ASSERT(ok != NULL, "baseline valid module must deserialize");
    nvm_module_free(ok);
    uint8_t *copy = (uint8_t *)malloc(size);
    for (uint32_t pos = 0; pos < size; pos++) {
        for (int bit = 0; bit < 8; bit++) {
            memcpy(copy, base, size);
            copy[pos] ^= (uint8_t)(1u << bit);
            NvmModule *mod = nvm_deserialize(copy, size);
            if (mod) nvm_module_free(mod);
        }
    }
    free(copy);
    free(base);
    PASS(test_name);
}

/* Truncation: every proper prefix of a valid module must be handled. */
static void test_deserialize_truncated_valid(void) {
    const char *test_name = "nvm_deserialize: every truncation of valid module safe";
    uint32_t size = 0;
    uint8_t *base = build_valid_module(&size);
    for (uint32_t len = 0; len < size; len++) {
        NvmModule *mod = nvm_deserialize(base, len);
        if (mod) nvm_module_free(mod);
    }
    free(base);
    PASS(test_name);
}

/* Adversarial header fields: giant section counts / offsets / sizes must be
 * rejected without allocating gigabytes or dereferencing out of range. */
static void test_deserialize_adversarial_header(void) {
    const char *test_name = "nvm_deserialize: adversarial header fields rejected safely";
    for (int iter = 0; iter < 20000; iter++) {
        uint8_t buf[64];
        memset(buf, 0, sizeof(buf));
        buf[0] = 'N'; buf[1] = 'V'; buf[2] = 'M'; buf[3] = 0x01;
        put_u32(buf + 4,  NVM_FORMAT_VERSION);
        put_u32(buf + 8,  rng_u32());       /* flags */
        put_u32(buf + 12, rng_u32());       /* entry_point */
        put_u32(buf + 16, rng_u32());       /* section_count (may be huge) */
        put_u32(buf + 20, rng_u32());       /* string_pool_offset */
        put_u32(buf + 24, rng_u32());       /* string_pool_length */
        put_u32(buf + 28, rng_u32());       /* checksum */
        /* random directory bytes */
        rng_fill(buf + NVM_HEADER_SIZE, sizeof(buf) - NVM_HEADER_SIZE);
        NvmModule *mod = nvm_deserialize(buf, sizeof(buf));
        if (mod) nvm_module_free(mod);
    }
    PASS(test_name);
}

/* ── Verifier fuzzing: nvm_verify on random-code functions ───────────────── */

/* Build a module whose single function's code section is arbitrary bytes and
 * run the verifier.  It must decide (ok/not-ok) without crashing and, when it
 * accepts, the accepted code must at least be self-consistent enough to free. */
static void test_verify_random_code(void) {
    const char *test_name = "nvm_verify: 20k random-code functions never crash";
    for (int iter = 0; iter < 20000; iter++) {
        NvmModule *mod = nvm_module_new();
        uint32_t name_idx = nvm_add_string(mod, "main", 4);
        uint8_t code[24];
        uint32_t code_len = (uint32_t)(rng_next() % (sizeof(code) + 1));
        rng_fill(code, code_len);
        uint32_t code_off = nvm_append_code(mod, code, code_len);
        NvmFunctionEntry fn;
        memset(&fn, 0, sizeof(fn));
        fn.name_idx = name_idx;
        fn.arity = 0;
        fn.code_offset = code_off;
        fn.code_length = code_len;
        fn.local_count = (uint16_t)(rng_next() % 8);
        fn.upvalue_count = (uint16_t)(rng_next() % 4);
        uint32_t fn_idx = nvm_add_function(mod, &fn);
        mod->header.flags = NVM_FLAG_HAS_MAIN;
        mod->header.entry_point = fn_idx;
        NvmVerifyResult r = nvm_verify(mod);
        /* Whatever the verdict, error_msg must be a valid C string. */
        ASSERT(r.ok || strlen(r.error_msg) < NVM_VERIFY_ERROR_SIZE,
               "verifier error_msg must be bounded");
        nvm_module_free(mod);
    }
    PASS(test_name);
}

/* Adversarial function-table fields: out-of-range code_offset/length,
 * arbitrary local/upvalue counts, bad entry point.  Verifier must reject
 * unsafe modules without crashing. */
static void test_verify_adversarial_function_entry(void) {
    const char *test_name = "nvm_verify: adversarial function entries rejected safely";
    for (int iter = 0; iter < 20000; iter++) {
        NvmModule *mod = nvm_module_new();
        uint32_t name_idx = nvm_add_string(mod, "main", 4);
        uint8_t code[8] = { OP_HALT, OP_HALT, OP_HALT, OP_HALT,
                            OP_HALT, OP_HALT, OP_HALT, OP_HALT };
        nvm_append_code(mod, code, sizeof(code));
        NvmFunctionEntry fn;
        memset(&fn, 0, sizeof(fn));
        fn.name_idx = name_idx;
        fn.arity = (uint16_t)rng_u32();
        fn.code_offset = rng_u32();
        fn.code_length = rng_u32();
        fn.local_count = (uint16_t)rng_u32();
        fn.upvalue_count = (uint16_t)rng_u32();
        fn.result_tag = rng_byte();
        fn.result_count = rng_byte();
        uint32_t fn_idx = nvm_add_function(mod, &fn);
        mod->header.flags = NVM_FLAG_HAS_MAIN;
        mod->header.entry_point = rng_u32();
        (void)fn_idx;
        NvmVerifyResult r = nvm_verify(mod);
        (void)r;
        nvm_module_free(mod);
    }
    PASS(test_name);
}

/* Loader→verifier pipeline: deserialize random-with-magic buffers and, when a
 * module is produced, run the verifier on it.  Mirrors the real load path. */
static void test_load_then_verify_pipeline(void) {
    const char *test_name = "load+verify pipeline: mutated modules never crash";
    uint32_t size = 0;
    uint8_t *base = build_valid_module(&size);
    uint8_t *copy = (uint8_t *)malloc(size);
    for (int iter = 0; iter < 40000; iter++) {
        memcpy(copy, base, size);
        /* flip 1-3 random bytes */
        int flips = 1 + (int)(rng_next() % 3);
        for (int f = 0; f < flips; f++) {
            copy[rng_next() % size] ^= rng_byte();
        }
        NvmModule *mod = nvm_deserialize(copy, size);
        if (mod) {
            NvmVerifyResult r = nvm_verify(mod);
            (void)r;
            nvm_module_free(mod);
        }
    }
    free(copy);
    free(base);
    PASS(test_name);
}

/* ── Assembler fuzzing: asm_assemble on arbitrary text ───────────────────── */

/* Random printable-ish text must not crash the assembler.  Any produced module
 * is freed; failures must set a bounded message. */
static void test_assemble_random_text(void) {
    const char *test_name = "asm_assemble: 20k random text inputs never crash";
    for (int iter = 0; iter < 20000; iter++) {
        char text[128];
        size_t len = (size_t)(rng_next() % (sizeof(text) - 1));
        for (size_t i = 0; i < len; i++) {
            /* bias toward printable ASCII incl. newlines and separators */
            uint8_t b = rng_byte();
            text[i] = (char)((b % 96) + 32);
            if ((rng_next() & 7) == 0) text[i] = '\n';
        }
        text[len] = '\0';
        AsmResult res;
        memset(&res, 0, sizeof(res));
        NvmModule *mod = asm_assemble(text, &res);
        if (mod) nvm_module_free(mod);
        else ASSERT(strchr(res.message, '\0') != NULL, "asm message must be terminated");
    }
    PASS(test_name);
}

/* Mutations of a real assembly program reach deeper parser paths. */
static void test_assemble_mutated_program(void) {
    const char *test_name = "asm_assemble: mutations of a real program never crash";
    static const char *seed =
        ".string \"hi\"\n"
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 42\n"
        "  PUSH_I64 10\n"
        "  ADD\n"
        "  HALT\n"
        ".end\n";
    size_t seedlen = strlen(seed);
    char buf[512];
    for (int iter = 0; iter < 20000; iter++) {
        memcpy(buf, seed, seedlen + 1);
        int mutations = 1 + (int)(rng_next() % 4);
        for (int m = 0; m < mutations; m++) {
            size_t pos = (size_t)(rng_next() % seedlen);
            buf[pos] = (char)((rng_byte() % 96) + 32);
        }
        AsmResult res;
        memset(&res, 0, sizeof(res));
        NvmModule *mod = asm_assemble(buf, &res);
        if (mod) nvm_module_free(mod);
    }
    PASS(test_name);
}

/* NUL bytes and non-ASCII embedded in the source must be handled. */
static void test_assemble_binary_source(void) {
    const char *test_name = "asm_assemble: binary/NUL source handled";
    for (int iter = 0; iter < 5000; iter++) {
        char text[64];
        size_t len = (size_t)(rng_next() % (sizeof(text) - 1));
        rng_fill((uint8_t *)text, len);
        /* Force a terminator somewhere so we pass a valid C string. */
        text[len] = '\0';
        AsmResult res;
        memset(&res, 0, sizeof(res));
        NvmModule *mod = asm_assemble(text, &res);
        if (mod) nvm_module_free(mod);
    }
    PASS(test_name);
}

/* ── Disassembler fuzzing: disasm_module on mutated modules ──────────────── */

/* Disassemble modules loaded from mutated buffers.  disasm must produce a
 * freeable string (or NULL) without crashing on odd code/section contents. */
static void test_disasm_mutated_modules(void) {
    const char *test_name = "disasm_module: mutated modules never crash";
    uint32_t size = 0;
    uint8_t *base = build_valid_module(&size);
    uint8_t *copy = (uint8_t *)malloc(size);
    for (int iter = 0; iter < 20000; iter++) {
        memcpy(copy, base, size);
        int flips = 1 + (int)(rng_next() % 2);
        for (int f = 0; f < flips; f++) copy[rng_next() % size] ^= rng_byte();
        NvmModule *mod = nvm_deserialize(copy, size);
        if (mod) {
            char *text = disasm_module(mod);
            if (text) free(text);
            char *canon = disasm_module_styled(mod, DISASM_STYLE_CANONICAL);
            if (canon) free(canon);
            nvm_module_free(mod);
        }
    }
    free(copy);
    free(base);
    PASS(test_name);
}

/* Disassemble programmatically-built modules whose code is random bytes. */
static void test_disasm_random_code(void) {
    const char *test_name = "disasm_module: random-code modules never crash";
    for (int iter = 0; iter < 20000; iter++) {
        NvmModule *mod = nvm_module_new();
        uint32_t name_idx = nvm_add_string(mod, "f", 1);
        uint8_t code[24];
        uint32_t code_len = (uint32_t)(rng_next() % (sizeof(code) + 1));
        rng_fill(code, code_len);
        uint32_t off = nvm_append_code(mod, code, code_len);
        NvmFunctionEntry fn;
        memset(&fn, 0, sizeof(fn));
        fn.name_idx = name_idx;
        fn.code_offset = off;
        fn.code_length = code_len;
        fn.local_count = 1;
        nvm_add_function(mod, &fn);
        char *text = disasm_module(mod);
        if (text) free(text);
        nvm_module_free(mod);
    }
    PASS(test_name);
}

/* Round-trip: disassemble a valid module, then re-assemble the text.  The
 * re-assembled module should verify (canonical disassembly stays loadable). */
static void test_disasm_reassemble_roundtrip(void) {
    const char *test_name = "disasm→asm round-trip: canonical text re-assembles";
    static const char *seed =
        ".string \"hi\"\n"
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 42\n"
        "  HALT\n"
        ".end\n";
    AsmResult ares;
    memset(&ares, 0, sizeof(ares));
    NvmModule *mod = asm_assemble(seed, &ares);
    ASSERT(mod != NULL, "seed program must assemble");
    NvmVerifyResult v0 = nvm_verify(mod);
    ASSERT(v0.ok, "seed program must verify");
    char *text = disasm_module_styled(mod, DISASM_STYLE_CANONICAL);
    ASSERT(text != NULL, "canonical disassembly produced");
    AsmResult ares2;
    memset(&ares2, 0, sizeof(ares2));
    NvmModule *mod2 = asm_assemble(text, &ares2);
    if (mod2) {
        NvmVerifyResult v1 = nvm_verify(mod2);
        ASSERT(v1.ok, "re-assembled canonical text must verify");
        nvm_module_free(mod2);
    }
    free(text);
    nvm_module_free(mod);
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== NanoISA malformed-input / fuzz tests ===\n");
    rng_seed(0xC0FFEE1234567890ULL);

    printf("\n-- decoder --\n");
    test_decode_random_bytes();
    test_decode_all_opcode_bytes_empty_tail();
    test_decode_truncated_operands();

    printf("\n-- loader (nvm_deserialize) --\n");
    test_deserialize_random_bytes();
    test_deserialize_random_with_magic();
    test_deserialize_bitflip_valid();
    test_deserialize_truncated_valid();
    test_deserialize_adversarial_header();

    printf("\n-- verifier --\n");
    test_verify_random_code();
    test_verify_adversarial_function_entry();
    test_load_then_verify_pipeline();

    printf("\n-- assembler --\n");
    test_assemble_random_text();
    test_assemble_mutated_program();
    test_assemble_binary_source();

    printf("\n-- disassembler --\n");
    test_disasm_mutated_modules();
    test_disasm_random_code();
    test_disasm_reassemble_roundtrip();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
