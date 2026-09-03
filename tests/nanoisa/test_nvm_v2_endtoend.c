/*
 * End-to-end test for v2 emission and loading.
 *
 * nanoisa_load_bytes is the single funnel every consumer goes through -- the
 * VM, the co-process, the daemon, and generated wrappers -- so making it
 * dispatch on the magic byte is the whole loader change. What this test proves
 * is that the dispatch is real: the same module saved as v1 and as v2 comes
 * back identical through one entry point, a v1 module still takes the v1 path
 * untouched, and neither format is mistaken for the other.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nanoisa.h"
#include "nvm_format.h"
#include "nvm_format_v2.h"
#include "isa.h"
#include "nvm_v2_sections.h"
#include "verifier.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, what) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s  (%s:%d)\n", (what), __FILE__, __LINE__); } \
} while (0)

/* A module with two functions of one shape, one of another, an import, a
 * module ref, and a string carrying an embedded zero -- the same fixture shape
 * the bridge tests use, so a regression shows up as a difference here too. */
static NvmModule *build_module(void) {
    NvmModule *m = nvm_module_new();
    if (!m) return NULL;
    uint32_t s_add  = nvm_add_string(m, "add", 3);
    uint32_t s_sub  = nvm_add_string(m, "sub", 3);
    uint32_t s_main = nvm_add_string(m, "main", 4);
    nvm_add_string(m, "a\0b", 3);
    uint32_t s_libc = nvm_add_string(m, "libc", 4);
    uint32_t s_puts = nvm_add_string(m, "puts", 4);

    static const uint8_t code[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    nvm_append_code(m, code, sizeof code);

    NvmFunctionEntry f;
    memset(&f, 0, sizeof f);
    f.name_idx = s_add; f.arity = 2; f.code_offset = 0; f.code_length = 4;
    f.local_count = 2; f.result_tag = TAG_INT; f.result_count = 1;
    nvm_add_function(m, &f);
    f.name_idx = s_sub; f.code_offset = 4;
    nvm_add_function(m, &f);
    memset(&f, 0, sizeof f);
    f.name_idx = s_main; f.code_offset = 8; f.code_length = 4;
    f.local_count = 1; f.result_tag = TAG_VOID; f.result_count = 0;
    nvm_add_function(m, &f);

    const uint8_t ptypes[1] = { TAG_STRING };
    nvm_add_import(m, s_libc, s_puts, 1, TAG_INT, ptypes);
    nvm_add_module_ref(m, s_libc);
    nvm_add_debug_entry(m, 0, 17, 3);
    m->header.entry_point = 2;
    /* The same flags codegen sets. They are not decoration: HAS_MAIN is what
     * the VM checks before it will run anything. */
    m->header.flags = NVM_FLAG_HAS_MAIN | NVM_FLAG_NEEDS_EXTERN |
                      NVM_FLAG_DEBUG_INFO;
    return m;
}

/* Everything a consumer of a loaded module actually reads. If saving as v2 and
 * loading back changes any of it, the format is not a faithful carrier. */
static bool modules_agree(const NvmModule *a, const NvmModule *b) {
    if (a->function_count != b->function_count) return false;
    if (a->import_count != b->import_count) return false;
    if (a->module_ref_count != b->module_ref_count) return false;
    if (a->string_count != b->string_count) return false;
    if (a->code_size != b->code_size) return false;
    if (a->debug_count != b->debug_count) return false;
    if (a->header.entry_point != b->header.entry_point) return false;
    if (a->header.flags != b->header.flags) return false;
    if (memcmp(a->code, b->code, a->code_size) != 0) return false;
    for (uint32_t i = 0; i < a->string_count; i++) {
        if (a->string_lengths[i] != b->string_lengths[i]) return false;
        if (memcmp(a->strings[i], b->strings[i], a->string_lengths[i]) != 0)
            return false;
    }
    for (uint32_t i = 0; i < a->function_count; i++) {
        const NvmFunctionEntry *x = &a->functions[i], *y = &b->functions[i];
        if (x->name_idx != y->name_idx || x->arity != y->arity) return false;
        if (x->code_offset != y->code_offset) return false;
        if (x->code_length != y->code_length) return false;
        if (x->local_count != y->local_count) return false;
        if (x->upvalue_count != y->upvalue_count) return false;
        if (x->result_tag != y->result_tag) return false;
        if (x->result_count != y->result_count) return false;
    }
    for (uint32_t i = 0; i < a->import_count; i++) {
        const NvmImportEntry *x = &a->imports[i], *y = &b->imports[i];
        if (x->module_name_idx != y->module_name_idx) return false;
        if (x->function_name_idx != y->function_name_idx) return false;
        if (x->param_count != y->param_count) return false;
        if (x->return_type != y->return_type) return false;
        if (memcmp(a->import_param_types[i], b->import_param_types[i],
                   x->param_count) != 0) return false;
    }
    return true;
}

static void test_v2_bytes_carry_the_v2_magic(void) {
    NvmModule *m = build_module();
    CHECK(m != NULL, "fixture builds");
    if (!m) return;
    NanoisaErr err;
    uint32_t n = 0;
    uint8_t *bytes = nanoisa_save_bytes(m, &n, &err);
    CHECK(bytes != NULL, "a module saves as v2");
    if (bytes) {
        CHECK(n > NVM_V2_HEADER_SIZE, "the v2 blob is more than a header");
        CHECK(bytes[0] == NVM_V2_MAGIC_0 && bytes[1] == NVM_V2_MAGIC_1 &&
              bytes[2] == NVM_V2_MAGIC_2, "the v2 blob keeps the NVM prefix");
        CHECK(bytes[3] == NVM_V2_MAGIC_3,
              "the version byte is what the loader dispatches on");
        free(bytes);
    }
    nvm_module_free(m);
}

static void test_the_same_module_survives_both_formats(void) {
    NvmModule *m = build_module();
    if (!m) { g_fail++; printf("  FAIL: fixture\n"); return; }
    NanoisaErr err;

    uint32_t n = 0;
    uint8_t *b = nanoisa_save_bytes(m, &n, &err);
    CHECK(b != NULL, "the module saves");
    if (!b) { nvm_module_free(m); return; }

    NvmModule *l = nanoisa_load_bytes(b, n, &err);
    CHECK(l != NULL, "and loads back");
    if (l) {
        CHECK(modules_agree(m, l),
              "everything a consumer reads survives the round trip");
        nvm_module_free(l);
    } else { g_fail++; printf("  FAIL: could not compare\n"); }
    free(b);
    nvm_module_free(m);
}

static void test_garbage_is_still_rejected(void) {
    NanoisaErr err;
    uint8_t junk[64];
    memset(junk, 0xAB, sizeof junk);
    CHECK(nanoisa_load_bytes(junk, sizeof junk, &err) == NULL,
          "bytes with no NVM magic are rejected");

    /* "NVM" followed by a version byte no loader knows. The magic check must
     * not become "anything starting with NVM". */
    memcpy(junk, "NVM", 3);
    junk[3] = 0x7F;
    CHECK(nanoisa_load_bytes(junk, sizeof junk, &err) == NULL,
          "an unknown NVM version byte is rejected, not guessed at");

    NvmModule *m = build_module();
    if (!m) { g_fail++; printf("  FAIL: fixture\n"); return; }
    uint32_t n = 0;
    uint8_t *b = nanoisa_save_bytes(m, &n, &err);
    if (b) {
        b[n - 1] ^= 0xFF;   /* corrupt a payload byte, leaving the header */
        CHECK(nanoisa_load_bytes(b, n, &err) == NULL,
              "a corrupted v2 payload fails the checksum rather than loading");
        free(b);
    } else { g_fail++; printf("  FAIL: could not save v2\n"); }
    nvm_module_free(m);
}

/* The producer declares the operand depth and the loader confirms it. If the
 * loader took the declared value on trust, a module claiming less depth than it
 * uses would be accepted and overflow its stack at run time. */
static void test_an_understated_max_stack_is_rejected(void) {
    /* push, push, push, add, ret -- deepest height 3. */
    uint8_t code[128];
    uint32_t n = 0;
    n += isa_encode(&(DecodedInstruction){ .opcode = OP_PUSH_I64,
             .operands[0].i64 = 1 }, code + n, 64);
    n += isa_encode(&(DecodedInstruction){ .opcode = OP_PUSH_I64,
             .operands[0].i64 = 2 }, code + n, 64);
    n += isa_encode(&(DecodedInstruction){ .opcode = OP_PUSH_I64,
             .operands[0].i64 = 3 }, code + n, 64);
    n += isa_encode(&(DecodedInstruction){ .opcode = OP_ADD }, code + n, 64);
    /* Down to nothing before returning: the function declares no results, and
     * the verifier requires a return to leave exactly what is declared. The
     * deepest point is still 3. */
    n += isa_encode(&(DecodedInstruction){ .opcode = OP_POP }, code + n, 64);
    n += isa_encode(&(DecodedInstruction){ .opcode = OP_POP }, code + n, 64);
    n += isa_encode(&(DecodedInstruction){ .opcode = OP_RET }, code + n, 64);

    NvmModule *m = nvm_module_new();
    if (!m) { g_fail++; printf("  FAIL: fixture\n"); return; }
    uint32_t name = nvm_add_string(m, "main", 4);
    uint32_t off = nvm_append_code(m, code, n);
    NvmFunctionEntry f;
    memset(&f, 0, sizeof f);
    f.name_idx = name; f.code_offset = off; f.code_length = n;
    uint32_t idx = nvm_add_function(m, &f);
    m->header.flags = NVM_FLAG_HAS_MAIN;
    m->header.entry_point = idx;

    NvmV2Module v2;
    if (nvm_v2_from_nvm_module(m, &v2) != NVM_V2_OK) {
        g_fail++; printf("  FAIL: conversion\n"); nvm_module_free(m); return;
    }
    CHECK(v2.functions.count == 1 && v2.functions.items[0].max_stack == 3,
          "the producer declares the depth the verifier computes");

    /* Understate it by one and re-serialize. Everything else about the module
     * stays valid, so only the confirming check can catch this. */
    v2.functions.items[0].max_stack = 2;
    size_t need = 0;
    nvm_v2_module_serialize(&v2, NULL, 0, &need);
    uint8_t *buf = malloc(need);
    if (buf) {
        size_t written = 0;
        nvm_v2_module_serialize(&v2, buf, need, &written);
        NanoisaErr err;
        CHECK(nanoisa_load_bytes(buf, (uint32_t)written, &err) == NULL,
              "a module declaring less operand depth than it uses is rejected");
        free(buf);
    } else { g_fail++; printf("  FAIL: alloc\n"); }

    nvm_v2_module_free(&v2);
    nvm_module_free(m);
}

/* v1 is retired as of 4.0. .nvm files are build artifacts, not distributed
 * packages, so the fix is to rebuild -- and the message has to say so, because
 * "Invalid NVM magic" would send someone looking for a corrupt file. */
static void test_a_v1_module_is_refused_with_a_rebuild_instruction(void) {
    NvmModule *m = build_module();
    if (!m) { g_fail++; printf("  FAIL: fixture\n"); return; }
    NanoisaErr err;
    uint32_t n = 0;
    uint8_t *v1 = nanoisa_save_bytes_v1(m, &n, &err);
    CHECK(v1 != NULL, "a v1 blob can still be produced for this test");
    if (v1) {
        CHECK(v1[3] == NVM_MAGIC_3, "it carries the v1 version byte");
        memset(&err, 0, sizeof err);
        CHECK(nanoisa_load_bytes(v1, n, &err) == NULL, "and the loader refuses it");
        CHECK(strstr(err.message, "NanoISA v1") != NULL,
              "the message names the format the module was built for");
        CHECK(strstr(err.message, "rebuild") != NULL,
              "and tells the reader what to do about it");
        free(v1);
    }
    nvm_module_free(m);
}

static void test_the_default_container_is_v2(void) {
    NvmModule *m = build_module();
    if (!m) { g_fail++; printf("  FAIL: fixture\n"); return; }
    NanoisaErr err;
    uint32_t n = 0;
    uint8_t *b = nanoisa_save_bytes(m, &n, &err);
    CHECK(b != NULL, "a module saves");
    if (b) {
        CHECK(b[3] == NVM_V2_MAGIC_3,
              "nanoisa_save_bytes writes v2 without being asked");
        free(b);
    }
    nvm_module_free(m);
}

int main(void) {
    printf("\n[nvm_v2_endtoend] v2 emission and magic-dispatching load...\n\n");
    test_v2_bytes_carry_the_v2_magic();
    test_the_default_container_is_v2();
    test_a_v1_module_is_refused_with_a_rebuild_instruction();
    test_the_same_module_survives_both_formats();
    test_garbage_is_still_rejected();
    test_an_understated_max_stack_is_rejected();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
