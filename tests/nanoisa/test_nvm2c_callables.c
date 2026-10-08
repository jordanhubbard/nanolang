#include "assembler.h"
#include "isa.h"
#include "nvm2c.h"
#include <stdlib.h>
#include "nvm2c_callables.h"
#include <stdio.h>
#include <string.h>

static int passed, failed;
#define CHECK(value) do { if (value) ++passed; else { ++failed; \
    fprintf(stderr, "I failed %s at line %d\n", #value, __LINE__); } } while (0)

static NvmModule *assemble(const char *text, int verify) {
    AsmResult result = {0};
    NvmModule *module = verify ? asm_assemble(text, &result) : asm_assemble_unverified(text, &result);
    if (!module) fprintf(stderr, "I could not assemble line %u: %s\n", result.line, result.message);
    CHECK(module != NULL);
    return module;
}
static void targets(NvmModule *module, NvmCallableAnalysis *a, uint32_t function,
                    size_t ordinal, const uint32_t *expected, size_t count) {
    const NvmFunctionEntry *fn = &module->functions[function];
    for (uint32_t pc = 0; pc < fn->code_length;) {
        DecodedInstruction in;
        uint32_t size = isa_decode(module->code + fn->code_offset + pc, fn->code_length - pc, &in);
        CHECK(size != 0);
        if (!size) return;
        if (in.opcode == OP_CALL_INDIRECT && ordinal-- == 0) {
            NvmShapeId shape = nvm_callable_at(a, function, pc);
            if (!count) { CHECK(!shape); return; }
            CHECK(shape != 0);
            if (!shape) return;
            CHECK(nvm_shape_function_count(&a->shapes, shape) == count);
            for (size_t i = 0; i < count; ++i) {
                uint32_t target = UINT32_MAX;
                CHECK(nvm_shape_function_target(&a->shapes, shape, i, &target));
                CHECK(target == expected[i]);
            }
            return;
        }
        pc += size;
    }
    CHECK(0);
}
static int analyze(NvmModule *m, NvmCallableAnalysis *a) {
    int ok = nvm_callable_analyze(m, a);
    if (!ok) fprintf(stderr, "I could not analyze: %s\n", a->error);
    CHECK(ok);
    return ok;
}
#define LEAVES ".function first 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n" \
               ".function second 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n"

static void direct_results_and_joins(void) {
    NvmModule *m = assemble(
        ".function main 0 1 0 void 0\nPUSH_BOOL 1\nCALL choose\nSTORE_LOCAL 0\n"
        "PUSH_I64 7\nLOAD_LOCAL 0\nCALL_INDIRECT 1 1\nPOP\nRET\n.end\n"
        ".function choose 1 1 0 function 1\nLOAD_LOCAL 0\nJMP_FALSE other\n"
        "FUNCREF first\nJMP joined\nother:\nFUNCREF second\njoined:\nRET\n.end\n"
        LEAVES, 1);
    if (!m) return;
    NvmCallableAnalysis a;
    if (analyze(m, &a)) { const uint32_t want[] = {2, 3}; targets(m, &a, 0, 0, want, 2); }
    nvm_callable_destroy(&a); nvm_module_free(m);
}

static void indirect_arguments_and_results(void) {
    NvmModule *m = assemble(
        ".function main 0 1 0 void 0\nFUNCREF first\nFUNCREF relay\nCALL_INDIRECT 1 1\n"
        "STORE_LOCAL 0\nPUSH_I64 7\nLOAD_LOCAL 0\nCALL_INDIRECT 1 1\nPOP\n"
        "FUNCREF first\nPUSH_I64 8\nFUNCREF apply\nCALL_INDIRECT 2 1\nPOP\nRET\n.end\n"
        ".function relay 1 1 0 function 1\nLOAD_LOCAL 0\nRET\n.end\n"
        ".function apply 2 2 0 int 1\nLOAD_LOCAL 1\nLOAD_LOCAL 0\nCALL_INDIRECT 1 1\nRET\n.end\n"
        LEAVES, 1);
    if (!m) return;
    NvmCallableAnalysis a;
    if (analyze(m, &a)) {
        const uint32_t relay[] = {1}, leaf[] = {3}, apply[] = {2};
        targets(m, &a, 0, 0, relay, 1); targets(m, &a, 0, 1, leaf, 1);
        targets(m, &a, 0, 2, apply, 1); targets(m, &a, 2, 0, leaf, 1);
    }
    nvm_callable_destroy(&a); nvm_module_free(m);
}

static void globals_loops_and_separate_producers(void) {
    NvmModule *m = assemble(
        ".function main 0 1 0 void 0\nFUNCREF first\nSTORE_GLOBAL 0\n"
        "again:\nPUSH_BOOL 0\nJMP_FALSE done\nFUNCREF second\nSTORE_GLOBAL 0\nJMP again\n"
        "done:\nPUSH_I64 1\nLOAD_GLOBAL 0\nCALL_INDIRECT 1 1\nPOP\n"
        "PUSH_I64 2\nFUNCREF first\nCALL_INDIRECT 1 1\nPOP\nRET\n.end\n"
        LEAVES, 1);
    if (!m) return;
    NvmCallableAnalysis a;
    if (analyze(m, &a)) {
        const uint32_t both[] = {1, 2}, first[] = {1};
        targets(m, &a, 0, 0, both, 2); targets(m, &a, 0, 1, first, 1);
    }
    nvm_callable_destroy(&a); nvm_module_free(m);
}

static void aggregate_and_array_aliases(void) {
    NvmModule *m = assemble(
        ".function main 0 2 0 void 0\nFUNCREF first\nARR_LITERAL 11 1\n"
        "AGG_PACK 0 0 0 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nCALL relay\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 1\nAGG_GET 0\nPUSH_I64 0\nFUNCREF second\nARR_SET\nPOP\n"
        "PUSH_I64 5\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nCALL_INDIRECT 1 1\nPOP\nRET\n.end\n"
        ".function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n"
        LEAVES, 1);
    if (!m) return;
    NvmCallableAnalysis a;
    if (analyze(m, &a)) { const uint32_t both[] = {2, 3}; targets(m, &a, 0, 0, both, 2); }
    nvm_callable_destroy(&a); nvm_module_free(m);
}

static void unresolved_and_refused(void) {
    const char *cases[] = {
        ".function main 0 0 0 void 0\nPUSH_I64 9\nCALL_INDIRECT 0 0\nRET\n.end\n",
        ".function main 0 0 0 void 0\nFUNCREF leaf\nCALL_INDIRECT 0 0\nRET\n.end\n"
        ".function leaf 1 1 0 void 0\nRET\n.end\n",
        ".function main 0 0 0 void 0\nFUNCREF 99\nPOP\nRET\n.end\n",
        ".function main 0 0 0 void 0\nPOP\nRET\n.end\n",
        ".function main 0 0 0 void 0\nFUNCREF leaf\nCALL_INDIRECT 0 0\nRET\n.end\n"
        ".function leaf 0 0 0 int 1\nPUSH_I64 1\nRET\n.end\n"
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; ++i) {
        NvmModule *m = assemble(cases[i], 0);
        if (!m) continue;
        NvmCallableAnalysis a;
        if (i == 0) {
            CHECK(nvm_callable_analyze(m, &a)); targets(m, &a, 0, 0, NULL, 0);
        } else { CHECK(!nvm_callable_analyze(m, &a)); CHECK(a.error[0]); }
        CHECK(!nvm_callable_at(&a, UINT32_MAX, 0));
        if (i == 1 || i == 4) {
            char error[256];
            char *native = nvm2c_emit(m, error, sizeof error);
            CHECK(native == NULL);
            CHECK(strstr(error, "matching callable argument and result counts") != NULL);
            free(native);
        }
        nvm_callable_destroy(&a); nvm_module_free(m);
    }
}
static void stack_loop_and_zero_target(void) {
    NvmModule *m = assemble(
        ".function main 0 0 0 void 0\nFUNCREF first\nagain:\nPUSH_BOOL 0\nJMP_FALSE done\n"
        "POP\nFUNCREF second\nJMP again\ndone:\nPUSH_I64 3\nSWAP\nCALL_INDIRECT 1 1\nPOP\n"
        "FUNCREF main\nCALL_INDIRECT 0 0\nRET\n.end\n" LEAVES, 1);
    if (!m) return;
    NvmCallableAnalysis a;
    if (analyze(m, &a)) {
        const uint32_t both[] = {1, 2}, entry[] = {0};
        targets(m, &a, 0, 0, both, 2); targets(m, &a, 0, 1, entry, 1);
    }
    nvm_callable_destroy(&a); nvm_module_free(m);
}

static void map_aliases(void) {
    NvmModule *m = assemble(
        ".string key \"key\"\n"
        ".function main 0 2 0 void 0\nHM_NEW 5 11\nPUSH_STR key\nFUNCREF first\nHM_SET\n"
        "STORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 1\nPUSH_STR key\nFUNCREF second\nHM_SET\nPOP\n"
        "PUSH_I64 5\nLOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nCALL_INDIRECT 1 1\nPOP\nRET\n.end\n"
        LEAVES, 1);
    if (!m) return;
    NvmCallableAnalysis a;
    if (analyze(m, &a)) { const uint32_t both[] = {1, 2}; targets(m, &a, 0, 0, both, 2); }
    nvm_callable_destroy(&a); nvm_module_free(m);
}

static void source_fixture(const char *path) {
    AsmResult result = {0};
    NvmModule *m = asm_assemble_file(path, &result);
    CHECK(m != NULL);
    if (!m) { fprintf(stderr, "%s\n", result.message); return; }
    NvmCallableAnalysis a;
    if (analyze(m, &a)) {
        const uint32_t both[] = {4, 5}, first[] = {4};
        for (size_t i = 0; i < 3; ++i) targets(m, &a, 0, i, both, 2);
        targets(m, &a, 0, 3, first, 1);
    }
    nvm_callable_destroy(&a); nvm_module_free(m);
}

int main(int argc, char **argv) {
    direct_results_and_joins(); indirect_arguments_and_results();
    globals_loops_and_separate_producers(); aggregate_and_array_aliases();
    unresolved_and_refused(); stack_loop_and_zero_target(); map_aliases();
    if (argc > 1) source_fixture(argv[1]);
    printf("Callable constraints: %d passed, %d failed\n", passed, failed);
    return failed != 0;
}
