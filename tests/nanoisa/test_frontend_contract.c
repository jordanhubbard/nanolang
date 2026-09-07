/*
 * 4.6 shared NanoISA frontend contract.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "assembler.h"
#include "frontend.h"
#include "isa.h"
#include "nvm_format.h"
#include "verifier.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, what) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s  (%s:%d)\n", (what), __FILE__, __LINE__); } \
} while (0)

static NvmModule *assemble_ok(const char *src, const char *label) {
    AsmResult result;
    memset(&result, 0, sizeof result);
    NvmModule *m = asm_assemble(src, &result);
    if (!m) {
        printf("  FAIL: %s assemble: %s (line %u)\n",
               label, result.message, result.line);
        g_fail++;
    }
    return m;
}

static void attach_debug(NvmModule *m) {
    if (!m) return;
    m->header.flags |= NVM_FLAG_DEBUG_INFO;
    nvm_add_debug_entry(m, 0, 1, 1);
}

static NlFrontendFacts facts_for(NlFrontendId id, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = id;
    f.source_path = path;
    f.purity = -1;
    f.exhaustiveness = -1;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return f;
}

static const char k_add[] =
    ".function add 2 2 0 int 1\n"
    "  LOAD_LOCAL 0\n"
    "  LOAD_LOCAL 1\n"
    "  ADD\n"
    "  RET\n"
    ".end\n";

static const char k_lib[] =
    ".function add 2 2 0 int 1\n"
    "  LOAD_LOCAL 0\n"
    "  LOAD_LOCAL 1\n"
    "  ADD\n"
    "  RET\n"
    ".end\n";

static const char k_caller[] =
    ".module_ref \"lib\"\n"
    ".function main 0 2 0 int 1\n"
    "  PUSH_I64 2\n"
    "  PUSH_I64 3\n"
    "  CALL_MODULE 0 0 2 1\n"
    "  RET\n"
    ".end\n";

static void test_goals_published_before_implementation(void) {
    NlFrontendId id;
    for (id = 0; id < NL_FE_COUNT; id++) {
        const NlFrontendGoal *g = nl_frontend_goal(id);
        CHECK(g != NULL, "every frontend has published bounded goals");
        CHECK(g->name && g->name[0], "goal names the language");
        CHECK(g->pressure && g->pressure[0], "goal names the ISA pressure");
        CHECK(g->in_scope && g->in_scope[0], "goal has an in-scope bound");
        CHECK(g->out_of_scope && g->out_of_scope[0], "goal has an out-of-scope bound");
        CHECK(g->suite && g->suite[0], "goal names its tests");
        CHECK(nl_frontend_goals_published(id), "goals_published matches the table");
    }
    CHECK(nl_frontend_implemented(NL_FE_NANOLANG), "NanoLang is implemented");
    CHECK(nl_frontend_implemented(NL_FE_FORTH), "Forth is implemented");
    CHECK(!nl_frontend_implemented(NL_FE_SCHEME), "Scheme is not started");
    CHECK(!nl_frontend_implemented(NL_FE_ML), "ML is not started");
    CHECK(!nl_frontend_implemented(NL_FE_ACTOR), "Actor is not started");
    CHECK(!nl_frontend_implemented(NL_FE_DATAFLOW), "Dataflow is not started");
    CHECK(!nl_frontend_implemented(NL_FE_OBJECT), "Object is not started");
    CHECK(!nl_frontend_implemented(NL_FE_SHELL), "Shell is not started");
    CHECK(!nl_frontend_implemented(NL_FE_LOGIC), "Logic is not started");
    CHECK(nl_frontend_goal(NL_FE_COUNT) == NULL, "out-of-range goal is refused");
}

static void test_phases_separate_language_from_isa(void) {
    CHECK(nl_frontend_phase_is_language_specific(NL_FE_PHASE_DESUGAR),
          "desugaring is language-specific");
    CHECK(nl_frontend_phase_is_language_specific(NL_FE_PHASE_TYPECHECK),
          "type analysis is language-specific");
    CHECK(!nl_frontend_phase_is_language_specific(NL_FE_PHASE_EMIT_NVM),
          "emit is language-neutral");
    CHECK(!nl_frontend_phase_is_language_specific(NL_FE_PHASE_VERIFY),
          "verify is language-neutral");
    CHECK(!nl_frontend_phase_is_language_specific(NL_FE_PHASE_OPTIMIZE_ISA),
          "ISA optimization is language-neutral");
}

static void test_toolchain_is_shared(void) {
    NlFrontendToolchain t = nl_frontend_toolchain();
    CHECK(t.nsi, "NSI is available to every frontend");
    CHECK(t.capabilities, "capabilities are available to every frontend");
    CHECK(t.ffi_isolation, "FFI isolation is available to every frontend");
    CHECK(t.debug, "debugger metadata is available to every frontend");
    CHECK(t.profiler, "profiler is available to every frontend");
    CHECK(t.nvm2c, "nvm2c is available to every frontend");
}

static void test_opcodes_are_shared(void) {
    unsigned op;
    CHECK(nl_frontend_opcode_allowed(OP_ADD), "ADD is a shared primitive");
    CHECK(nl_frontend_opcode_allowed(OP_CALL_MODULE),
          "CALL_MODULE is a shared primitive");
    CHECK(!nl_frontend_opcode_allowed(0xFF),
          "0xFF is not a frontend-private opcode");
    for (op = 0; op < 256; op++) {
        int allowed = nl_frontend_opcode_allowed((uint8_t)op);
        int in_isa = isa_get_info((uint8_t)op) != NULL;
        CHECK(allowed == in_isa, "allowed opcodes are exactly the shared ISA");
        if (allowed != in_isa) break;
    }
}

static void test_nanolang_and_forth_accept_same_module_shape(void) {
    NvmModule *nano = assemble_ok(k_add, "nanolang add");
    NvmModule *forth = assemble_ok(k_add, "forth add");
    NlFrontendFacts fn = facts_for(NL_FE_NANOLANG, "add.nano");
    NlFrontendFacts ff = facts_for(NL_FE_FORTH, "add.fs");
    NlFrontendResult rn, rf;
    static const char *effects[] = { "IO" };
    static const char *caps[] = { "cap:nanolang/log.write" };

    CHECK(nano != NULL && forth != NULL, "both add fixtures assemble");
    if (!nano || !forth) {
        nvm_module_free(nano);
        nvm_module_free(forth);
        return;
    }
    attach_debug(nano);
    attach_debug(forth);
    fn.effect_count = 1;
    fn.effects = effects;
    fn.cap_count = 1;
    fn.capabilities = caps;
    fn.purity = 0;

    rn = nl_frontend_accept(nano, &fn);
    rf = nl_frontend_accept(forth, &ff);
    CHECK(rn.ok, "NanoLang add is accepted");
    if (!rn.ok) printf("    %s\n", rn.error);
    CHECK(rf.ok, "Forth add is accepted");
    if (!rf.ok) printf("    %s\n", rf.error);
    CHECK(nano->header.format_version == NVM_FORMAT_VERSION,
          "NanoLang emits the shared module version");
    CHECK(forth->header.format_version == NVM_FORMAT_VERSION,
          "Forth emits the shared module version");
    nvm_module_free(nano);
    nvm_module_free(forth);
}

static void test_scheme_refused_until_implemented(void) {
    NvmModule *m = assemble_ok(k_add, "scheme add");
    NlFrontendFacts f;
    NlFrontendResult r;
    if (!m) return;
    attach_debug(m);
    f = facts_for(NL_FE_SCHEME, "add.scm");
    r = nl_frontend_accept(m, &f);
    CHECK(!r.ok, "Scheme is not implemented");
    CHECK(strstr(r.error, "not implemented") != NULL,
          "Scheme error names the unpublished implementation");
    nvm_module_free(m);
}

static void test_requires_locations_types_and_shared_diags(void) {
    NvmModule *m = assemble_ok(k_add, "locations");
    NlFrontendFacts f;
    NlFrontendResult r;
    if (!m) return;

    f = facts_for(NL_FE_NANOLANG, "add.nano");
    r = nl_frontend_accept(m, &f);
    CHECK(!r.ok, "missing DEBUG entries fail closed");

    attach_debug(m);
    f.diagnostics_shared = 0;
    r = nl_frontend_accept(m, &f);
    CHECK(!r.ok, "private diagnostics fail closed");

    f.diagnostics_shared = 1;
    f.source_path = "";
    r = nl_frontend_accept(m, &f);
    CHECK(!r.ok, "empty source path fails closed");

    f.source_path = "add.nano";
    f.effect_count = 1;
    f.effects = NULL;
    r = nl_frontend_accept(m, &f);
    CHECK(!r.ok, "missing effect list fails closed");

    {
        static const char *bad[] = { "Kernel" };
        f.effects = bad;
        r = nl_frontend_accept(m, &f);
        CHECK(!r.ok, "unknown effect fails closed");
    }

    {
        static const char *badcap[] = { "log.write" };
        f.effect_count = 0;
        f.effects = NULL;
        f.cap_count = 1;
        f.capabilities = badcap;
        r = nl_frontend_accept(m, &f);
        CHECK(!r.ok, "capability without cap: prefix fails closed");
    }

    m->header.format_version = 1;
    f.cap_count = 0;
    f.capabilities = NULL;
    r = nl_frontend_accept(m, &f);
    CHECK(!r.ok, "v1 module format fails closed");
    nvm_module_free(m);

    r = nl_frontend_accept(NULL, &f);
    CHECK(!r.ok, "null module fails closed");
    r = nl_frontend_accept((NvmModule *)1, NULL);
    CHECK(!r.ok, "null facts fail closed");
}

static void test_cross_frontend_shared_library(void) {
    NvmModule *lib = assemble_ok(k_lib, "shared lib");
    NvmModule *nano = assemble_ok(k_caller, "nanolang caller");
    NvmModule *forth = assemble_ok(k_caller, "forth caller");
    NlFrontendFacts fn, ff;
    NlFrontendResult rn, rf;
    const NvmModule *table[1];

    CHECK(lib && nano && forth, "shared-library fixtures assemble");
    if (!lib || !nano || !forth) {
        nvm_module_free(lib);
        nvm_module_free(nano);
        nvm_module_free(forth);
        return;
    }
    attach_debug(lib);
    attach_debug(nano);
    attach_debug(forth);

    fn = facts_for(NL_FE_NANOLANG, "caller.nano");
    ff = facts_for(NL_FE_FORTH, "caller.fs");
    table[0] = lib;

    rn = nl_frontend_accept_linked(nano, &fn, table, 1);
    rf = nl_frontend_accept_linked(forth, &ff, table, 1);
    CHECK(rn.ok, "NanoLang caller links against the shared library");
    if (!rn.ok) printf("    %s\n", rn.error);
    CHECK(rf.ok, "Forth caller links against the same shared library");
    if (!rf.ok) printf("    %s\n", rf.error);

    nvm_module_free(lib);
    nvm_module_free(nano);
    nvm_module_free(forth);
}

int main(void) {
    printf("\n[frontend] shared NanoISA frontend contract...\n\n");
    test_goals_published_before_implementation();
    test_phases_separate_language_from_isa();
    test_toolchain_is_shared();
    test_opcodes_are_shared();
    test_nanolang_and_forth_accept_same_module_shape();
    test_scheme_refused_until_implemented();
    test_requires_locations_types_and_shared_diags();
    test_cross_frontend_shared_library();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
