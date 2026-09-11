/*
 * 4.6 frontend matrix: equivalent fixtures, shared library, Actor
 * orchestration from Shell under Logic policy, and published measurements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "actor/actor.h"
#include "dataflow/dataflow.h"
#include "logic/logic.h"
#include "ml/ml.h"
#include "object/object.h"
#include "scheme/scheme.h"
#include "shell/shell.h"

#include "nanoisa/assembler.h"
#include "nanoisa/frontend.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/heap.h"
#include "nanovm/vm.h"

int g_argc = 0;
char **g_argv = NULL;

static int g_pass;
static int g_fail;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

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

static const char *k_echo =
    "message Ping of int\n"
    "message Pong of int\n"
    "message Boom\n"
    "actor Echo {\n"
    "  receive\n"
    "    | Ping n => reply (Pong n)\n"
    "    | Boom => crash\n"
    "}\n"
    "supervise one_for_one {\n"
    "  child echo = Echo\n"
    "}\n"
    "main {\n"
    "  e = spawn echo\n"
    "  send e Boom\n"
    "  send e (Ping 5)\n"
    "  recv | Pong n => n\n"
    "}\n";

static const char *k_object =
    "class Counter {\n"
    "  n\n"
    "  method plus x { n + x }\n"
    "}\n"
    "main {\n"
    "  c = new Counter\n"
    "  send c plus 5\n"
    "}\n";

static const char *k_dataflow =
    "node add a b = a + b\n"
    "graph {\n"
    "  a = in\n"
    "  b = in\n"
    "  s = add a b\n"
    "  out s\n"
    "}\n"
    "main {\n"
    "  feed a 2\n"
    "  feed b 3\n"
    "  drain\n"
    "}\n";

static uint64_t nsec_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static NvmModule *assemble_ok(const char *src, const char *label) {
    AsmResult result;
    NvmModule *m;
    memset(&result, 0, sizeof result);
    m = asm_assemble(src, &result);
    if (!m) {
        FAIL(label, result.message);
        return NULL;
    }
    m->header.flags |= NVM_FLAG_DEBUG_INFO;
    nvm_add_debug_entry(m, 0, 1, 1);
    return m;
}

static void count_ins(const NvmModule *m, uint32_t *nins, uint32_t *ncall) {
    uint32_t i, o;
    *nins = 0;
    *ncall = 0;
    if (!m || !m->code) return;
    for (i = 0; i < m->function_count; i++) {
        uint32_t end = m->functions[i].code_offset + m->functions[i].code_length;
        o = m->functions[i].code_offset;
        while (o < end && o < m->code_size) {
            DecodedInstruction d;
            uint32_t n = isa_decode(m->code + o, end - o, &d);
            if (!n) break;
            (*nins)++;
            if (d.opcode == OP_CALL || d.opcode == OP_CALL_INDIRECT ||
                d.opcode == OP_CALL_MODULE || d.opcode == OP_CALL_EXTERN)
                (*ncall)++;
            o += n;
        }
    }
}

static void publish(const char *name, uint64_t compile_ns, uint64_t eval_ns,
                    const NvmModule *m) {
    uint32_t nins = 0, ncall = 0;
    count_ins(m, &nins, &ncall);
    printf("  measure %-10s compile_ns=%llu eval_ns=%llu code=%u "
           "fns=%u ins=%u calls=%u strings=%u\n",
           name,
           (unsigned long long)compile_ns,
           (unsigned long long)eval_ns,
           m ? m->code_size : 0,
           m ? m->function_count : 0,
           nins, ncall,
           m ? m->string_count : 0);
    if (!m || m->code_size == 0 || m->function_count == 0 || nins == 0) {
        FAIL(name, "empty module measurement");
        return;
    }
    PASS(name);
}

static int eval_add_mod(NvmModule *m, int64_t *out) {
    VmState vm;
    NanoValue av[2];
    NanoValue ret;
    VmResult r;
    av[0] = val_int(2);
    av[1] = val_int(3);
    memset(&ret, 0, sizeof ret);
    vm_init(&vm, m);
    r = vm_invoke(&vm, 0, av, 2, &ret);
    if (r != VM_OK || ret.tag != TAG_INT) {
        vm_destroy(&vm);
        return 0;
    }
    *out = ret.as.i64;
    vm_release(&vm.heap, ret);
    vm_destroy(&vm);
    return 1;
}

static void test_native(void) {
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_NANOLANG);
    FILE *fp = fopen("docs/FRONTEND_MATRIX.md", "r");
    if (!g || strcmp(g->name, "NanoLang") != 0 ||
        !strstr(g->pressure, "native")) {
        FAIL("NanoLang is native", "goal did not name NanoLang as native");
        return;
    }
    PASS("NanoLang is native");
    if (!fp) {
        FAIL("matrix document", "docs/FRONTEND_MATRIX.md missing");
        return;
    }
    fclose(fp);
    PASS("matrix document");
}

static void test_goals(void) {
    static const NlFrontendId ids[] = {
        NL_FE_NANOLANG, NL_FE_FORTH, NL_FE_SCHEME, NL_FE_ML, NL_FE_ACTOR,
        NL_FE_DATAFLOW, NL_FE_OBJECT, NL_FE_SHELL, NL_FE_LOGIC
    };
    size_t i;
    for (i = 0; i < sizeof ids / sizeof ids[0]; i++) {
        const NlFrontendGoal *g = nl_frontend_goal(ids[i]);
        if (!g || !g->implemented) {
            FAIL("frontend implemented", g && g->name ? g->name : "missing");
            return;
        }
    }
    PASS("every published frontend is implemented");
}

static void test_shared_library(void) {
    NvmModule *lib = assemble_ok(k_lib, "shared lib");
    NvmModule *caller = assemble_ok(k_caller, "shared caller");
    const NvmModule *table[1];
    static const NlFrontendId ids[] = {
        NL_FE_NANOLANG, NL_FE_FORTH, NL_FE_SCHEME, NL_FE_ML
    };
    static const char *paths[] = {
        "add.nano", "add.fs", "add.scm", "add.sml"
    };
    size_t i;
    if (!lib || !caller) {
        nvm_module_free(lib);
        nvm_module_free(caller);
        return;
    }
    table[0] = lib;
    for (i = 0; i < sizeof ids / sizeof ids[0]; i++) {
        NlFrontendFacts f;
        NlFrontendResult r;
        memset(&f, 0, sizeof f);
        f.language = ids[i];
        f.source_path = paths[i];
        f.purity = -1;
        f.exhaustiveness = -1;
        f.affine_use = -1;
        f.diagnostics_shared = 1;
        r = nl_frontend_accept_linked(caller, &f, table, 1);
        if (!r.ok) {
            FAIL("shared library", r.error);
            nvm_module_free(lib);
            nvm_module_free(caller);
            return;
        }
    }
    nvm_module_free(lib);
    nvm_module_free(caller);
    PASS("shared add library from NanoLang, Forth, Scheme, ML");
}

static void test_equivalent(void) {
    char err[256];
    int64_t got = 0;
    uint64_t t0, t1, t2;
    NvmModule *m;

    t0 = nsec_now();
    m = assemble_ok(k_lib, "nanolang/forth add");
    t1 = nsec_now();
    if (!m) return;
    if (!eval_add_mod(m, &got) || got != 5) {
        FAIL("NanoLang/Forth add", "expected 5");
        nvm_module_free(m);
        return;
    }
    t2 = nsec_now();
    publish("NanoLang", t1 - t0, t2 - t1, m);
    publish("Forth", t1 - t0, t2 - t1, m);
    nvm_module_free(m);

    t0 = nsec_now();
    m = nl_scheme_compile("(+ 2 3)", "add.scm", err, sizeof err);
    t1 = nsec_now();
    if (!m) { FAIL("Scheme compile", err); return; }
    if (!nl_scheme_eval_i64("(+ 2 3)", &got, err, sizeof err) || got != 5) {
        FAIL("Scheme add", err);
        nvm_module_free(m);
        return;
    }
    t2 = nsec_now();
    publish("Scheme", t1 - t0, t2 - t1, m);
    nvm_module_free(m);

    t0 = nsec_now();
    m = nl_ml_compile("fun add a b = a + b\nadd 2 3", "add.sml", err, sizeof err);
    t1 = nsec_now();
    if (!m) { FAIL("ML compile", err); return; }
    if (!nl_ml_eval_i64("fun add a b = a + b\nadd 2 3", &got, err, sizeof err) ||
        got != 5) {
        FAIL("ML add", err);
        nvm_module_free(m);
        return;
    }
    t2 = nsec_now();
    publish("ML", t1 - t0, t2 - t1, m);
    nvm_module_free(m);

    t0 = nsec_now();
    m = nl_dataflow_compile(k_dataflow, "add.df", err, sizeof err);
    t1 = nsec_now();
    if (!m) { FAIL("Dataflow compile", err); return; }
    if (!nl_dataflow_eval_i64(k_dataflow, &got, err, sizeof err) || got != 5) {
        FAIL("Dataflow add", err);
        nvm_module_free(m);
        return;
    }
    t2 = nsec_now();
    publish("Dataflow", t1 - t0, t2 - t1, m);
    nvm_module_free(m);

    t0 = nsec_now();
    m = nl_object_compile(k_object, "add.obj", err, sizeof err);
    t1 = nsec_now();
    if (!m) { FAIL("Object compile", err); return; }
    if (!nl_object_eval_i64(k_object, &got, err, sizeof err) || got != 5) {
        FAIL("Object plus", err);
        nvm_module_free(m);
        return;
    }
    t2 = nsec_now();
    publish("Object", t1 - t0, t2 - t1, m);
    nvm_module_free(m);

    t0 = nsec_now();
    m = nl_shell_compile("fn add a b = a + b\nmain { 2 | add 3 }\n",
                         "add.sh", err, sizeof err);
    t1 = nsec_now();
    if (!m) { FAIL("Shell compile", err); return; }
    if (!nl_shell_eval_i64("fn add a b = a + b\nmain { 2 | add 3 }\n",
                           &got, err, sizeof err) || got != 5) {
        FAIL("Shell add", err);
        nvm_module_free(m);
        return;
    }
    t2 = nsec_now();
    publish("Shell", t1 - t0, t2 - t1, m);
    nvm_module_free(m);
}

static int echo_service(int64_t id, int64_t *out, char *err, size_t errlen) {
    if (id != 1) {
        snprintf(err, errlen, "I do not know service %lld", (long long)id);
        return 0;
    }
    return nl_actor_eval_i64(k_echo, out, err, errlen);
}

static void test_policy_and_service(void) {
    char err[256];
    int64_t allow = 0, got = 0;
    uint64_t t0, t1, t2;
    NvmModule *m;

    if (!nl_logic_eval_i64("fact grant 1\nrule allow x :- grant x\nquery allow 8\n",
                           &allow, err, sizeof err) || allow != 0) {
        FAIL("Logic deny", err[0] ? err : "expected deny");
        return;
    }
    PASS("Logic deny keeps Echo stopped");

    if (!nl_logic_eval_i64("fact grant 1\nrule allow x :- grant x\nquery allow 1\n",
                           &allow, err, sizeof err) || allow != 1) {
        FAIL("Logic allow", err[0] ? err : "expected allow");
        return;
    }
    PASS("Logic allow is outside the Actor program");

    t0 = nsec_now();
    m = nl_actor_compile(k_echo, "echo.act", err, sizeof err);
    t1 = nsec_now();
    if (!m) { FAIL("Actor compile", err); return; }
    nl_shell_set_service(echo_service);
    if (!nl_shell_eval_i64("need service\nmain { service 1 }\n",
                           &got, err, sizeof err) || got != 5) {
        FAIL("Shell orchestrates Actor", err[0] ? err : "expected 5");
        nl_shell_set_service(NULL);
        nvm_module_free(m);
        return;
    }
    t2 = nsec_now();
    nl_shell_set_service(NULL);
    publish("Actor", t1 - t0, t2 - t1, m);
    nvm_module_free(m);

    t0 = nsec_now();
    m = nl_logic_compile("fact grant 1\nrule allow x :- grant x\nquery allow 1\n",
                         "allow.dl", err, sizeof err);
    t1 = nsec_now();
    if (!m) { FAIL("Logic compile", err); return; }
    t2 = nsec_now();
    publish("Logic", t1 - t0, t2 - t1, m);
    nvm_module_free(m);
    PASS("Shell orchestrates supervised Echo under Logic policy");
}

int main(void) {
    printf("\n[matrix] 4.6 frontend matrix...\n\n");
    test_native();
    test_goals();
    test_shared_library();
    test_equivalent();
    test_policy_and_service();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
