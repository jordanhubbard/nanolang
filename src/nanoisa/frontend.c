#include "frontend.h"
#include "isa.h"
#include "verifier.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static const char *const k_effect_names[] = { "IO", "Err", "State" };

static const NlFrontendGoal k_goals[NL_FE_COUNT] = {
    {
        NL_FE_NANOLANG, "NanoLang",
        "native language: explicit types, shadow tests, prefix calls",
        "the language I already compile to verified NanoISA",
        "new syntax sugar",
        "make test",
        1
    },
    {
        NL_FE_FORTH, "Nano Forth",
        "language + assembler + compiler in one design, as an ISA proof",
        "colon definitions compiled to verified NanoISA; Jackson evidence",
        "Forth 2012 Standard System banner; INCLUDED / File-Access",
        "make test-forth-core make test-forth-jackson",
        1
    },
    {
        NL_FE_SCHEME, "Nano Scheme",
        "allocation, callables, tail calls, dynamic values, live code",
        "lexical scope, closures, first-class procedures, recursive data, "
        "interactive evaluation, proper tail calls with constant frame depth",
        "continuations until closures and exceptions are stable",
        "a pinned recognized Scheme subset with documented exclusions",
        1
    },
    {
        NL_FE_ML, "Nano ML",
        "generics, aggregates, exhaustive match, closures, signatures",
        "static inference, ADTs, pattern matching, immutable values, HOFs",
        "a full Standard ML or OCaml implementation",
        "shared aggregate and NSI programs with NanoLang",
        1
    },
    {
        NL_FE_ACTOR, "Nano Actor",
        "isolation, mailboxes, supervision, hot replacement, restart-safe caps",
        "isolated NanoVM contexts first, then Phase 18 process boundaries",
        "a new kernel; embedding policy in the actor language",
        "make test-actor",
        1
    },
    {
        NL_FE_DATAFLOW, "Nano Dataflow",
        "bulk transfer, provenance, replay, cancellation, parallel determinism",
        "typed nodes, streams, backpressure, explicit effects",
        "changing program semantics when the scheduler moves",
        "make test-dataflow",
        1
    },
    {
        NL_FE_OBJECT, "Nano Object",
        "dynamic dispatch, inline caches, layout evolution, live methods",
        "message dispatch, object identity, mutable graphs, reflection",
        "cache-specific portable opcodes",
        "measure specialization without exposing caches in NanoISA",
        0
    },
    {
        NL_FE_SHELL, "Nano Shell",
        "capability-safe orchestration with structured values",
        "processes, files, networks, services only through explicit caps",
        "text-only pipelines as the data model; admin use before policy exists",
        "typed values across pipelines; text parsing as an adapter",
        0
    },
    {
        NL_FE_LOGIC, "Nano Logic",
        "declarative authorization and policy as verified NanoISA",
        "bounded Datalog: facts, rules, queries, deterministic fixed-point",
        "choice points or tabling unless the subset needs them",
        "policy queries compiled to NanoISA or a documented restricted profile",
        0
    }
};

#ifdef __GNUC__
__attribute__((format(printf, 1, 2)))
#endif
static NlFrontendResult fail_fmt(const char *fmt, ...) {
    NlFrontendResult r;
    va_list ap;
    memset(&r, 0, sizeof r);
    r.ok = 0;
    va_start(ap, fmt);
    vsnprintf(r.error, sizeof r.error, fmt, ap);
    va_end(ap);
    return r;
}

static NlFrontendResult ok_result(void) {
    NlFrontendResult r;
    memset(&r, 0, sizeof r);
    r.ok = 1;
    return r;
}

static int effect_allowed(const char *name) {
    size_t i;
    if (!name) return 0;
    for (i = 0; i < sizeof k_effect_names / sizeof k_effect_names[0]; i++) {
        if (strcmp(name, k_effect_names[i]) == 0) return 1;
    }
    return 0;
}

static int cap_allowed(const char *name) {
    return name && strncmp(name, "cap:", 4) == 0 && name[4] != '\0';
}

const char *nl_frontend_name(NlFrontendId id) {
    if ((unsigned)id >= (unsigned)NL_FE_COUNT) return NULL;
    return k_goals[id].name;
}

const NlFrontendGoal *nl_frontend_goal(NlFrontendId id) {
    if ((unsigned)id >= (unsigned)NL_FE_COUNT) return NULL;
    return &k_goals[id];
}

int nl_frontend_goals_published(NlFrontendId id) {
    return nl_frontend_goal(id) != NULL;
}

int nl_frontend_implemented(NlFrontendId id) {
    const NlFrontendGoal *g = nl_frontend_goal(id);
    return g ? g->implemented : 0;
}

int nl_frontend_phase_is_language_specific(NlFrontendPhase phase) {
    return phase == NL_FE_PHASE_DESUGAR || phase == NL_FE_PHASE_TYPECHECK;
}

int nl_frontend_opcode_allowed(uint8_t opcode) {
    return isa_get_info(opcode) != NULL;
}

NlFrontendToolchain nl_frontend_toolchain(void) {
    NlFrontendToolchain t;
    t.nsi = 1;
    t.capabilities = 1;
    t.ffi_isolation = 1;
    t.debug = 1;
    t.profiler = 1;
    t.nvm2c = 1;
    return t;
}

static NlFrontendResult check_opcodes(const NvmModule *mod) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        uint32_t pc = 0;
        if (fn->code_offset > mod->code_size ||
            fn->code_length > mod->code_size - fn->code_offset) {
            return fail_fmt("function %u code range is outside the module", i);
        }
        while (pc < fn->code_length) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(mod->code + fn->code_offset + pc,
                                    fn->code_length - pc, &ins);
            if (n == 0) {
                return fail_fmt("function %u: undecodable instruction at %u",
                                i, pc);
            }
            if (!nl_frontend_opcode_allowed(ins.opcode)) {
                return fail_fmt(
                    "function %u: opcode 0x%02X is not a shared NanoISA primitive",
                    i, ins.opcode);
            }
            pc += n;
        }
    }
    return ok_result();
}

static NlFrontendResult check_typed_functions(const NvmModule *mod) {
    uint32_t i;
    if (mod->function_count == 0) {
        return fail_fmt("a frontend module must contain at least one function");
    }
    for (i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        if (fn->name_idx >= mod->string_count) {
            return fail_fmt("function %u has no name in the constant pool", i);
        }
        if (fn->result_tag >= TAG_COUNT) {
            return fail_fmt("function %u has an invalid result tag", i);
        }
        if ((fn->result_count == 0) != (fn->result_tag == TAG_VOID)) {
            return fail_fmt("function %u result count and tag disagree", i);
        }
        if (fn->local_count < fn->arity) {
            return fail_fmt("function %u has fewer locals than parameters", i);
        }
    }
    return ok_result();
}

static NlFrontendResult check_locations(const NvmModule *mod,
                                        const NlFrontendFacts *facts) {
    if (!facts->source_path || facts->source_path[0] == '\0') {
        return fail_fmt("source path is required");
    }
    if (mod->debug_count == 0) {
        return fail_fmt("source locations are required (DEBUG entries)");
    }
    if (!(mod->header.flags & NVM_FLAG_DEBUG_INFO)) {
        return fail_fmt("DEBUG_INFO flag must be set when locations are present");
    }
    return ok_result();
}

static NlFrontendResult check_facts(const NlFrontendFacts *facts) {
    uint32_t i;
    if (facts->purity < -1 || facts->purity > 1) {
        return fail_fmt("purity must be -1, 0, or 1");
    }
    if (facts->exhaustiveness < -1 || facts->exhaustiveness > 1) {
        return fail_fmt("exhaustiveness must be -1, 0, or 1");
    }
    if (facts->affine_use < -1 || facts->affine_use > 1) {
        return fail_fmt("affine_use must be -1, 0, or 1");
    }
    if (!facts->diagnostics_shared) {
        return fail_fmt("frontends must use the shared diagnostic ids");
    }
    if (facts->effect_count > 0 && !facts->effects) {
        return fail_fmt("effect list is missing");
    }
    for (i = 0; i < facts->effect_count; i++) {
        if (!effect_allowed(facts->effects[i])) {
            return fail_fmt("unknown effect '%s'",
                            facts->effects[i] ? facts->effects[i] : "");
        }
    }
    if (facts->cap_count > 0 && !facts->capabilities) {
        return fail_fmt("capability list is missing");
    }
    for (i = 0; i < facts->cap_count; i++) {
        if (!cap_allowed(facts->capabilities[i])) {
            return fail_fmt("capability must be a cap: identifier");
        }
    }
    return ok_result();
}

NlFrontendResult nl_frontend_accept(const NvmModule *mod,
                                    const NlFrontendFacts *facts) {
    NvmVerifyResult v;
    NlFrontendResult r;

    if (!mod) return fail_fmt("module is null");
    if (!facts) return fail_fmt("frontend facts are required");
    if ((unsigned)facts->language >= (unsigned)NL_FE_COUNT) {
        return fail_fmt("unknown frontend id");
    }
    if (!nl_frontend_goals_published(facts->language)) {
        return fail_fmt("bounded goals are missing for this frontend");
    }
    if (!nl_frontend_implemented(facts->language)) {
        return fail_fmt("%s is not implemented; its bounded goals are published",
                        k_goals[facts->language].name);
    }
    if (mod->header.format_version != NVM_FORMAT_VERSION) {
        return fail_fmt("frontend must emit NanoISA module format %u, not %u",
                        NVM_FORMAT_VERSION, mod->header.format_version);
    }

    r = check_facts(facts);
    if (!r.ok) return r;
    r = check_typed_functions(mod);
    if (!r.ok) return r;
    r = check_locations(mod, facts);
    if (!r.ok) return r;
    r = check_opcodes(mod);
    if (!r.ok) return r;

    v = nvm_verify(mod);
    if (!v.ok) {
        return fail_fmt("verifier rejected the module: %s", v.error_msg);
    }
    return ok_result();
}

NlFrontendResult nl_frontend_accept_linked(const NvmModule *mod,
                                           const NlFrontendFacts *facts,
                                           const NvmModule *const *linked,
                                           uint32_t linked_count) {
    NvmVerifyResult v;
    NlFrontendResult r = nl_frontend_accept(mod, facts);
    if (!r.ok) return r;
    v = nvm_verify_linked(mod, linked, linked_count);
    if (!v.ok) {
        return fail_fmt("linked verifier rejected the module: %s", v.error_msg);
    }
    return ok_result();
}
