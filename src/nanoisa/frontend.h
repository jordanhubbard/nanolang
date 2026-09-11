/*
 * Shared NanoISA frontend contract (4.6).
 *
 * Every language lowers to the same verified .nvm v2 module. Language-specific
 * work stops at emit. Optimization, verification, NSI, capabilities, FFI
 * isolation, debug, profiling, and translators are shared.
 *
 * Authority: docs/NANOISA_FRONTEND.md
 */

#ifndef NANOISA_FRONTEND_H
#define NANOISA_FRONTEND_H

#include "nvm_format.h"
#include <stddef.h>
#include <stdint.h>

#define NL_FRONTEND_ERR_SIZE 256

typedef enum {
    NL_FE_NANOLANG = 0,
    NL_FE_FORTH,
    NL_FE_SCHEME,
    NL_FE_ML,
    NL_FE_ACTOR,
    NL_FE_DATAFLOW,
    NL_FE_OBJECT,
    NL_FE_SHELL,
    NL_FE_LOGIC,
    NL_FE_COUNT
} NlFrontendId;

typedef enum {
    NL_FE_PHASE_DESUGAR = 0,
    NL_FE_PHASE_TYPECHECK,
    NL_FE_PHASE_EMIT_NVM,
    NL_FE_PHASE_VERIFY,
    NL_FE_PHASE_OPTIMIZE_ISA,
    NL_FE_PHASE_COUNT
} NlFrontendPhase;

typedef struct {
    int ok;
    char error[NL_FRONTEND_ERR_SIZE];
} NlFrontendResult;

typedef struct {
    int nsi;
    int capabilities;
    int ffi_isolation;
    int debug;
    int profiler;
    int nvm2c;
} NlFrontendToolchain;

typedef struct {
    NlFrontendId language;
    const char *source_path;
    uint32_t effect_count;
    const char *const *effects;
    uint32_t cap_count;
    const char *const *capabilities;
    int purity;          /* -1 unknown, 0 impure, 1 pure */
    int exhaustiveness;  /* -1 unknown, 0 no, 1 yes */
    int affine_use;      /* -1 unknown, 0 no, 1 yes */
    int diagnostics_shared;
} NlFrontendFacts;

typedef struct {
    NlFrontendId id;
    const char *name;
    const char *pressure;
    const char *in_scope;
    const char *out_of_scope;
    const char *suite;
    int implemented;
} NlFrontendGoal;

const char *nl_frontend_name(NlFrontendId id);
const NlFrontendGoal *nl_frontend_goal(NlFrontendId id);
int nl_frontend_goals_published(NlFrontendId id);
int nl_frontend_implemented(NlFrontendId id);

int nl_frontend_phase_is_language_specific(NlFrontendPhase phase);
int nl_frontend_opcode_allowed(uint8_t opcode);

NlFrontendToolchain nl_frontend_toolchain(void);

NlFrontendResult nl_frontend_accept(const NvmModule *mod,
                                    const NlFrontendFacts *facts);

NlFrontendResult nl_frontend_accept_linked(const NvmModule *mod,
                                           const NlFrontendFacts *facts,
                                           const NvmModule *const *linked,
                                           uint32_t linked_count);

#endif /* NANOISA_FRONTEND_H */
