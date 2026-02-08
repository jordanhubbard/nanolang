#ifndef NANOCORE_SUBSET_H
#define NANOCORE_SUBSET_H

#include "nanolang.h"

/* Trust levels for formal verification status */
typedef enum {
    TRUST_VERIFIED    = 1,  /* NanoCore only. Type soundness + determinism proven in Coq */
    TRUST_TYPECHECKED = 2,  /* Full NanoLang types. Checked by typechecker, not formally proven */
    TRUST_FFI         = 3,  /* Calls FFI/extern C code. No formal verification */
    TRUST_UNSAFE      = 4   /* Inside unsafe blocks or manual memory management */
} TrustLevel;

/* Result for a single function's trust analysis */
typedef struct {
    const char *function_name;
    TrustLevel level;
    const char *reason;       /* Human-readable explanation */
    Type return_type;
    int param_count;
} FunctionTrust;

/* Trust report for an entire program */
typedef struct {
    FunctionTrust *entries;
    int count;
    int capacity;
} TrustReport;

/* Generate a trust report for all functions in the program */
TrustReport *nanocore_trust_report(ASTNode *program, Environment *env);

/* Print a trust report to stdout */
void nanocore_print_trust_report(TrustReport *report, const char *filename);

/* Free a trust report */
void nanocore_free_trust_report(TrustReport *report);

/* Check if a single AST node is in the NanoCore subset */
bool nanocore_is_subset(ASTNode *node, Environment *env);

/* Get the trust level for a single function */
TrustLevel nanocore_function_trust(ASTNode *func, Environment *env);

#endif /* NANOCORE_SUBSET_H */
