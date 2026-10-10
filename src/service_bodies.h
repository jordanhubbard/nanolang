#ifndef NL_SERVICE_BODIES_H
#define NL_SERVICE_BODIES_H
#include "service_namespace.h"
/* I describe nominal body checking only. Status zero is not ownership or
 * execution authority. Every fact borrows its AST node from the namespace. */
typedef struct {
    const ASTNode *node;
    TypeInfo type;
    /* I reserve UINT32_MAX for builtin string length; zero is not a declaration. */
    uint32_t declaration, borrow_mode;
} NlServiceBodyFact;
typedef struct NlServiceBodyCheck {
    /* 0 checked, 1 invalid, 2 unsupported source, 3 size limit.
     * Allocation failure returns NULL instead of a partial result. */
    unsigned status;
    int line, column;
    const char *diagnostic;
    size_t count, functions, shadows;
    NlServiceBodyFact *facts;
    /* I own signatures synthesized for helper values; AST annotations stay borrowed. */
    FunctionSignature **callables;
    size_t callable_count;
} NlServiceBodyCheck;
NlServiceBodyCheck *nl_service_check_bodies(const NlServiceNamespace *);
void nl_service_body_check_free(NlServiceBodyCheck *);
#endif
