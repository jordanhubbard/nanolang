#ifndef NL_SERVICE_OWNERSHIP_H
#define NL_SERVICE_OWNERSHIP_H
#include "service_bodies.h"

/* I retain lexical transfer facts, not bytecode or host authority. */
enum { NL_SERVICE_BIND=1, NL_SERVICE_MOVE, NL_SERVICE_BORROW,
       NL_SERVICE_END_BORROW, NL_SERVICE_REFINE };
typedef struct {
    const ASTNode *node;
    /* Binding IDs are lexical and local to this check, never host handles.
     * REFINE mode is the source arm index; other modes are 0/1/2 for
     * owned/shared/exclusive. The retained body facts supply nominal types. */
    uint32_t action, binding, mode;
} NlServiceOwnershipFact;
typedef struct NlServiceOwnershipCheck {
    unsigned status; /* 0 checked, 1 invalid, 2 unsupported, 3 size/allocation limit. */
    const char *diagnostic;
    int line, column;
    size_t count, functions, shadows;
    NlServiceOwnershipFact *facts;
} NlServiceOwnershipCheck;
/* I borrow the complete namespace/body facts during checking. Output nodes
 * remain borrowed from that namespace. Allocation failure can return NULL. */
NlServiceOwnershipCheck *nl_service_check_ownership(const NlServiceNamespace *, const NlServiceBodyCheck *);
void nl_service_ownership_free(NlServiceOwnershipCheck *);
#endif
