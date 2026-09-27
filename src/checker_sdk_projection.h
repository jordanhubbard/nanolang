#ifndef CHECKER_SDK_PROJECTION_H
#define CHECKER_SDK_PROJECTION_H
#include "nanolang.h"
#include "nanoisa/preparation_budget.h"
/* I capture actual checked source occurrences, not executable SDK authority.
 * Keep AST and Environment alive and immutable until I am freed. These views
 * borrow complete annotations and their owner/substitution context. Source and
 * Environment ordinals are NOT emitted layout/type/signature ordinals. */
typedef struct CheckerSdkProjection CheckerSdkProjection;
typedef enum { CHECKER_SDK_CAPTURED, CHECKER_SDK_INVALID, CHECKER_SDK_LIMIT,
               CHECKER_SDK_MEMORY, CHECKER_SDK_CHECK_FAILED } CheckerSdkStatus;
typedef struct {
    const ASTNode *declaration;
    Type kind;
    size_t source_ordinal, environment_ordinal;
    const char *owner; /* Exact annotation context. */
    const char *declaration_owner; /* Opaque source origin is distinct. */
} CheckerSdkSourceRow;
/* I run the ordinary real source check and issue a capture only on success.
 * Projection failure preserves output/budget, not the checker's existing
 * Environment/AST mutation semantics. No nested/concurrent capture is supported.
 * The budget bounds projection work/storage; the ordinary checker remains its
 * separate compiler operation. The output slot must be unpublished. */
CheckerSdkStatus type_check_sdk_projection(ASTNode *,Environment *,bool module,
    NvmPreparationBudget *,CheckerSdkProjection **);
void checker_sdk_projection_free(CheckerSdkProjection *);
bool checker_sdk_projection_row(const CheckerSdkProjection *,size_t source_ordinal,
    CheckerSdkSourceRow *);
/* This exact retained environment supplies the existing nominal/substitution
 * machinery during synchronous producer construction. It is borrowed, not a
 * mutable public registry or a serialized proof. */
const Environment *checker_sdk_projection_environment(const CheckerSdkProjection *);
#endif
