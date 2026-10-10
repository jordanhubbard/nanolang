#ifndef NL_SERVICE_NAMESPACE_H
#define NL_SERVICE_NAMESPACE_H
#include "nanolang.h"
#include "nanoisa/file_source_plan.h"

/* I retain source identities before nominal checking and wire lowering. */
enum {
    NL_SERVICE_FUNCTION = 1, NL_SERVICE_RECORD, NL_SERVICE_UNION,
    NL_SERVICE_ENUM, NL_SERVICE_OPAQUE, NL_SERVICE_GLOBAL,
    NL_SERVICE_MODULE, NL_SERVICE_TYPE, NL_SERVICE_METHOD
};
typedef struct {
    const char *name;
    uint32_t id, target, module, target_module, kind, ordinal, request;
    bool exported, immutable;
    ASTNode *declaration; /* I borrow the parsed graph during compilation. */
} NlServiceName;
typedef struct NlServiceNamespace NlServiceNamespace;
/* I require the complete, dependency-first parsed graph. I never parse source
 * or grant service authority here. Failure leaves *out unchanged. */
NlFileSourceStatus nl_service_namespace_build(ASTNode *, Environment *, ModuleList *,
                                              const char *, NlServiceNamespace **out);
void nl_service_namespace_free(NlServiceNamespace *);
size_t nl_service_namespace_count(const NlServiceNamespace *);
const NlServiceName *nl_service_namespace_name(const NlServiceNamespace *, size_t);
const NlServiceName *nl_service_namespace_lookup(const NlServiceNamespace *,
                                                const char *module, const char *name);
const char *nl_service_namespace_module(const NlServiceNamespace *, uint32_t);
ASTNode *nl_service_namespace_program(const NlServiceNamespace *, uint32_t);
const NlFileSourcePlan *nl_service_namespace_plan(const NlServiceNamespace *);
int64_t nl_service_namespace_catalog(const NlServiceNamespace *, uint32_t module);
/* I return scalar or exact catalog types, never authority from spelling alone.
 * Returned TypeInfo values contain no owning pointers. Failure preserves out. */
typedef struct {
    uint32_t declaration, module, ordinal, parameter_count;
    TypeInfo parameters[2], result;
    uint32_t input_mode;
} NlServiceSignature;
bool nl_service_type(const NlServiceNamespace *, const char *module,
                     const char *name, TypeInfo *out);
bool nl_service_member_type(const NlServiceNamespace *, const TypeInfo *,
                            const char *member, TypeInfo *out);
bool nl_service_method_type(const NlServiceNamespace *, const char *module,
                            const char *name, NlServiceSignature *out);
#endif
