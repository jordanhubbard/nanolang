#ifndef NL_NSI_SERVICE_CATALOG_INTERNAL_H
#define NL_NSI_SERVICE_CATALOG_INTERNAL_H
#include "nsi_service_catalog.h"
#include <string.h>

/* My plans select these immutable descriptors in production code. This helper
 * compares already valid in-memory arrays/strings and grants no authority. */
typedef struct {
    const char *interface_id, *interface_name;
    const char *error_id, *error_name, *error_version;
    const char *capability_id, *capability_name;
    const NlServicePlanMethod *methods;
    size_t method_count;
    const NlServicePlanType *types;
    size_t type_count;
} NlServiceCatalog;
static inline bool nl_service_catalog_text_equal(const char *a, const char *b) {
    return (!a || !b) ? a == b : strcmp(a, b) == 0;
}
static inline bool nl_service_catalog_document_equal(const NlNsi *n, const NlServiceCatalog *c) {
    if (!n || n->version != NL_NSI_VERSION ||
        !nl_service_catalog_text_equal(n->iface.id, c->interface_id) ||
        !nl_service_catalog_text_equal(n->iface.name, c->interface_name) ||
        n->method_count != c->method_count || n->type_count != c->type_count ||
        n->error_count != 1 || n->capability_count != 1 ||
        !n->methods || !n->types || !n->errors || !n->capabilities) return false;
    if (!nl_service_catalog_text_equal(n->errors[0].id, c->error_id) ||
        !nl_service_catalog_text_equal(n->errors[0].name, c->error_name) ||
        !nl_service_catalog_text_equal(n->errors[0].version, c->error_version) ||
        !nl_service_catalog_text_equal(n->capabilities[0].id, c->capability_id) ||
        !nl_service_catalog_text_equal(n->capabilities[0].name, c->capability_name)) return false;
    for (size_t i = 0; i < c->method_count; i++) {
        const NlNsiMethod *a = &n->methods[i];
        const NlServicePlanMethod *b = &c->methods[i];
        if (!nl_service_catalog_text_equal(a->id, b->id) ||
            !nl_service_catalog_text_equal(a->name, b->name) || a->idempotent != 0 ||
            a->param_count != b->param_count || !a->params) return false;
        for (size_t j = 0; j < b->param_count; j++) {
            const NlNsiParam *x = &a->params[j];
            const NlServicePlanParam *y = &b->params[j];
            if (!nl_service_catalog_text_equal(x->id, y->id) ||
                !nl_service_catalog_text_equal(x->name, y->name) ||
                !nl_service_catalog_text_equal(x->type_id, y->type_id) ||
                x->direction != y->direction || x->ownership != y->ownership ||
                x->lifetime != y->lifetime || x->mutability != y->mutability ||
                x->optional != 0 || x->streaming != NL_NSI_STREAM_NONE) return false;
        }
    }
    for (size_t i = 0; i < c->type_count; i++) {
        const NlNsiType *a = &n->types[i];
        const NlServicePlanType *b = &c->types[i];
        if (!nl_service_catalog_text_equal(a->id, b->id) ||
            !nl_service_catalog_text_equal(a->name, b->name) || a->kind != b->kind ||
            a->member_count != b->member_count || a->element_id || a->method_id || a->result_id ||
            (b->member_count && !a->members) || (!b->member_count && a->members)) return false;
        for (size_t j = 0; j < b->member_count; j++) {
            const NlNsiMember *x = &a->members[j];
            const NlServicePlanMember *y = &b->members[j];
            if (!nl_service_catalog_text_equal(x->id, y->id) ||
                !nl_service_catalog_text_equal(x->name, y->name) ||
                !nl_service_catalog_text_equal(x->type_id, y->type_id)) return false;
        }
    }
    return true;
}
#endif
