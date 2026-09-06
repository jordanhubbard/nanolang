#ifndef NL_NSI_CAP_H
#define NL_NSI_CAP_H

#include <stddef.h>
#include <stdint.h>

#define NL_CAP_OK 0
#define NL_CAP_ERR_FORGED 1
#define NL_CAP_ERR_REVOKED 2
#define NL_CAP_ERR_STALE 3
#define NL_CAP_ERR_RIGHTS 4
#define NL_CAP_ERR_TRANSFER 5
#define NL_CAP_ERR_FULL 6
#define NL_CAP_ERR_MALFORMED 7

#define NL_CAP_READ      1u
#define NL_CAP_WRITE     2u
#define NL_CAP_MAP       4u
#define NL_CAP_SEAL      8u
#define NL_CAP_TRANSFER  16u
#define NL_CAP_BORROW    32u
#define NL_CAP_RETURN    64u
#define NL_CAP_REVOKE    128u
#define NL_CAP_DELEGATE  256u
#define NL_CAP_EVAL      512u
#define NL_CAP_BUFFER    1024u
#define NL_CAP_ECHO      2048u
#define NL_CAP_CHROME    4096u

typedef enum {
    NL_CAP_AUDIT_CREATE = 0,
    NL_CAP_AUDIT_DELEGATE,
    NL_CAP_AUDIT_USE,
    NL_CAP_AUDIT_REVOKE,
    NL_CAP_AUDIT_FAIL,
    NL_CAP_AUDIT_TRANSFER,
    NL_CAP_AUDIT_RESTART
} NlCapAuditKind;

typedef struct {
    uint64_t secret;
    uint32_t slot;
    uint32_t generation;
} NlCap;

typedef struct NlCapTable NlCapTable;

NlCapTable *nl_cap_table_create(void);
void nl_cap_table_destroy(NlCapTable *t);

int nl_cap_mint(NlCapTable *t, const char *type_id, const char *service_id,
                uint32_t rights, int transferable, const char *scope, NlCap *out);
int nl_cap_check(NlCapTable *t, const NlCap *c, uint32_t need);
int nl_cap_attenuate(NlCapTable *t, const NlCap *parent, uint32_t rights, NlCap *out);
int nl_cap_transfer(NlCapTable *t, const NlCap *src, NlCap *out);
int nl_cap_revoke(NlCapTable *t, const NlCap *c);
int nl_cap_restart(NlCapTable *t);
int nl_cap_invalidate_service(NlCapTable *t, const char *service_id);

int nl_cap_from_integer(NlCapTable *t, uint64_t n, NlCap *out);
int nl_cap_from_pointer(NlCapTable *t, void *p, NlCap *out);

int nl_cap_resource_own(NlCapTable *t, const char *nsi_type_id,
                        const char *service_id, uint32_t rights, NlCap *out);
int nl_cap_resource_consume(NlCapTable *t, const NlCap *c);

int nl_cap_forth_bind(NlCapTable *t, const NlCap *c, uint64_t *cell);
int nl_cap_forth_lookup(NlCapTable *t, uint64_t cell, NlCap *out);

uint32_t nl_cap_rights(NlCapTable *t, const NlCap *c);
uint32_t nl_cap_generation(NlCapTable *t, const NlCap *c);
const char *nl_cap_type_id(NlCapTable *t, const NlCap *c);
const char *nl_cap_service_id(NlCapTable *t, const NlCap *c);
const char *nl_cap_scope(NlCapTable *t, const NlCap *c);
int nl_cap_revoked(NlCapTable *t, const NlCap *c);
int nl_cap_transferable(NlCapTable *t, const NlCap *c);

int nl_cap_audit_count(const NlCapTable *t);
int nl_cap_audit_kind(const NlCapTable *t, int i);
const char *nl_cap_audit_id(const NlCapTable *t, int i);

#endif
