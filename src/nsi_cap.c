#include "nsi_cap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <stdlib.h>
#else
#include <unistd.h>
#endif

#define NL_CAP_SLOTS 64
#define NL_CAP_AUDIT 128
#define NL_CAP_FORTH 64

typedef struct {
    int used;
    int revoked;
    int transferable;
    uint32_t rights;
    uint32_t generation;
    uint64_t secret;
    char type_id[128];
    char service_id[128];
    char scope[256];
} NlCapSlot;

typedef struct {
    int kind;
    char id[32];
} NlCapAudit;

struct NlCapTable {
    NlCapSlot slots[NL_CAP_SLOTS];
    uint32_t next_generation;
    uint64_t forth_secret[NL_CAP_FORTH];
    uint32_t forth_slot[NL_CAP_FORTH];
    int forth_used[NL_CAP_FORTH];
    NlCapAudit audit[NL_CAP_AUDIT];
    int audit_n;
    int audit_seq;
};

static void bounded_copy(char *dest, size_t dest_size, const char *src) {
    size_t n;
    if (!dest || dest_size == 0) {
        return;
    }
    if (!src) {
        dest[0] = '\0';
        return;
    }
    n = strlen(src);
    if (n >= dest_size) {
        n = dest_size - 1;
    }
    memcpy(dest, src, n);
    dest[n] = '\0';
}

static void fill_rand(void *buf, size_t n) {
#if defined(__APPLE__)
    arc4random_buf(buf, n);
#else
    if (getentropy(buf, n) != 0)
        memset(buf, 0x5a, n);
#endif
}

static void audit(NlCapTable *t, int kind) {
    NlCapAudit *a;
    if (!t || t->audit_n >= NL_CAP_AUDIT) return;
    a = &t->audit[t->audit_n++];
    a->kind = kind;
    t->audit_seq++;
    snprintf(a->id, sizeof(a->id), "CAP%02u", (unsigned)t->audit_seq);
}

static int find_live(NlCapTable *t, const NlCap *c, NlCapSlot **out) {
    NlCapSlot *s;
    if (!t || !c) {
        if (t) audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_MALFORMED;
    }
    if (c->slot >= NL_CAP_SLOTS) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_FORGED;
    }
    s = &t->slots[c->slot];
    if (!s->used || s->secret != c->secret) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_FORGED;
    }
    if (s->generation != c->generation) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_STALE;
    }
    if (s->revoked) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_REVOKED;
    }
    if (out) *out = s;
    return NL_CAP_OK;
}

static int mint_slot(NlCapTable *t, const char *type_id, const char *service_id,
                     uint32_t rights, int transferable, const char *scope,
                     NlCap *out) {
    int i;
    uint64_t secret = 0;
    if (!t || !type_id || !service_id || !out) return NL_CAP_ERR_MALFORMED;
    if (type_id[0] == '\0' || service_id[0] == '\0') return NL_CAP_ERR_MALFORMED;
    for (i = 0; i < NL_CAP_SLOTS; i++) {
        if (!t->slots[i].used) break;
    }
    if (i == NL_CAP_SLOTS) return NL_CAP_ERR_FULL;
    fill_rand(&secret, sizeof(secret));
    if (secret == 0) secret = 1;
    t->next_generation++;
    memset(&t->slots[i], 0, sizeof(t->slots[i]));
    t->slots[i].used = 1;
    t->slots[i].revoked = 0;
    t->slots[i].transferable = transferable ? 1 : 0;
    t->slots[i].rights = rights;
    t->slots[i].generation = t->next_generation;
    t->slots[i].secret = secret;
    bounded_copy(t->slots[i].type_id, sizeof(t->slots[i].type_id), type_id);
    bounded_copy(t->slots[i].service_id, sizeof(t->slots[i].service_id), service_id);
    if (scope)
        bounded_copy(t->slots[i].scope, sizeof(t->slots[i].scope), scope);
    out->secret = secret;
    out->slot = (uint32_t)i;
    out->generation = t->slots[i].generation;
    return NL_CAP_OK;
}

NlCapTable *nl_cap_table_create(void) {
    NlCapTable *t = calloc(1, sizeof(*t));
    return t;
}

void nl_cap_table_destroy(NlCapTable *t) {
    free(t);
}

int nl_cap_mint(NlCapTable *t, const char *type_id, const char *service_id,
                uint32_t rights, int transferable, const char *scope, NlCap *out) {
    int rc = mint_slot(t, type_id, service_id, rights, transferable, scope, out);
    if (rc == NL_CAP_OK) audit(t, NL_CAP_AUDIT_CREATE);
    else if (t) audit(t, NL_CAP_AUDIT_FAIL);
    return rc;
}

int nl_cap_check(NlCapTable *t, const NlCap *c, uint32_t need) {
    NlCapSlot *s = NULL;
    int rc = find_live(t, c, &s);
    if (rc != NL_CAP_OK) return rc;
    if ((s->rights & need) != need) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_RIGHTS;
    }
    audit(t, NL_CAP_AUDIT_USE);
    return NL_CAP_OK;
}

int nl_cap_attenuate(NlCapTable *t, const NlCap *parent, uint32_t rights, NlCap *out) {
    NlCapSlot *s = NULL;
    int rc = find_live(t, parent, &s);
    if (rc != NL_CAP_OK) return rc;
    if ((s->rights & NL_CAP_DELEGATE) == 0) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_RIGHTS;
    }
    if ((rights & s->rights) != rights) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_RIGHTS;
    }
    rc = mint_slot(t, s->type_id, s->service_id, rights, s->transferable, s->scope, out);
    if (rc == NL_CAP_OK) audit(t, NL_CAP_AUDIT_DELEGATE);
    else audit(t, NL_CAP_AUDIT_FAIL);
    return rc;
}

int nl_cap_transfer(NlCapTable *t, const NlCap *src, NlCap *out) {
    NlCapSlot *s = NULL;
    int rc = find_live(t, src, &s);
    if (rc != NL_CAP_OK) return rc;
    if (!s->transferable || (s->rights & NL_CAP_TRANSFER) == 0) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_TRANSFER;
    }
    rc = mint_slot(t, s->type_id, s->service_id, s->rights, 1, s->scope, out);
    if (rc != NL_CAP_OK) {
        audit(t, NL_CAP_AUDIT_FAIL);
        return rc;
    }
    s->revoked = 1;
    audit(t, NL_CAP_AUDIT_TRANSFER);
    return NL_CAP_OK;
}

int nl_cap_revoke(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    int rc = find_live(t, c, &s);
    if (rc != NL_CAP_OK) return rc;
    if ((s->rights & NL_CAP_REVOKE) == 0 && (s->rights & NL_CAP_DELEGATE) == 0) {
        /* creator may revoke with REVOKE; delegate parent also allowed via own revoke right */
        audit(t, NL_CAP_AUDIT_FAIL);
        return NL_CAP_ERR_RIGHTS;
    }
    s->revoked = 1;
    audit(t, NL_CAP_AUDIT_REVOKE);
    return NL_CAP_OK;
}

int nl_cap_restart(NlCapTable *t) {
    int i;
    if (!t) return NL_CAP_ERR_MALFORMED;
    for (i = 0; i < NL_CAP_SLOTS; i++) {
        if (t->slots[i].used) {
            t->slots[i].revoked = 1;
            t->slots[i].generation = 0;
            t->slots[i].secret = 0;
            t->slots[i].used = 0;
        }
    }
    for (i = 0; i < NL_CAP_FORTH; i++)
        t->forth_used[i] = 0;
    t->next_generation++;
    audit(t, NL_CAP_AUDIT_RESTART);
    return NL_CAP_OK;
}

int nl_cap_invalidate_service(NlCapTable *t, const char *service_id) {
    int i;
    if (!t || !service_id || !service_id[0]) return NL_CAP_ERR_MALFORMED;
    for (i = 0; i < NL_CAP_SLOTS; i++) {
        if (!t->slots[i].used) continue;
        if (strcmp(t->slots[i].service_id, service_id) != 0) continue;
        t->slots[i].revoked = 1;
        t->slots[i].generation = 0;
        t->slots[i].secret = 0;
        t->slots[i].used = 0;
    }
    for (i = 0; i < NL_CAP_FORTH; i++) {
        uint32_t slot;
        if (!t->forth_used[i]) continue;
        slot = t->forth_slot[i];
        if (slot < NL_CAP_SLOTS && !t->slots[slot].used)
            t->forth_used[i] = 0;
    }
    t->next_generation++;
    audit(t, NL_CAP_AUDIT_RESTART);
    return NL_CAP_OK;
}

int nl_cap_from_integer(NlCapTable *t, uint64_t n, NlCap *out) {
    (void)n;
    if (out) memset(out, 0, sizeof(*out));
    if (t) audit(t, NL_CAP_AUDIT_FAIL);
    return NL_CAP_ERR_FORGED;
}

int nl_cap_from_pointer(NlCapTable *t, void *p, NlCap *out) {
    (void)p;
    if (out) memset(out, 0, sizeof(*out));
    if (t) audit(t, NL_CAP_AUDIT_FAIL);
    return NL_CAP_ERR_FORGED;
}

int nl_cap_resource_own(NlCapTable *t, const char *nsi_type_id,
                        const char *service_id, uint32_t rights, NlCap *out) {
    return nl_cap_mint(t, nsi_type_id, service_id, rights | NL_CAP_TRANSFER,
                       1, NULL, out);
}

int nl_cap_resource_consume(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    int rc = find_live(t, c, &s);
    if (rc != NL_CAP_OK) return rc;
    s->revoked = 1;
    audit(t, NL_CAP_AUDIT_REVOKE);
    return NL_CAP_OK;
}

int nl_cap_forth_bind(NlCapTable *t, const NlCap *c, uint64_t *cell) {
    NlCapSlot *s = NULL;
    int i;
    uint64_t token = 0;
    int rc = find_live(t, c, &s);
    if (rc != NL_CAP_OK) return rc;
    if (!cell) return NL_CAP_ERR_MALFORMED;
    for (i = 0; i < NL_CAP_FORTH; i++) {
        if (!t->forth_used[i]) break;
    }
    if (i == NL_CAP_FORTH) return NL_CAP_ERR_FULL;
    fill_rand(&token, sizeof(token));
    if (token == 0 || token < 0x10000ull) token += 0x10000ull;
    t->forth_used[i] = 1;
    t->forth_secret[i] = token;
    t->forth_slot[i] = c->slot;
    *cell = token;
    return NL_CAP_OK;
}

int nl_cap_forth_lookup(NlCapTable *t, uint64_t cell, NlCap *out) {
    int i;
    if (!t || !out) return NL_CAP_ERR_MALFORMED;
    for (i = 0; i < NL_CAP_FORTH; i++) {
        if (t->forth_used[i] && t->forth_secret[i] == cell) {
            NlCapSlot *s = &t->slots[t->forth_slot[i]];
            if (!s->used || s->revoked) {
                audit(t, NL_CAP_AUDIT_FAIL);
                return NL_CAP_ERR_REVOKED;
            }
            out->secret = s->secret;
            out->slot = t->forth_slot[i];
            out->generation = s->generation;
            return NL_CAP_OK;
        }
    }
    audit(t, NL_CAP_AUDIT_FAIL);
    return NL_CAP_ERR_FORGED;
}

uint32_t nl_cap_rights(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    if (find_live(t, c, &s) != NL_CAP_OK) return 0;
    return s->rights;
}

uint32_t nl_cap_generation(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    if (find_live(t, c, &s) != NL_CAP_OK) return 0;
    return s->generation;
}

const char *nl_cap_type_id(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    if (find_live(t, c, &s) != NL_CAP_OK) return "";
    return s->type_id;
}

const char *nl_cap_service_id(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    if (find_live(t, c, &s) != NL_CAP_OK) return "";
    return s->service_id;
}

const char *nl_cap_scope(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    if (find_live(t, c, &s) != NL_CAP_OK) return "";
    return s->scope;
}

int nl_cap_revoked(NlCapTable *t, const NlCap *c) {
    if (!t || !c || c->slot >= NL_CAP_SLOTS) return 1;
    if (!t->slots[c->slot].used) return 1;
    if (t->slots[c->slot].secret != c->secret) return 1;
    return t->slots[c->slot].revoked;
}

int nl_cap_transferable(NlCapTable *t, const NlCap *c) {
    NlCapSlot *s = NULL;
    if (find_live(t, c, &s) != NL_CAP_OK) return 0;
    return s->transferable;
}

int nl_cap_audit_count(const NlCapTable *t) {
    return t ? t->audit_n : 0;
}

int nl_cap_audit_kind(const NlCapTable *t, int i) {
    if (!t || i < 0 || i >= t->audit_n) return -1;
    return t->audit[i].kind;
}

const char *nl_cap_audit_id(const NlCapTable *t, int i) {
    if (!t || i < 0 || i >= t->audit_n) return "";
    return t->audit[i].id;
}
