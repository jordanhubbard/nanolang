#include "nsi_cap.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static uint32_t full_rights(void) {
    return NL_CAP_READ | NL_CAP_WRITE | NL_CAP_MAP | NL_CAP_SEAL | NL_CAP_TRANSFER |
           NL_CAP_BORROW | NL_CAP_RETURN | NL_CAP_REVOKE | NL_CAP_DELEGATE |
           NL_CAP_EVAL | NL_CAP_BUFFER | NL_CAP_ECHO | NL_CAP_CHROME;
}

static int has_kind(NlCapTable *t, int kind) {
    int i;
    for (i = 0; i < nl_cap_audit_count(t); i++)
        if (nl_cap_audit_kind(t, i) == kind) return 1;
    return 0;
}

static void test_unforgeable(void) {
    const char *name = "cap: integers and pointers are forged";
    NlCapTable *t = nl_cap_table_create();
    NlCap c;
    if (!t) { FAIL(name, "table"); return; }
    if (nl_cap_from_integer(t, 42, &c) != NL_CAP_ERR_FORGED) {
        FAIL(name, "integer"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_from_pointer(t, (void *)t, &c) != NL_CAP_ERR_FORGED) {
        FAIL(name, "pointer"); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_cap_table_destroy(t);
}

static void test_table_fields(void) {
    const char *name = "cap: type, service, rights, delegation, revocation";
    NlCapTable *t = nl_cap_table_create();
    NlCap c;
    if (!t) { FAIL(name, "table"); return; }
    if (nl_cap_mint(t, "nsi:nanolang/fs#File", "nsi:nanolang/fs",
                    full_rights(), 1, "/tmp", &c) != NL_CAP_OK) {
        FAIL(name, "mint"); nl_cap_table_destroy(t); return;
    }
    if (strcmp(nl_cap_type_id(t, &c), "nsi:nanolang/fs#File") != 0) {
        FAIL(name, "type"); nl_cap_table_destroy(t); return;
    }
    if (strcmp(nl_cap_service_id(t, &c), "nsi:nanolang/fs") != 0) {
        FAIL(name, "service"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_rights(t, &c) != full_rights()) {
        FAIL(name, "rights"); nl_cap_table_destroy(t); return;
    }
    if (!nl_cap_transferable(t, &c) || strcmp(nl_cap_scope(t, &c), "/tmp") != 0) {
        FAIL(name, "policy"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_revoke(t, &c) != NL_CAP_OK || !nl_cap_revoked(t, &c)) {
        FAIL(name, "revoke"); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_cap_table_destroy(t);
}

static void test_attenuate(void) {
    const char *name = "cap: rights attenuate on delegate";
    NlCapTable *t = nl_cap_table_create();
    NlCap p, d;
    if (!t) { FAIL(name, "table"); return; }
    if (nl_cap_mint(t, "t", "s", full_rights(), 1, NULL, &p) != NL_CAP_OK) {
        FAIL(name, "mint"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_attenuate(t, &p, NL_CAP_READ | NL_CAP_WRITE | NL_CAP_MAP, &d) != NL_CAP_OK) {
        FAIL(name, "attenuate"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_check(t, &d, NL_CAP_READ) != NL_CAP_OK) {
        FAIL(name, "read"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_check(t, &d, NL_CAP_TRANSFER) != NL_CAP_ERR_RIGHTS) {
        FAIL(name, "no widen"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_attenuate(t, &p, full_rights() | 0x8000u, &d) != NL_CAP_ERR_RIGHTS) {
        FAIL(name, "subset"); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_cap_table_destroy(t);
}

static void test_transfer_permission(void) {
    const char *name = "cap: transfer requires permission";
    NlCapTable *t = nl_cap_table_create();
    NlCap a, b, c;
    if (!t) { FAIL(name, "table"); return; }
    if (nl_cap_mint(t, "t", "s", NL_CAP_READ | NL_CAP_DELEGATE, 0, NULL, &a) != NL_CAP_OK) {
        FAIL(name, "mint"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_transfer(t, &a, &b) != NL_CAP_ERR_TRANSFER) {
        FAIL(name, "blocked"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_mint(t, "t", "s", NL_CAP_READ | NL_CAP_TRANSFER, 1, NULL, &b) != NL_CAP_OK) {
        FAIL(name, "mint2"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_transfer(t, &b, &c) != NL_CAP_OK || nl_cap_check(t, &b, NL_CAP_READ) != NL_CAP_ERR_REVOKED) {
        FAIL(name, "move"); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_cap_table_destroy(t);
}

static void test_restart_generation(void) {
    const char *name = "cap: restart invalidates; generation not reused";
    NlCapTable *t = nl_cap_table_create();
    NlCap a, b;
    uint32_t gen;
    if (!t) { FAIL(name, "table"); return; }
    if (nl_cap_mint(t, "t", "svc-a", full_rights(), 1, NULL, &a) != NL_CAP_OK) {
        FAIL(name, "mint"); nl_cap_table_destroy(t); return;
    }
    gen = nl_cap_generation(t, &a);
    if (nl_cap_restart(t) != NL_CAP_OK) {
        FAIL(name, "restart"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_check(t, &a, NL_CAP_READ) != NL_CAP_ERR_FORGED) {
        FAIL(name, "stale"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_mint(t, "t", "svc-a", full_rights(), 1, NULL, &b) != NL_CAP_OK) {
        FAIL(name, "remint"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_generation(t, &b) == gen || nl_cap_generation(t, &b) == 0) {
        FAIL(name, "reuse"); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_cap_table_destroy(t);
}

static void test_resource_and_forth(void) {
    const char *name = "cap: resource own/consume; Forth cells not pointers";
    NlCapTable *t = nl_cap_table_create();
    NlCap c, d;
    uint64_t cell = 0;
    if (!t) { FAIL(name, "table"); return; }
    if (nl_cap_resource_own(t, "nsi:nanolang/fs#File", "nsi:nanolang/fs",
                            NL_CAP_READ | NL_CAP_REVOKE, &c) != NL_CAP_OK) {
        FAIL(name, "own"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_forth_bind(t, &c, &cell) != NL_CAP_OK || cell < 0x10000ull) {
        FAIL(name, "bind"); nl_cap_table_destroy(t); return;
    }
    if ((void *)(uintptr_t)cell == (void *)t) {
        FAIL(name, "pointer"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_forth_lookup(t, cell, &d) != NL_CAP_OK || d.slot != c.slot) {
        FAIL(name, "lookup"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_forth_lookup(t, 1, &d) != NL_CAP_ERR_FORGED) {
        FAIL(name, "forged cell"); nl_cap_table_destroy(t); return;
    }
    if (nl_cap_resource_consume(t, &c) != NL_CAP_OK ||
        nl_cap_check(t, &c, NL_CAP_READ) != NL_CAP_ERR_REVOKED) {
        FAIL(name, "consume"); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_cap_table_destroy(t);
}

static void test_audit(void) {
    const char *name = "cap: audit create, delegate, use, revoke, fail";
    NlCapTable *t = nl_cap_table_create();
    NlCap p, d;
    if (!t) { FAIL(name, "table"); return; }
    nl_cap_mint(t, "t", "s", full_rights(), 1, NULL, &p);
    nl_cap_check(t, &p, NL_CAP_READ);
    nl_cap_attenuate(t, &p, NL_CAP_READ | NL_CAP_DELEGATE, &d);
    nl_cap_revoke(t, &d);
    nl_cap_from_integer(t, 7, &d);
    if (!has_kind(t, NL_CAP_AUDIT_CREATE) || !has_kind(t, NL_CAP_AUDIT_USE) ||
        !has_kind(t, NL_CAP_AUDIT_DELEGATE) || !has_kind(t, NL_CAP_AUDIT_REVOKE) ||
        !has_kind(t, NL_CAP_AUDIT_FAIL) || nl_cap_audit_id(t, 0)[0] == '\0') {
        FAIL(name, "kinds"); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_cap_table_destroy(t);
}

int main(void) {
    printf("NSI capability runtime tests\n");
    test_unforgeable();
    test_table_fields();
    test_attenuate();
    test_transfer_permission();
    test_restart_generation();
    test_resource_and_forth();
    test_audit();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
