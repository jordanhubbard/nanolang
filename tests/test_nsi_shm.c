#include "nsi_cap.h"
#include "nsi_shm.h"

#include <stdio.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static uint32_t shm_rights(void) {
    return NL_CAP_READ | NL_CAP_WRITE | NL_CAP_MAP | NL_CAP_SEAL | NL_CAP_TRANSFER |
           NL_CAP_BORROW | NL_CAP_RETURN | NL_CAP_REVOKE | NL_CAP_DELEGATE;
}

static NlCapTable *minted(NlCap *c) {
    NlCapTable *t = nl_cap_table_create();
    if (!t) return NULL;
    if (nl_cap_mint(t, "nsi:nanolang/shm", "nsi:nanolang/shm", shm_rights(), 1, NULL, c) != 0) {
        nl_cap_table_destroy(t);
        return NULL;
    }
    return t;
}

static void test_kinds_and_copy(void) {
    const char *name = "shm: kinds, mmap and copy fallback match";
    NlCap c;
    NlCapTable *t = minted(&c);
    NlShm *mapr;
    NlShm *copyr;
    unsigned char in[32];
    unsigned char a[32];
    unsigned char b[32];
    NlShmKind kinds[5];
    int i;
    if (!t) { FAIL(name, "mint"); return; }
    memset(in, 0x5a, sizeof(in));
    kinds[0] = NL_SHM_AUDIO;
    kinds[1] = NL_SHM_GRAPHICS;
    kinds[2] = NL_SHM_NET;
    kinds[3] = NL_SHM_FILE;
    kinds[4] = NL_SHM_GPU;
    for (i = 0; i < 5; i++) {
        NlShm *r = nl_shm_create(t, &c, 32, kinds[i], i == 0 ? 1 : 0);
        if (!r || nl_shm_kind(r) != kinds[i] || nl_shm_size(r) != 32) {
            FAIL(name, "kind");
            nl_shm_destroy(r);
            nl_cap_table_destroy(t);
            return;
        }
        nl_shm_destroy(r);
    }
    mapr = nl_shm_create(t, &c, 32, NL_SHM_FILE, 0);
    copyr = nl_shm_create(t, &c, 32, NL_SHM_FILE, 1);
    if (!mapr || !copyr || !nl_shm_copy_fallback(copyr)) {
        FAIL(name, "create"); nl_shm_destroy(mapr); nl_shm_destroy(copyr);
        nl_cap_table_destroy(t); return;
    }
    if (nl_shm_write(mapr, 0, 32, in) != 0 || nl_shm_write(copyr, 0, 32, in) != 0) {
        FAIL(name, "write"); nl_shm_destroy(mapr); nl_shm_destroy(copyr);
        nl_cap_table_destroy(t); return;
    }
    if (nl_shm_read(mapr, 0, 32, a) != 0 || nl_shm_read(copyr, 0, 32, b) != 0 ||
        memcmp(a, b, 32) != 0 || memcmp(a, in, 32) != 0) {
        FAIL(name, "match"); nl_shm_destroy(mapr); nl_shm_destroy(copyr);
        nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_shm_destroy(mapr);
    nl_shm_destroy(copyr);
    nl_cap_table_destroy(t);
}

static void test_validate(void) {
    const char *name = "shm: offset, length, alignment, direction";
    NlCap c;
    NlCapTable *t = minted(&c);
    NlShm *r;
    if (!t) { FAIL(name, "mint"); return; }
    r = nl_shm_create(t, &c, 64, NL_SHM_GPU, 1);
    if (!r) { FAIL(name, "create"); nl_cap_table_destroy(t); return; }
    if (nl_shm_map(r, 1, 8, 8, NL_SHM_DIR_READ) != NL_SHM_ERR_ALIGN) {
        FAIL(name, "align"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    if (nl_shm_map(r, 0, 128, 8, NL_SHM_DIR_READ) != NL_SHM_ERR_RANGE) {
        FAIL(name, "range"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    if (nl_shm_map(r, 0, 8, 8, 0) != NL_SHM_ERR_DIR) {
        FAIL(name, "dir"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    if (nl_shm_map(r, 0, 8, 8, NL_SHM_DIR_READ | NL_SHM_DIR_WRITE) != NL_SHM_OK) {
        FAIL(name, "ok"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_shm_destroy(r);
    nl_cap_table_destroy(t);
}

static void test_rights_own_seal(void) {
    const char *name = "shm: rights, ownership, seal, revoke";
    NlCap c;
    NlCapTable *t = minted(&c);
    NlShm *r;
    unsigned char buf[8];
    memset(buf, 1, sizeof(buf));
    if (!t) { FAIL(name, "mint"); return; }
    r = nl_shm_create(t, &c, 32, NL_SHM_AUDIO, 1);
    if (!r) { FAIL(name, "create"); nl_cap_table_destroy(t); return; }
    if (nl_shm_transfer(r) != 0 || !nl_shm_service_owns(r) ||
        nl_shm_write(r, 0, 8, buf) != NL_SHM_ERR_OWNED) {
        FAIL(name, "owned"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    if (nl_shm_borrow(r) != 0 || nl_shm_return(r) != 0 ||
        nl_shm_write(r, 0, 8, buf) != 0) {
        FAIL(name, "return"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    if (nl_shm_seal(r) != 0 ||
        nl_shm_map(r, 0, 8, 8, NL_SHM_DIR_READ) != NL_SHM_ERR_SEALED) {
        FAIL(name, "seal"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    if (nl_shm_revoke(r) != 0 || nl_shm_read(r, 0, 8, buf) != NL_SHM_ERR) {
        FAIL(name, "revoke"); nl_shm_destroy(r); nl_cap_table_destroy(t); return;
    }
    PASS(name);
    nl_shm_destroy(r);
    nl_cap_table_destroy(t);
}

static void test_bench(void) {
    const char *name = "shm: bench control, copy, map by payload";
    NlShmBench small;
    NlShmBench big;
    if (nl_shm_bench(64, &small) != 0 || nl_shm_bench(4096, &big) != 0) {
        FAIL(name, "run"); return;
    }
    if (small.payload != 64 || big.payload != 4096 ||
        small.copies != 32 || big.mappings != 16 ||
        small.control_ns < 0 || big.copy_ns < 0 || big.map_ns < 0) {
        FAIL(name, "fields"); return;
    }
    PASS(name);
}

int main(void) {
    printf("NSI shared-memory tests\n");
    test_kinds_and_copy();
    test_validate();
    test_rights_own_seal();
    test_bench();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
