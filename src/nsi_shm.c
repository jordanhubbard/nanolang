#include "nsi_shm.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

struct NlShm {
    NlCapTable *table;
    NlCap cap;
    size_t size;
    NlShmKind kind;
    unsigned char *data;
    int mmaped;
    int copy_fallback;
    int sealed;
    int service_owns;
    int borrowed;
    int revoked;
    int mapped;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

NlShm *nl_shm_create(NlCapTable *t, const NlCap *cap, size_t bytes, NlShmKind kind,
                     int force_copy) {
    NlShm *r;
    uint32_t need = NL_CAP_READ | NL_CAP_WRITE | NL_CAP_MAP;
    if (!t || !cap || bytes == 0 || bytes > (16u * 1024u * 1024u)) return NULL;
    if (nl_cap_check(t, cap, need) != NL_CAP_OK) return NULL;
    r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->table = t;
    r->cap = *cap;
    r->size = bytes;
    r->kind = kind;
    if (!force_copy) {
        void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_SHARED, -1, 0);
        if (p != MAP_FAILED) {
            r->data = p;
            r->mmaped = 1;
        }
    }
    if (!r->data) {
        r->data = calloc(1, bytes);
        if (!r->data) {
            free(r);
            return NULL;
        }
        r->copy_fallback = 1;
    }
    return r;
}

void nl_shm_destroy(NlShm *r) {
    if (!r) return;
    if (r->mmaped && r->data)
        munmap(r->data, r->size);
    else
        free(r->data);
    free(r);
}

int nl_shm_map(NlShm *r, size_t off, size_t len, size_t align, int dir) {
    if (!r || r->revoked) return NL_SHM_ERR;
    if (r->sealed) return NL_SHM_ERR_SEALED;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_MAP) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    if (align == 0 || (align & (align - 1)) != 0) return NL_SHM_ERR_ALIGN;
    if (off % align != 0) return NL_SHM_ERR_ALIGN;
    if (len == 0 || off + len < off || off + len > r->size) return NL_SHM_ERR_RANGE;
    if ((dir & (NL_SHM_DIR_READ | NL_SHM_DIR_WRITE)) == 0) return NL_SHM_ERR_DIR;
    if ((dir & NL_SHM_DIR_WRITE) && r->service_owns) return NL_SHM_ERR_OWNED;
    if ((dir & NL_SHM_DIR_READ) && nl_cap_check(r->table, &r->cap, NL_CAP_READ) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    if ((dir & NL_SHM_DIR_WRITE) && nl_cap_check(r->table, &r->cap, NL_CAP_WRITE) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    r->mapped = 1;
    return NL_SHM_OK;
}

int nl_shm_read(NlShm *r, size_t off, size_t len, void *dst) {
    if (!r || !dst || r->revoked) return NL_SHM_ERR;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_READ) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    if (len == 0 || off + len < off || off + len > r->size) return NL_SHM_ERR_RANGE;
    memcpy(dst, r->data + off, len);
    return NL_SHM_OK;
}

int nl_shm_write(NlShm *r, size_t off, size_t len, const void *src) {
    if (!r || !src || r->revoked) return NL_SHM_ERR;
    if (r->service_owns) return NL_SHM_ERR_OWNED;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_WRITE) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    if (len == 0 || off + len < off || off + len > r->size) return NL_SHM_ERR_RANGE;
    memcpy(r->data + off, src, len);
    return NL_SHM_OK;
}

int nl_shm_seal(NlShm *r) {
    if (!r || r->revoked) return NL_SHM_ERR;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_SEAL) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    r->sealed = 1;
    return NL_SHM_OK;
}

int nl_shm_transfer(NlShm *r) {
    if (!r || r->revoked) return NL_SHM_ERR;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_TRANSFER) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    r->service_owns = 1;
    r->borrowed = 0;
    return NL_SHM_OK;
}

int nl_shm_borrow(NlShm *r) {
    if (!r || r->revoked) return NL_SHM_ERR;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_BORROW) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    r->borrowed = 1;
    return NL_SHM_OK;
}

int nl_shm_return(NlShm *r) {
    if (!r || r->revoked) return NL_SHM_ERR;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_RETURN) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    r->service_owns = 0;
    r->borrowed = 0;
    return NL_SHM_OK;
}

int nl_shm_revoke(NlShm *r) {
    if (!r) return NL_SHM_ERR;
    if (nl_cap_check(r->table, &r->cap, NL_CAP_REVOKE) != NL_CAP_OK)
        return NL_SHM_ERR_RIGHTS;
    r->revoked = 1;
    return NL_SHM_OK;
}

int nl_shm_copy_fallback(const NlShm *r) {
    return r ? r->copy_fallback : 1;
}

int nl_shm_service_owns(const NlShm *r) {
    return r ? r->service_owns : 0;
}

NlShmKind nl_shm_kind(const NlShm *r) {
    return r ? r->kind : NL_SHM_FILE;
}

size_t nl_shm_size(const NlShm *r) {
    return r ? r->size : 0;
}

int nl_shm_bench(size_t payload, NlShmBench *out) {
    NlCapTable *t;
    NlCap cap;
    NlShm *r;
    unsigned char *buf;
    uint64_t t0, t1;
    int i;
    uint32_t rights = NL_CAP_READ | NL_CAP_WRITE | NL_CAP_MAP | NL_CAP_SEAL |
                      NL_CAP_TRANSFER | NL_CAP_BORROW | NL_CAP_RETURN | NL_CAP_REVOKE |
                      NL_CAP_DELEGATE;
    if (!out || payload == 0 || payload > (1024u * 1024u)) return NL_SHM_ERR;
    memset(out, 0, sizeof(*out));
    out->payload = payload;
    t = nl_cap_table_create();
    if (!t) return NL_SHM_ERR;
    if (nl_cap_mint(t, "nsi:nanolang/shm", "nsi:nanolang/bench", rights, 1, NULL, &cap) != 0) {
        nl_cap_table_destroy(t);
        return NL_SHM_ERR;
    }
    buf = malloc(payload);
    if (!buf) {
        nl_cap_table_destroy(t);
        return NL_SHM_ERR;
    }
    memset(buf, 0xab, payload);
    t0 = now_ns();
    for (i = 0; i < 64; i++)
        (void)nl_cap_check(t, &cap, NL_CAP_READ);
    t1 = now_ns();
    out->control_ns = (double)(t1 - t0) / 64.0;
    r = nl_shm_create(t, &cap, payload, NL_SHM_FILE, 1);
    if (!r) {
        free(buf);
        nl_cap_table_destroy(t);
        return NL_SHM_ERR;
    }
    t0 = now_ns();
    for (i = 0; i < 16; i++) {
        if (nl_shm_write(r, 0, payload, buf) != 0 || nl_shm_read(r, 0, payload, buf) != 0) {
            nl_shm_destroy(r);
            free(buf);
            nl_cap_table_destroy(t);
            return NL_SHM_ERR;
        }
    }
    t1 = now_ns();
    out->copy_ns = (double)(t1 - t0) / 16.0;
    out->copies = 32;
    out->used_copy_fallback = 1;
    nl_shm_destroy(r);
    r = nl_shm_create(t, &cap, payload, NL_SHM_FILE, 0);
    if (!r) {
        free(buf);
        nl_cap_table_destroy(t);
        return NL_SHM_ERR;
    }
    t0 = now_ns();
    for (i = 0; i < 16; i++) {
        if (nl_shm_map(r, 0, payload, 1, NL_SHM_DIR_READ | NL_SHM_DIR_WRITE) != 0) {
            nl_shm_destroy(r);
            free(buf);
            nl_cap_table_destroy(t);
            return NL_SHM_ERR;
        }
    }
    t1 = now_ns();
    out->map_ns = (double)(t1 - t0) / 16.0;
    out->mappings = 16;
    out->used_copy_fallback = nl_shm_copy_fallback(r);
    nl_shm_destroy(r);
    free(buf);
    nl_cap_table_destroy(t);
    return NL_SHM_OK;
}
