#include "nsi_journal.h"

#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Compact SHA-256 (public-domain style). */

static uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

static void sha256_transform(uint32_t s[8], const unsigned char block[64]) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    int i;
    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = s[0]; b = s[1]; c = s[2]; d = s[3];
    e = s[4]; f = s[5]; g = s[6]; h = s[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

static void sha256(const unsigned char *data, size_t n, unsigned char out[32]) {
    uint32_t s[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    unsigned char block[64];
    size_t i;
    uint64_t bits = (uint64_t)n * 8ull;
    memset(block, 0, sizeof(block));
    while (n >= 64) {
        sha256_transform(s, data);
        data += 64;
        n -= 64;
    }
    memcpy(block, data, n);
    block[n] = 0x80;
    if (n >= 56) {
        sha256_transform(s, block);
        memset(block, 0, sizeof(block));
    }
    block[56] = (unsigned char)(bits >> 56);
    block[57] = (unsigned char)(bits >> 48);
    block[58] = (unsigned char)(bits >> 40);
    block[59] = (unsigned char)(bits >> 32);
    block[60] = (unsigned char)(bits >> 24);
    block[61] = (unsigned char)(bits >> 16);
    block[62] = (unsigned char)(bits >> 8);
    block[63] = (unsigned char)bits;
    sha256_transform(s, block);
    for (i = 0; i < 8; i++) {
        out[i * 4] = (unsigned char)(s[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(s[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(s[i] >> 8);
        out[i * 4 + 3] = (unsigned char)s[i];
    }
}

void nl_sha256_hex(const unsigned char *data, size_t n, char hex[NL_JOURNAL_HASH_HEX]) {
    unsigned char d[32];
    int i;
    sha256(data, n, d);
    for (i = 0; i < 32; i++)
        snprintf(hex + i * 2, 3, "%02x", d[i]);
    hex[64] = 0;
}

static void hmac_sha256(const unsigned char *key, size_t klen,
                        const unsigned char *msg, size_t mlen,
                        unsigned char out[32]) {
    unsigned char k[64];
    unsigned char ipad[64];
    unsigned char opad[64];
    unsigned char inner[32];
    unsigned char *ibuf;
    unsigned char obuf[64 + 32];
    size_t i;
    memset(k, 0, sizeof(k));
    if (klen > 64) {
        sha256(key, klen, k);
    } else {
        memcpy(k, key, klen);
    }
    for (i = 0; i < 64; i++) {
        ipad[i] = (unsigned char)(k[i] ^ 0x36u);
        opad[i] = (unsigned char)(k[i] ^ 0x5cu);
    }
    ibuf = malloc(64 + mlen);
    if (!ibuf) {
        memset(out, 0, 32);
        return;
    }
    memcpy(ibuf, ipad, 64);
    memcpy(ibuf + 64, msg, mlen);
    sha256(ibuf, 64 + mlen, inner);
    free(ibuf);
    memcpy(obuf, opad, 64);
    memcpy(obuf + 64, inner, 32);
    sha256(obuf, 96, out);
}

static void bounded_copy(char *dest, size_t dest_size, const char *src) {
    size_t n;
    if (!dest || dest_size == 0) return;
    if (!src) { dest[0] = 0; return; }
    n = strlen(src);
    if (n >= dest_size) n = dest_size - 1;
    memcpy(dest, src, n);
    dest[n] = 0;
}

static void xor_buf(char *s, const char *key) {
    size_t i, kn;
    if (!s || !key || !key[0]) return;
    kn = strlen(key);
    for (i = 0; s[i]; i++)
        s[i] = (char)((unsigned char)s[i] ^ (unsigned char)key[i % kn]);
}

struct NlJournal {
    NlJournalEvent ev[NL_JOURNAL_MAX];
    int n;
    uint32_t cursor;
    char mac[NL_JOURNAL_HASH_HEX];
    int signed_ok;
    int sealed;
    char mock_method[80];
    char mock_result[128];
};

NlJournal *nl_journal_create(void) {
    return calloc(1, sizeof(NlJournal));
}

void nl_journal_destroy(NlJournal *j) {
    free(j);
}

int nl_journal_record(NlJournal *j, NlJournalKind kind, const char *trap,
                      const char *capability, const char *method,
                      const char *payload, const char *result,
                      const char *result_schema, uint64_t timing_ns,
                      int generation, const char *impl_version) {
    NlJournalEvent *e;
    const char *pay;
    if (!j || j->n >= NL_JOURNAL_MAX) return NL_JOURNAL_ERR;
    if (j->sealed) return NL_JOURNAL_ERR_SEALED;
    e = &j->ev[j->n];
    memset(e, 0, sizeof(*e));
    e->seq = (uint32_t)j->n;
    e->kind = kind;
    bounded_copy(e->trap, sizeof(e->trap), trap);
    bounded_copy(e->capability, sizeof(e->capability), capability);
    bounded_copy(e->method, sizeof(e->method), method);
    pay = payload ? payload : "";
    if (j->mock_method[0] && method && strcmp(j->mock_method, method) == 0)
        bounded_copy(e->result, sizeof(e->result), j->mock_result);
    else
        bounded_copy(e->result, sizeof(e->result), result);
    bounded_copy(e->payload, sizeof(e->payload), pay);
    nl_sha256_hex((const unsigned char *)pay, strlen(pay), e->arg_hash);
    bounded_copy(e->result_schema, sizeof(e->result_schema),
                 result_schema ? result_schema : "string");
    e->timing_ns = timing_ns;
    e->generation = generation;
    bounded_copy(e->impl_version, sizeof(e->impl_version),
                 impl_version ? impl_version : "0");
    j->n++;
    j->cursor = e->seq;
    j->signed_ok = 0;
    return NL_JOURNAL_OK;
}

int nl_journal_count(const NlJournal *j) {
    return j ? j->n : 0;
}

const NlJournalEvent *nl_journal_event(const NlJournal *j, int i) {
    if (!j || i < 0 || i >= j->n) return NULL;
    return &j->ev[i];
}

int nl_journal_replay(NlJournal *j, uint32_t seq, char *out, size_t n) {
    const NlJournalEvent *e;
    if (!j || !out || n == 0) return NL_JOURNAL_ERR;
    if (seq >= (uint32_t)j->n) return NL_JOURNAL_ERR;
    e = &j->ev[seq];
    if (e->fault) return NL_JOURNAL_ERR_FAULT;
    bounded_copy(out, n, e->result);
    j->cursor = seq;
    return NL_JOURNAL_OK;
}

int nl_journal_validate(const NlJournal *j) {
    int i;
    if (!j) return NL_JOURNAL_ERR;
    for (i = 0; i < j->n; i++) {
        char hex[NL_JOURNAL_HASH_HEX];
        const NlJournalEvent *e = &j->ev[i];
        if (e->seq != (uint32_t)i) return NL_JOURNAL_ERR_ORDER;
        if (!e->redacted) {
            nl_sha256_hex((const unsigned char *)e->payload, strlen(e->payload), hex);
            if (strcmp(hex, e->arg_hash) != 0) return NL_JOURNAL_ERR_ARG;
        }
        if (e->result_schema[0] == 0) return NL_JOURNAL_ERR_SCHEMA;
        if (e->kind == NL_JKIND_SERVICE && e->capability[0] == 0 &&
            strcmp(e->trap, "TRAP_NONE") != 0 &&
            strcmp(e->trap, "TRAP_ERROR") != 0 &&
            strcmp(e->trap, "TRAP_HALT") != 0 &&
            strcmp(e->trap, "TRAP_ASSERT") != 0)
            return NL_JOURNAL_ERR_CAP;
    }
    return NL_JOURNAL_OK;
}

int nl_journal_mock(NlJournal *j, const char *method, const char *result) {
    if (!j || !method) return NL_JOURNAL_ERR;
    bounded_copy(j->mock_method, sizeof(j->mock_method), method);
    bounded_copy(j->mock_result, sizeof(j->mock_result), result ? result : "");
    return NL_JOURNAL_OK;
}

int nl_journal_inject_fault(NlJournal *j, uint32_t seq) {
    if (!j || seq >= (uint32_t)j->n) return NL_JOURNAL_ERR;
    j->ev[seq].fault = 1;
    return NL_JOURNAL_OK;
}

int nl_journal_checkpoint(const NlJournal *j, uint32_t *seq_out) {
    if (!j || !seq_out) return NL_JOURNAL_ERR;
    *seq_out = j->cursor;
    return NL_JOURNAL_OK;
}

int nl_journal_seek(NlJournal *j, uint32_t seq) {
    if (!j || seq >= (uint32_t)j->n) return NL_JOURNAL_ERR;
    j->cursor = seq;
    return NL_JOURNAL_OK;
}

static void serialize(const NlJournal *j, char *buf, size_t n) {
    int i;
    size_t used = 0;
    buf[0] = 0;
    used = (size_t)snprintf(buf, n, "v%d n=%d", NL_JOURNAL_VERSION, j->n);
    for (i = 0; i < j->n && used + 1 < n; i++) {
        const NlJournalEvent *e = &j->ev[i];
        int w = snprintf(buf + used, n - used,
                         "|%u:%d:%s:%s:%s:%s:%s:%s:%u:%d:%s",
                         e->seq, (int)e->kind, e->trap, e->capability, e->method,
                         e->arg_hash, e->result, e->result_schema,
                         (unsigned)e->timing_ns, e->generation, e->impl_version);
        if (w < 0) break;
        used += (size_t)w;
    }
}

int nl_journal_hash(const NlJournal *j, char hex[NL_JOURNAL_HASH_HEX]) {
    char buf[4096];
    if (!j || !hex) return NL_JOURNAL_ERR;
    serialize(j, buf, sizeof(buf));
    nl_sha256_hex((const unsigned char *)buf, strlen(buf), hex);
    return NL_JOURNAL_OK;
}

int nl_journal_sign(NlJournal *j, const char *key) {
    char buf[4096];
    unsigned char mac[32];
    int i;
    if (!j || !key) return NL_JOURNAL_ERR;
    serialize(j, buf, sizeof(buf));
    hmac_sha256((const unsigned char *)key, strlen(key),
                (const unsigned char *)buf, strlen(buf), mac);
    for (i = 0; i < 32; i++)
        snprintf(j->mac + i * 2, 3, "%02x", mac[i]);
    j->mac[64] = 0;
    j->signed_ok = 1;
    return NL_JOURNAL_OK;
}

int nl_journal_verify(const NlJournal *j, const char *key) {
    char buf[4096];
    unsigned char mac[32];
    char hex[NL_JOURNAL_HASH_HEX];
    int i;
    if (!j || !key || !j->signed_ok) return NL_JOURNAL_ERR;
    serialize(j, buf, sizeof(buf));
    hmac_sha256((const unsigned char *)key, strlen(key),
                (const unsigned char *)buf, strlen(buf), mac);
    for (i = 0; i < 32; i++)
        snprintf(hex + i * 2, 3, "%02x", mac[i]);
    hex[64] = 0;
    return strcmp(hex, j->mac) == 0 ? NL_JOURNAL_OK : NL_JOURNAL_ERR;
}

int nl_journal_redact(NlJournal *j, uint32_t seq) {
    if (!j || seq >= (uint32_t)j->n) return NL_JOURNAL_ERR;
    j->ev[seq].payload[0] = 0;
    j->ev[seq].redacted = 1;
    return NL_JOURNAL_OK;
}

char *nl_journal_export_redacted(const NlJournal *j) {
    cJSON *root;
    cJSON *arr;
    char *out;
    int i;
    if (!j) return NULL;
    root = cJSON_CreateObject();
    arr = cJSON_CreateArray();
    cJSON_AddNumberToObject(root, "journal_version", NL_JOURNAL_VERSION);
    for (i = 0; i < j->n; i++) {
        const NlJournalEvent *e = &j->ev[i];
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "seq", e->seq);
        cJSON_AddNumberToObject(o, "kind", (int)e->kind);
        cJSON_AddStringToObject(o, "trap", e->trap);
        cJSON_AddStringToObject(o, "capability", e->capability);
        cJSON_AddStringToObject(o, "method", e->method);
        cJSON_AddStringToObject(o, "arg_hash", e->arg_hash);
        cJSON_AddStringToObject(o, "payload", "");
        cJSON_AddStringToObject(o, "result_schema", e->result_schema);
        cJSON_AddBoolToObject(o, "redacted", 1);
        cJSON_AddItemToArray(arr, o);
    }
    cJSON_AddItemToObject(root, "events", arr);
    out = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return out;
}

int nl_journal_seal(NlJournal *j, const char *key) {
    int i;
    if (!j || !key) return NL_JOURNAL_ERR;
    for (i = 0; i < j->n; i++)
        xor_buf(j->ev[i].payload, key);
    j->sealed = 1;
    return NL_JOURNAL_OK;
}

int nl_journal_open(NlJournal *j, const char *key) {
    int i;
    if (!j || !key) return NL_JOURNAL_ERR;
    for (i = 0; i < j->n; i++)
        xor_buf(j->ev[i].payload, key);
    j->sealed = 0;
    return NL_JOURNAL_OK;
}
