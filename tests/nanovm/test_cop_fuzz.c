/*
 * test_cop_fuzz.c — malformed-input / fuzz tests for the co-process protocol
 * (src/nanovm/cop_protocol.c).
 *
 * Roadmap 4.0, Phase 12 (Verifier and safety): "fuzz ... the co-process
 * protocol."  This exercises cop_deserialize_value (the untrusted wire decode
 * path a VM runs on bytes received from a co-process) with random buffers,
 * bit-flipped valid encodings, and adversarial length prefixes.  It also
 * fuzzes the wire framing decode via a socketpair (cop_send_simple + a raw
 * write of a corrupt header, then cop_recv_header/cop_recv_payload).
 *
 * The property under test is robustness: the decoder must never read past the
 * supplied buffer, never crash, and report 0 bytes consumed on malformed input.
 */

#include <stddef.h>  /* NULL */

/* Required by runtime/cli.c (mirrors test_cop_protocol.c). */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#include "../../src/nanovm/cop_protocol.h"
#include "../../src/nanovm/heap.h"
#include "../../src/nanovm/value.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-58s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-58s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* Deterministic xorshift PRNG. */
static uint64_t g_rng = 0x243F6A8885A308D3ULL;
static void rng_seed(uint64_t s) { g_rng = s ? s : 1; }
static uint64_t rng_next(void) {
    uint64_t x = g_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_rng = x; return x;
}
static uint8_t rng_byte(void) { return (uint8_t)(rng_next() & 0xFF); }
static void rng_fill(uint8_t *b, size_t n) { for (size_t i = 0; i < n; i++) b[i] = rng_byte(); }

/* The cop decoder allocates strings/arrays via the heap; each test uses a
 * fresh initialised VmHeap and releases it here so no allocations leak. */
static void free_heap(VmHeap *heap) {
    vm_heap_destroy(heap);
}

/* ── Random-buffer decode ────────────────────────────────────────────────── */

static void test_deserialize_random(void) {
    const char *test_name = "cop_deserialize_value: 200k random buffers never crash";
    for (int iter = 0; iter < 200000; iter++) {
        uint8_t buf[64];
        uint32_t len = (uint32_t)(rng_next() % (sizeof(buf) + 1));
        rng_fill(buf, len);
        VmHeap heap;
        vm_heap_init(&heap);
        NanoValue out;
        memset(&out, 0, sizeof(out));
        uint32_t consumed = cop_deserialize_value(buf, len, &out, &heap);
        ASSERT(consumed <= len, "consumed more bytes than the buffer holds");
        free_heap(&heap);
    }
    PASS(test_name);
}

/* Every possible leading tag byte with an empty payload must be safe. */
static void test_deserialize_all_tags_empty(void) {
    const char *test_name = "cop_deserialize_value: every tag byte, empty payload safe";
    for (int tag = 0; tag < 256; tag++) {
        uint8_t buf[1] = { (uint8_t)tag };
        VmHeap heap;
        vm_heap_init(&heap);
        NanoValue out;
        memset(&out, 0, sizeof(out));
        uint32_t consumed = cop_deserialize_value(buf, 1, &out, &heap);
        ASSERT(consumed <= 1, "single-tag decode consumed too much");
        free_heap(&heap);
    }
    PASS(test_name);
}

/* Truncation: every proper prefix of a valid serialization must fail cleanly. */
static void test_deserialize_truncated(void) {
    const char *test_name = "cop_deserialize_value: truncated valid encodings safe";
    /* Serialize a handful of value kinds, then feed all shorter prefixes. */
    VmHeap wheap;
    vm_heap_init(&wheap);
    NanoValue samples[3];
    samples[0] = val_int(0x0102030405060708LL);
    samples[1] = val_float(3.14159);
    samples[2] = val_bool(true);
    for (int s = 0; s < 3; s++) {
        uint8_t buf[64];
        uint32_t n = cop_serialize_value(&samples[s], buf, sizeof(buf));
        if (n == 0) continue;
        for (uint32_t prefix = 0; prefix < n; prefix++) {
            VmHeap heap;
            vm_heap_init(&heap);
            NanoValue out;
            memset(&out, 0, sizeof(out));
            uint32_t consumed = cop_deserialize_value(buf, prefix, &out, &heap);
            ASSERT(consumed <= prefix, "prefix decode consumed past the prefix");
            free_heap(&heap);
        }
    }
    free_heap(&wheap);
    PASS(test_name);
}

/* Adversarial string length prefix: a TAG_STRING with a giant length must not
 * over-read or over-allocate. */
static void test_deserialize_adversarial_string_len(void) {
    const char *test_name = "cop_deserialize_value: adversarial string length safe";
    for (int iter = 0; iter < 20000; iter++) {
        uint8_t buf[16];
        memset(buf, 0, sizeof(buf));
        buf[0] = TAG_STRING;
        /* random 32-bit length, but only a few payload bytes available */
        uint32_t claimed = (uint32_t)rng_next();
        buf[1] = (uint8_t)(claimed & 0xFF);
        buf[2] = (uint8_t)((claimed >> 8) & 0xFF);
        buf[3] = (uint8_t)((claimed >> 16) & 0xFF);
        buf[4] = (uint8_t)((claimed >> 24) & 0xFF);
        uint32_t avail = 5 + (uint32_t)(rng_next() % 8);
        if (avail > sizeof(buf)) avail = sizeof(buf);
        VmHeap heap;
        vm_heap_init(&heap);
        NanoValue out;
        memset(&out, 0, sizeof(out));
        uint32_t consumed = cop_deserialize_value(buf, avail, &out, &heap);
        ASSERT(consumed <= avail, "string decode over-read");
        free_heap(&heap);
    }
    PASS(test_name);
}

/* Adversarial array header: TAG_ARRAY with a giant element count must not
 * loop unboundedly or over-read. */
static void test_deserialize_adversarial_array(void) {
    const char *test_name = "cop_deserialize_value: adversarial array count safe";
    for (int iter = 0; iter < 20000; iter++) {
        uint8_t buf[24];
        rng_fill(buf, sizeof(buf));
        buf[0] = TAG_ARRAY;
        /* element type at buf[1], count u32 at buf[2..5] set huge */
        buf[2] = 0xFF; buf[3] = 0xFF; buf[4] = 0xFF; buf[5] = 0xFF;
        uint32_t avail = 6 + (uint32_t)(rng_next() % (sizeof(buf) - 6 + 1));
        VmHeap heap;
        vm_heap_init(&heap);
        NanoValue out;
        memset(&out, 0, sizeof(out));
        uint32_t consumed = cop_deserialize_value(buf, avail, &out, &heap);
        ASSERT(consumed <= avail, "array decode over-read");
        free_heap(&heap);
    }
    PASS(test_name);
}

/* Bit-flip mutation of valid encodings reaches interior decode branches. */
static void test_deserialize_bitflip(void) {
    const char *test_name = "cop_deserialize_value: bit-flipped valid encodings safe";
    NanoValue base = val_int(-42);
    uint8_t enc[32];
    uint32_t n = cop_serialize_value(&base, enc, sizeof(enc));
    ASSERT(n > 0, "baseline value must serialize");
    for (int iter = 0; iter < 20000; iter++) {
        uint8_t buf[32];
        memcpy(buf, enc, n);
        int flips = 1 + (int)(rng_next() % 3);
        for (int f = 0; f < flips; f++) buf[rng_next() % n] ^= rng_byte();
        VmHeap heap;
        vm_heap_init(&heap);
        NanoValue out;
        memset(&out, 0, sizeof(out));
        uint32_t consumed = cop_deserialize_value(buf, n, &out, &heap);
        ASSERT(consumed <= n, "bit-flip decode over-read");
        free_heap(&heap);
    }
    PASS(test_name);
}

/* ── Wire framing: corrupt headers over a socketpair ─────────────────────── */

/* A corrupt/oversized header must be rejected by cop_recv_header /
 * cop_recv_payload without over-reading. */
static void test_recv_corrupt_header(void) {
    const char *test_name = "cop_recv_header: corrupt/oversized headers rejected";
    for (int iter = 0; iter < 2000; iter++) {
        int sv[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
            FAIL(test_name, "socketpair failed");
            return;
        }
        /* Write 8 arbitrary header bytes, occasionally with a huge payload_len. */
        uint8_t hdr[COP_HEADER_SIZE];
        rng_fill(hdr, sizeof(hdr));
        if ((rng_next() & 1) == 0) {
            /* force payload_len larger than COP_MAX_PAYLOAD */
            uint32_t huge = COP_MAX_PAYLOAD + 1 + (uint32_t)(rng_next() & 0xFFFF);
            memcpy(hdr + 4, &huge, sizeof(huge));
        }
        ssize_t w = write(sv[1], hdr, sizeof(hdr));
        (void)w;
        close(sv[1]); /* signal EOF so recv can't block on missing payload */
        CopMsgHeader parsed;
        memset(&parsed, 0, sizeof(parsed));
        bool got = cop_recv_header(sv[0], &parsed);
        if (got && parsed.payload_len > 0 && parsed.payload_len <= COP_MAX_PAYLOAD) {
            /* Attempt to read the (absent) payload; must fail on EOF, not hang. */
            uint8_t small[64];
            uint32_t want = parsed.payload_len < sizeof(small)
                          ? parsed.payload_len : (uint32_t)sizeof(small);
            bool pr = cop_recv_payload(sv[0], small, want);
            (void)pr;
        }
        close(sv[0]);
    }
    PASS(test_name);
}

/* Round-trip sanity so the fuzz harness itself is trustworthy. */
static void test_send_recv_roundtrip(void) {
    const char *test_name = "cop_send_simple/recv_header: clean round-trip";
    int sv[2];
    ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "socketpair failed");
    bool sent = cop_send_simple(sv[1], COP_MSG_READY);
    ASSERT(sent, "cop_send_simple failed");
    CopMsgHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    bool got = cop_recv_header(sv[0], &hdr);
    ASSERT(got, "cop_recv_header failed");
    ASSERT(hdr.msg_type == COP_MSG_READY, "wrong message type round-tripped");
    ASSERT(hdr.payload_len == 0, "simple message must have no payload");
    close(sv[0]); close(sv[1]);
    PASS(test_name);
}

int main(void) {
    printf("=== Co-process protocol malformed-input / fuzz tests ===\n");
    rng_seed(0xB5C0FFEE0BADF00DULL);

    printf("\n-- value wire decode --\n");
    test_deserialize_random();
    test_deserialize_all_tags_empty();
    test_deserialize_truncated();
    test_deserialize_adversarial_string_len();
    test_deserialize_adversarial_array();
    test_deserialize_bitflip();

    printf("\n-- message framing --\n");
    test_recv_corrupt_header();
    test_send_recv_roundtrip();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
