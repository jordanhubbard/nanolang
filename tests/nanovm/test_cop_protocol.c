/*
 * test_cop_protocol.c — unit tests for src/nanovm/cop_protocol.c
 *
 * Exercises: cop_serialize_value, cop_deserialize_value (round-trips),
 * cop_send/cop_recv_header/cop_recv_payload/cop_send_simple (via pipes).
 */

#include <stddef.h>  /* NULL */

/* Required by runtime/cli.c */
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
#include <assert.h>
#include <unistd.h>

static int g_pass = 0, g_fail = 0;
#define TEST(name) static void test_##name(void)
#define RUN(name)  do { test_##name(); \
    printf("  %-55s PASS\n", #name "..."); g_pass++; } while(0)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
    g_fail++; return; } } while(0)
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { \
    printf("  FAIL: %s == %s  (%s:%d)\n", #a, #b, __FILE__, __LINE__); \
    g_fail++; return; } } while(0)

/* ── Serialize/Deserialize round-trip helpers ──────────────────────────── */

TEST(serialize_int) {
    uint8_t buf[32];
    NanoValue v = val_int(42);
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 9);  /* 1 tag + 8 bytes */
    ASSERT(buf[0] == TAG_INT);

    VmHeap heap = {0};
    NanoValue out;
    uint32_t consumed = cop_deserialize_value(buf, n, &out, &heap);
    ASSERT(consumed == n);
    ASSERT(out.tag == TAG_INT);
    ASSERT(out.as.i64 == 42);
}

TEST(serialize_negative_int) {
    uint8_t buf[32];
    NanoValue v = val_int(-1000);
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 9);

    VmHeap heap = {0};
    NanoValue out;
    cop_deserialize_value(buf, n, &out, &heap);
    ASSERT(out.tag == TAG_INT);
    ASSERT(out.as.i64 == -1000);
}

TEST(serialize_float) {
    uint8_t buf[32];
    NanoValue v = val_float(3.14);
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 9);  /* 1 tag + 8 bytes */

    VmHeap heap = {0};
    NanoValue out;
    cop_deserialize_value(buf, n, &out, &heap);
    ASSERT(out.tag == TAG_FLOAT);
    ASSERT(out.as.f64 > 3.13 && out.as.f64 < 3.15);
}

TEST(serialize_bool_true) {
    uint8_t buf[8];
    NanoValue v = val_bool(true);
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 2);  /* 1 tag + 1 byte */

    VmHeap heap = {0};
    NanoValue out;
    cop_deserialize_value(buf, n, &out, &heap);
    ASSERT(out.tag == TAG_BOOL);
    ASSERT(out.as.boolean == true);
}

TEST(serialize_bool_false) {
    uint8_t buf[8];
    NanoValue v = val_bool(false);
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 2);

    VmHeap heap = {0};
    NanoValue out;
    cop_deserialize_value(buf, n, &out, &heap);
    ASSERT(out.tag == TAG_BOOL);
    ASSERT(out.as.boolean == false);
}

TEST(serialize_void) {
    uint8_t buf[8];
    NanoValue v = val_void();
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 1);  /* tag only */

    VmHeap heap = {0};
    NanoValue out;
    uint32_t consumed = cop_deserialize_value(buf, n, &out, &heap);
    ASSERT(consumed == 1);
    ASSERT(out.tag == TAG_VOID);
}

TEST(serialize_buffer_too_small) {
    uint8_t buf[4];  /* too small for int (needs 9) */
    NanoValue v = val_int(99);
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 0);  /* fails gracefully */
}

TEST(deserialize_truncated) {
    uint8_t buf[32];
    NanoValue v = val_int(123);
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 9);

    VmHeap heap = {0};
    NanoValue out;
    /* Truncate buffer to 3 bytes (too short for int payload) */
    uint32_t consumed = cop_deserialize_value(buf, 3, &out, &heap);
    ASSERT(consumed == 0);
}

TEST(serialize_opaque) {
    uint8_t buf[32];
    NanoValue v;
    v.tag = TAG_OPAQUE;
    v.as.i64 = 12345;
    uint32_t n = cop_serialize_value(&v, buf, sizeof(buf));
    ASSERT(n == 9);

    VmHeap heap = {0};
    NanoValue out;
    cop_deserialize_value(buf, n, &out, &heap);
    ASSERT(out.tag == TAG_OPAQUE);
    ASSERT(out.as.i64 == 12345);
}

/* ── Pipe-based send/recv tests ────────────────────────────────────────── */

TEST(send_recv_simple) {
    int fds[2];
    ASSERT(pipe(fds) == 0);

    bool sent = cop_send_simple(fds[1], COP_MSG_READY);
    ASSERT(sent);

    CopMsgHeader hdr;
    bool ok = cop_recv_header(fds[0], &hdr);
    ASSERT(ok);
    ASSERT(hdr.msg_type == COP_MSG_READY);
    ASSERT(hdr.payload_len == 0);
    ASSERT(hdr.version == COP_PROTO_VERSION);

    close(fds[0]);
    close(fds[1]);
}

TEST(send_recv_with_payload) {
    int fds[2];
    ASSERT(pipe(fds) == 0);

    const char *payload = "hello world";
    uint32_t len = (uint32_t)strlen(payload);
    bool sent = cop_send(fds[1], COP_MSG_FFI_RESULT, payload, len);
    ASSERT(sent);

    CopMsgHeader hdr;
    bool ok = cop_recv_header(fds[0], &hdr);
    ASSERT(ok);
    ASSERT(hdr.msg_type == COP_MSG_FFI_RESULT);
    ASSERT_EQ(hdr.payload_len, len);

    char buf[64] = {0};
    ok = cop_recv_payload(fds[0], buf, len);
    ASSERT(ok);
    ASSERT(strcmp(buf, "hello world") == 0);

    close(fds[0]);
    close(fds[1]);
}

TEST(recv_header_closed_pipe) {
    int fds[2];
    ASSERT(pipe(fds) == 0);
    close(fds[1]);  /* Close write end — recv should fail */

    CopMsgHeader hdr;
    bool ok = cop_recv_header(fds[0], &hdr);
    ASSERT(!ok);

    close(fds[0]);
}

TEST(send_recv_shutdown_msg) {
    int fds[2];
    ASSERT(pipe(fds) == 0);

    cop_send_simple(fds[1], COP_MSG_SHUTDOWN);

    CopMsgHeader hdr;
    cop_recv_header(fds[0], &hdr);
    ASSERT(hdr.msg_type == COP_MSG_SHUTDOWN);

    close(fds[0]);
    close(fds[1]);
}

/* ── main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[cop_protocol] Co-process protocol tests...\n\n");
    RUN(serialize_int);
    RUN(serialize_negative_int);
    RUN(serialize_float);
    RUN(serialize_bool_true);
    RUN(serialize_bool_false);
    RUN(serialize_void);
    RUN(serialize_buffer_too_small);
    RUN(deserialize_truncated);
    RUN(serialize_opaque);
    RUN(send_recv_simple);
    RUN(send_recv_with_payload);
    RUN(recv_header_closed_pipe);
    RUN(send_recv_shutdown_msg);

    printf("\n");
    if (g_fail == 0) {
        printf("All %d cop_protocol tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d cop_protocol tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
