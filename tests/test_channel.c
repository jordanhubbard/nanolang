/*
 * test_channel.c — unit tests for channel.c (CSP channel primitives)
 *
 * Tests channel lifecycle, buffered/unbuffered send/recv, select, and registry.
 */

#include "../src/channel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_channel_new_buffered(void) {
    const char *test_name = "channel_new: buffered channel allocated";
    Channel *ch = channel_new(8);
    ASSERT(ch != NULL, "channel_new should return non-NULL for capacity 8");
    ASSERT(!channel_is_closed(ch), "new channel should not be closed");
    channel_free(ch);
    PASS(test_name);
}

static void test_channel_new_unbuffered(void) {
    const char *test_name = "channel_new: unbuffered channel allocated";
    Channel *ch = channel_new(0);
    ASSERT(ch != NULL, "channel_new should return non-NULL for capacity 0");
    channel_free(ch);
    PASS(test_name);
}

static void test_channel_new_zero_capacity(void) {
    const char *test_name = "channel_new: capacity 0 is unbuffered";
    Channel *ch = channel_new(0);
    ASSERT(ch != NULL, "channel_new(0) should succeed");
    /* Unbuffered: try_send should return false (no waiting receiver) */
    bool sent = channel_try_send(ch, 42);
    ASSERT(!sent, "unbuffered channel try_send should fail with no receiver");
    channel_free(ch);
    PASS(test_name);
}

static void test_channel_close(void) {
    const char *test_name = "channel_close: marks channel closed";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");
    ASSERT(!channel_is_closed(ch), "channel should start open");
    channel_close(ch);
    ASSERT(channel_is_closed(ch), "channel should be closed after close");
    channel_free(ch);
    PASS(test_name);
}

static void test_channel_free_null(void) {
    const char *test_name = "channel_free: NULL is safe";
    channel_free(NULL); /* Should not crash */
    PASS(test_name);
}

static void test_channel_try_send_recv_buffered(void) {
    const char *test_name = "channel_try_send/recv: buffered round-trip";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    bool sent = channel_try_send(ch, 100);
    ASSERT(sent, "try_send should succeed on buffered channel with space");

    ChanVal val = 0;
    bool recvd = channel_try_recv(ch, &val);
    ASSERT(recvd, "try_recv should succeed after send");
    ASSERT(val == 100, "received value should match sent value");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_try_recv_empty(void) {
    const char *test_name = "channel_try_recv: empty channel returns false";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    ChanVal val = -1;
    bool recvd = channel_try_recv(ch, &val);
    ASSERT(!recvd, "try_recv on empty channel should return false");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_try_send_full(void) {
    const char *test_name = "channel_try_send: full channel returns false";
    Channel *ch = channel_new(2);
    ASSERT(ch != NULL, "channel_new should succeed");

    bool s1 = channel_try_send(ch, 1);
    bool s2 = channel_try_send(ch, 2);
    bool s3 = channel_try_send(ch, 3); /* Should fail — full */
    ASSERT(s1, "first send should succeed");
    ASSERT(s2, "second send should succeed");
    ASSERT(!s3, "third send on full channel should fail");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_multiple_values(void) {
    const char *test_name = "channel: send/recv multiple values in order";
    Channel *ch = channel_new(8);
    ASSERT(ch != NULL, "channel_new should succeed");

    for (int i = 0; i < 5; i++) {
        bool sent = channel_try_send(ch, (ChanVal)i * 10);
        ASSERT(sent, "each send should succeed");
    }

    for (int i = 0; i < 5; i++) {
        ChanVal val = -1;
        bool recvd = channel_try_recv(ch, &val);
        ASSERT(recvd, "each recv should succeed");
        ASSERT(val == (ChanVal)i * 10, "values should be in FIFO order");
    }

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_blocking_send(void) {
    const char *test_name = "channel_send: buffered blocking send succeeds";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    /* Blocking send with TASK_NONE (main context) on buffered channel */
    bool ok = channel_send(ch, 42, TASK_NONE);
    ASSERT(ok, "channel_send should succeed on buffered channel with space");

    ChanVal val = 0;
    bool recvd = channel_try_recv(ch, &val);
    ASSERT(recvd, "should receive the sent value");
    ASSERT(val == 42, "received value should be 42");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_blocking_recv(void) {
    const char *test_name = "channel_recv: buffered blocking recv succeeds";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    channel_try_send(ch, 99);

    ChanVal val = 0;
    bool ok = channel_recv(ch, &val, TASK_NONE);
    ASSERT(ok, "channel_recv should succeed when data is available");
    ASSERT(val == 99, "received value should be 99");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_recv_closed_empty(void) {
    const char *test_name = "channel_recv: closed+empty channel returns false";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    channel_close(ch);
    ChanVal val = -1;
    bool ok = channel_recv(ch, &val, TASK_NONE);
    ASSERT(!ok, "recv on closed empty channel should return false");
    ASSERT(val == 0, "val should be 0 on closed empty channel");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_select_ready_recv(void) {
    const char *test_name = "channel_select: selects ready recv case";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    channel_try_send(ch, 77);

    ChanVal out = 0;
    SelectCase cases[1] = {{
        .kind = SELECT_RECV,
        .ch = ch,
        .send_val = 0,
        .recv_out = &out,
        .ready = false
    }};
    int idx = channel_select(cases, 1);
    ASSERT(idx == 0, "select should return index 0 for ready recv case");
    ASSERT(out == 77, "select should deliver the received value");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_select_ready_send(void) {
    const char *test_name = "channel_select: selects ready send case";
    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    SelectCase cases[1] = {{
        .kind = SELECT_SEND,
        .ch = ch,
        .send_val = 55,
        .recv_out = NULL,
        .ready = false
    }};
    int idx = channel_select(cases, 1);
    ASSERT(idx == 0, "select should return index 0 for ready send case");

    /* Verify value was sent */
    ChanVal val = 0;
    bool recvd = channel_try_recv(ch, &val);
    ASSERT(recvd, "should receive the selected send value");
    ASSERT(val == 55, "value should be 55");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_select_default(void) {
    const char *test_name = "channel_select: default case when nothing ready";
    Channel *ch = channel_new(0); /* unbuffered, no receiver */
    ASSERT(ch != NULL, "channel_new should succeed");

    SelectCase cases[2] = {
        { .kind = SELECT_SEND, .ch = ch, .send_val = 1, .recv_out = NULL, .ready = false },
        { .kind = SELECT_DEFAULT, .ch = NULL, .send_val = 0, .recv_out = NULL, .ready = false }
    };
    int idx = channel_select(cases, 2);
    ASSERT(idx == 1, "select should return default case index when send not ready");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_select_none_ready(void) {
    const char *test_name = "channel_select: returns -1 when nothing ready (no default)";
    Channel *ch = channel_new(0); /* unbuffered */
    ASSERT(ch != NULL, "channel_new should succeed");

    SelectCase cases[1] = {{
        .kind = SELECT_SEND,
        .ch = ch,
        .send_val = 1,
        .recv_out = NULL,
        .ready = false
    }};
    int idx = channel_select(cases, 1);
    ASSERT(idx == -1, "select should return -1 when no case is ready");

    channel_free(ch);
    PASS(test_name);
}

static void test_channel_registry_new_free(void) {
    const char *test_name = "channel_registry: new and free";
    ChannelRegistry *reg = channel_registry_new();
    ASSERT(reg != NULL, "channel_registry_new should return non-NULL");
    channel_registry_free(reg);
    PASS(test_name);
}

static void test_channel_registry_add_get(void) {
    const char *test_name = "channel_registry: add and get by handle";
    ChannelRegistry *reg = channel_registry_new();
    ASSERT(reg != NULL, "registry_new should succeed");

    Channel *ch = channel_new(4);
    ASSERT(ch != NULL, "channel_new should succeed");

    int handle = channel_registry_add(reg, ch);
    ASSERT(handle >= 0, "add should return a non-negative handle");

    Channel *got = channel_registry_get(reg, handle);
    ASSERT(got == ch, "get should return the same channel pointer");

    channel_registry_free(reg); /* frees all channels in registry */
    PASS(test_name);
}

static void test_channel_registry_get_invalid(void) {
    const char *test_name = "channel_registry: get invalid handle returns NULL";
    ChannelRegistry *reg = channel_registry_new();
    ASSERT(reg != NULL, "registry_new should succeed");

    Channel *got = channel_registry_get(reg, 999);
    ASSERT(got == NULL, "get with invalid handle should return NULL");

    channel_registry_free(reg);
    PASS(test_name);
}

static void test_channel_registry_free_null(void) {
    const char *test_name = "channel_registry_free: NULL is safe";
    channel_registry_free(NULL);
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[channel] Channel concurrency primitive tests...\n\n");

    test_channel_new_buffered();
    test_channel_new_unbuffered();
    test_channel_new_zero_capacity();
    test_channel_close();
    test_channel_free_null();
    test_channel_try_send_recv_buffered();
    test_channel_try_recv_empty();
    test_channel_try_send_full();
    test_channel_multiple_values();
    test_channel_blocking_send();
    test_channel_blocking_recv();
    test_channel_recv_closed_empty();
    test_channel_select_ready_recv();
    test_channel_select_ready_send();
    test_channel_select_default();
    test_channel_select_none_ready();
    test_channel_registry_new_free();
    test_channel_registry_add_get();
    test_channel_registry_get_invalid();
    test_channel_registry_free_null();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
