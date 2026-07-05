/*
 * test_ringbuf.c — Unit tests for include/ringbuf.h
 *
 * Tests:
 *   1. push / pop round-trip
 *   2. empty predicate on fresh buffer
 *   3. full predicate after filling buffer
 *   4. push returns -1 when full
 *   5. pop returns -1 when empty
 *   6. wraparound — produce/consume across the index boundary
 *   7. peek does not consume the entry
 *   8. count tracks fill level accurately
 *   9. typed RINGBUF_DEFINE produces independent instances
 *  10. FIFO/count remain correct across repeated wraparound cycles
 *  11. failed push on full buffer preserves count AND FIFO contents
 *  12. failed pop on empty buffer leaves caller output unchanged
 *  13. failed peek on empty buffer leaves caller output unchanged
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Use a small capacity so we can exercise full/wraparound easily */
#define RINGBUF_SIZE 8u
#define MSG_RB_CAPACITY 8u
#define RINGBUF_NO_DEFAULT   /* we define our own typed variants below */
#include "../include/ringbuf.h"

/* -----------------------------------------------------------------------
 * Typed ring buffers used in tests
 * ----------------------------------------------------------------------- */

typedef struct {
    uint32_t id;
    uint8_t  data[4];
} msg_t;

RINGBUF_DEFINE(msg_rb, msg_t, MSG_RB_CAPACITY)

typedef struct {
    uint64_t tick;
} tick_t;

RINGBUF_DEFINE(tick_rb, tick_t, 16)

/* -----------------------------------------------------------------------
 * Minimal test harness
 * ----------------------------------------------------------------------- */

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, desc)                                              \
    do {                                                                \
        if (cond) {                                                     \
            printf("  PASS  %s\n", (desc));                            \
            g_pass++;                                                   \
        } else {                                                        \
            printf("  FAIL  %s  (line %d)\n", (desc), __LINE__);       \
            g_fail++;                                                   \
        }                                                               \
    } while (0)

/* -----------------------------------------------------------------------
 * Test 1: basic push / pop round-trip
 * ----------------------------------------------------------------------- */
static void test_push_pop(void) {
    printf("[1] push/pop round-trip\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    msg_t in  = { .id = 42, .data = {1, 2, 3, 4} };
    msg_t out = { 0 };

    ASSERT(msg_rb_push(&rb, &in)  == 0, "push returns 0");
    ASSERT(msg_rb_pop (&rb, &out) == 0, "pop returns 0");
    ASSERT(out.id == 42,               "id preserved");
    ASSERT(memcmp(out.data, in.data, 4) == 0, "data preserved");
}

/* -----------------------------------------------------------------------
 * Test 2: empty predicate on fresh buffer
 * ----------------------------------------------------------------------- */
static void test_empty(void) {
    printf("[2] empty predicate\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    ASSERT(msg_rb_empty(&rb) == 1, "fresh buffer is empty");
    ASSERT(msg_rb_count(&rb) == 0, "count is 0 on fresh buffer");

    msg_t m = { .id = 1 };
    msg_rb_push(&rb, &m);
    ASSERT(msg_rb_empty(&rb) == 0, "not empty after push");
}

/* -----------------------------------------------------------------------
 * Test 3: full predicate after filling buffer
 * ----------------------------------------------------------------------- */
static void test_full(void) {
    printf("[3] full predicate\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    /* capacity = 8 */
    for (uint32_t i = 0; i < 8; i++) {
        msg_t m = { .id = i };
        ASSERT(msg_rb_push(&rb, &m) == 0, "push succeeds while not full");
    }
    ASSERT(msg_rb_full (&rb) == 1, "buffer is full after 8 pushes");
    ASSERT(msg_rb_count(&rb) == 8, "count equals capacity");
}

/* -----------------------------------------------------------------------
 * Test 4: push returns -1 when full
 * ----------------------------------------------------------------------- */
static void test_push_full(void) {
    printf("[4] push on full buffer\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    msg_t m = { .id = 0 };
    for (int i = 0; i < 8; i++) msg_rb_push(&rb, &m);

    ASSERT(msg_rb_push(&rb, &m) == -1, "push returns -1 when full");
    ASSERT(msg_rb_count(&rb) == 8,    "count unchanged after failed push");
}

/* -----------------------------------------------------------------------
 * Test 5: pop returns -1 when empty
 * ----------------------------------------------------------------------- */
static void test_pop_empty(void) {
    printf("[5] pop on empty buffer\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    msg_t out = { 0 };
    ASSERT(msg_rb_pop(&rb, &out) == -1, "pop returns -1 when empty");
}

/* -----------------------------------------------------------------------
 * Test 6: wraparound — produce/consume across the power-of-2 boundary
 * ----------------------------------------------------------------------- */
static void test_wraparound(void) {
    printf("[6] wraparound\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    /* Fill then drain 5 slots — moves tail to index 5 */
    msg_t m   = { 0 };
    msg_t out = { 0 };
    for (uint32_t i = 0; i < 5; i++) {
        m.id = i;
        msg_rb_push(&rb, &m);
    }
    for (int i = 0; i < 5; i++) msg_rb_pop(&rb, &out);

    /* Now push 8 items; indices will wrap around past 7 */
    for (uint32_t i = 0; i < 8; i++) {
        m.id = 100 + i;
        ASSERT(msg_rb_push(&rb, &m) == 0, "push during wraparound succeeds");
    }
    ASSERT(msg_rb_full(&rb) == 1, "full after wraparound fill");

    /* Drain and verify order */
    int order_ok = 1;
    for (uint32_t i = 0; i < 8; i++) {
        msg_rb_pop(&rb, &out);
        if (out.id != 100 + i) order_ok = 0;
    }
    ASSERT(order_ok,               "FIFO order preserved across wraparound");
    ASSERT(msg_rb_empty(&rb) == 1, "empty after full drain");
}

/* -----------------------------------------------------------------------
 * Test 7: peek does not consume
 * ----------------------------------------------------------------------- */
static void test_peek(void) {
    printf("[7] peek without consume\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    msg_t in  = { .id = 77 };
    msg_t out = { 0 };
    msg_rb_push(&rb, &in);

    ASSERT(msg_rb_peek(&rb, &out) == 0,  "peek returns 0 when non-empty");
    ASSERT(out.id == 77,                  "peek returns correct value");
    ASSERT(msg_rb_count(&rb) == 1,        "peek does not consume — count still 1");

    /* peek on empty */
    msg_rb_pop(&rb, &out);
    ASSERT(msg_rb_peek(&rb, &out) == -1, "peek returns -1 when empty");
}

/* -----------------------------------------------------------------------
 * Test 8: count tracks fill level
 * ----------------------------------------------------------------------- */
static void test_count(void) {
    printf("[8] count\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    msg_t m = { 0 };
    for (uint32_t i = 0; i < 5; i++) {
        m.id = i;
        msg_rb_push(&rb, &m);
        ASSERT((size_t)msg_rb_count(&rb) == (size_t)(i + 1), "count increments on push");
    }
    msg_t out;
    for (int i = 4; i >= 0; i--) {
        msg_rb_pop(&rb, &out);
        ASSERT((size_t)msg_rb_count(&rb) == (size_t)i, "count decrements on pop");
    }
}

/* -----------------------------------------------------------------------
 * Test 9: two independent typed ring buffers do not alias
 * ----------------------------------------------------------------------- */
static void test_independent_types(void) {
    printf("[9] independent typed instances\n");
    msg_rb_t  mrb;
    tick_rb_t trb;
    msg_rb_init (&mrb);
    tick_rb_init(&trb);

    msg_t  m = { .id = 55 };
    tick_t t = { .tick = 0xDEADBEEF };

    msg_rb_push (&mrb, &m);
    tick_rb_push(&trb, &t);

    ASSERT(msg_rb_count (&mrb) == 1, "msg  ring count = 1");
    ASSERT(tick_rb_count(&trb) == 1, "tick ring count = 1");

    tick_t tout = { 0 };
    msg_t  mout = { 0 };
    tick_rb_pop(&trb, &tout);
    msg_rb_pop (&mrb, &mout);

    ASSERT(mout.id   == 55,          "msg  value correct");
    ASSERT(tout.tick == 0xDEADBEEF,  "tick value correct");
}

/* -----------------------------------------------------------------------
 * Test 10: repeated wraparound cycles with interleaved push/pop
 * ----------------------------------------------------------------------- */
static void test_multi_wraparound_cycles(void) {
    printf("[10] multi-cycle wraparound FIFO/count\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    msg_t in  = { 0 };
    msg_t out = { 0 };

    /*
     * Start away from index 0 so every capacity-sized cycle crosses the
     * physical array boundary instead of merely ending at it.
     */
    for (uint32_t i = 0; i < 5; i++) {
        in.id = 500 + i;
        ASSERT(msg_rb_push(&rb, &in) == 0, "precondition push succeeds");
        ASSERT(msg_rb_pop(&rb, &out) == 0,  "precondition pop succeeds");
        ASSERT(out.id == 500 + i,           "precondition FIFO order");
    }
    ASSERT(msg_rb_empty(&rb) == 1, "precondition leaves buffer empty");
    ASSERT(msg_rb_count(&rb) == 0, "precondition leaves count at 0");

    uint32_t next_id = 1000;
    uint32_t expected_id = next_id;
    uint32_t expected_index = 5u;

    for (uint32_t cycle = 0; cycle < 3; cycle++) {
        size_t expected_count = 0;
        char desc[96];

        for (uint32_t pair = 0; pair < MSG_RB_CAPACITY / 2u; pair++) {
            in.id = next_id++;
            ASSERT(msg_rb_push(&rb, &in) == 0, "cycle first push succeeds");
            expected_count++;
            ASSERT(msg_rb_count(&rb) == expected_count, "count tracks first push");

            in.id = next_id++;
            ASSERT(msg_rb_push(&rb, &in) == 0, "cycle second push succeeds");
            expected_count++;
            ASSERT(msg_rb_count(&rb) == expected_count, "count tracks second push");

            ASSERT(msg_rb_pop(&rb, &out) == 0, "cycle interleaved pop succeeds");
            ASSERT(out.id == expected_id++,    "FIFO order during interleaved pop");
            expected_count--;
            ASSERT(msg_rb_count(&rb) == expected_count, "count tracks interleaved pop");
        }

        while (expected_count > 0) {
            ASSERT(msg_rb_pop(&rb, &out) == 0, "cycle drain pop succeeds");
            ASSERT(out.id == expected_id++,    "FIFO order during cycle drain");
            expected_count--;
            ASSERT(msg_rb_count(&rb) == expected_count, "count tracks cycle drain");
        }

        expected_index += MSG_RB_CAPACITY;
        snprintf(desc, sizeof(desc), "cycle %u head advanced one full wrap", cycle + 1u);
        ASSERT(rb.head == expected_index, desc);
        snprintf(desc, sizeof(desc), "cycle %u tail advanced one full wrap", cycle + 1u);
        ASSERT(rb.tail == expected_index, desc);
        snprintf(desc, sizeof(desc), "cycle %u empty after complete wrap", cycle + 1u);
        ASSERT(msg_rb_empty(&rb) == 1, desc);
        snprintf(desc, sizeof(desc), "cycle %u count reset after complete wrap", cycle + 1u);
        ASSERT(msg_rb_count(&rb) == 0, desc);
        snprintf(desc, sizeof(desc), "cycle %u not full after complete wrap", cycle + 1u);
        ASSERT(msg_rb_full(&rb) == 0, desc);
    }

    ASSERT(expected_id == next_id, "all cycle items consumed in FIFO order");
}

/* -----------------------------------------------------------------------
 * Test 11: failed push on full buffer preserves count AND FIFO contents
 *
 * Regression: a buggy implementation might silently overwrite the oldest
 * entry (converting to a lossy overwrite ring).  Verify that after a
 * rejected push, every item drained in FIFO order is the one that was
 * originally enqueued, not the rejected payload.
 * ----------------------------------------------------------------------- */
static void test_push_full_preserves_fifo(void) {
    printf("[11] failed push on full buffer preserves count and FIFO\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    /* Fill with IDs 0..7 */
    uint32_t i;
    for (i = 0; i < 8u; i++) {
        msg_t m = { .id = i, .data = {(uint8_t)i, 0, 0, 0} };
        msg_rb_push(&rb, &m);
    }

    /* Attempt to push a sentinel that must NOT appear in a subsequent drain */
    msg_t sentinel = { .id = 0xDEADu, .data = {0xFF, 0xFF, 0xFF, 0xFF} };
    int rc = msg_rb_push(&rb, &sentinel);
    ASSERT(rc == -1, "push on full returns -1");
    ASSERT(msg_rb_count(&rb) == 8, "count unchanged after rejected push");

    /* Drain and verify original FIFO order — sentinel must not appear */
    int fifo_ok   = 1;
    int no_sentinel = 1;
    for (i = 0; i < 8u; i++) {
        msg_t out = { 0 };
        msg_rb_pop(&rb, &out);
        if (out.id != i)       fifo_ok    = 0;
        if (out.id == 0xDEADu) no_sentinel = 0;
    }
    ASSERT(fifo_ok,    "FIFO contents intact after rejected push");
    ASSERT(no_sentinel, "sentinel value absent — no overwrite occurred");
    ASSERT(msg_rb_empty(&rb) == 1, "buffer empty after full drain");
}

/* -----------------------------------------------------------------------
 * Test 12: failed pop on empty buffer leaves caller output unchanged
 *
 * Regression: a buggy pop might write a zero-struct (or garbage) into
 * *out even when returning -1.  The caller's variable must be bit-for-bit
 * identical to its value before the failed pop.
 * ----------------------------------------------------------------------- */
static void test_pop_empty_output_unchanged(void) {
    printf("[12] failed pop on empty buffer leaves caller output unchanged\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    /* out is initialised to a known sentinel pattern */
    msg_t out;
    out.id      = 0xCAFEBABEu;
    out.data[0] = 0xAA;
    out.data[1] = 0xBB;
    out.data[2] = 0xCC;
    out.data[3] = 0xDD;

    int rc = msg_rb_pop(&rb, &out);
    ASSERT(rc == -1, "pop on empty returns -1");
    ASSERT(out.id      == 0xCAFEBABEu, "out.id unchanged after failed pop");
    ASSERT(out.data[0] == 0xAA,        "out.data[0] unchanged after failed pop");
    ASSERT(out.data[1] == 0xBB,        "out.data[1] unchanged after failed pop");
    ASSERT(out.data[2] == 0xCC,        "out.data[2] unchanged after failed pop");
    ASSERT(out.data[3] == 0xDD,        "out.data[3] unchanged after failed pop");
}

/* -----------------------------------------------------------------------
 * Test 13: failed peek on empty buffer leaves caller output unchanged
 *
 * Mirrors test 12 for peek: a failed peek must not clobber *out.
 * ----------------------------------------------------------------------- */
static void test_peek_empty_output_unchanged(void) {
    printf("[13] failed peek on empty buffer leaves caller output unchanged\n");
    msg_rb_t rb;
    msg_rb_init(&rb);

    /* out is initialised to a known sentinel pattern */
    msg_t out;
    out.id      = 0xDEADC0DEu;
    out.data[0] = 0x11;
    out.data[1] = 0x22;
    out.data[2] = 0x33;
    out.data[3] = 0x44;

    int rc = msg_rb_peek(&rb, &out);
    ASSERT(rc == -1, "peek on empty returns -1");
    ASSERT(out.id      == 0xDEADC0DEu, "out.id unchanged after failed peek");
    ASSERT(out.data[0] == 0x11,         "out.data[0] unchanged after failed peek");
    ASSERT(out.data[1] == 0x22,         "out.data[1] unchanged after failed peek");
    ASSERT(out.data[2] == 0x33,         "out.data[2] unchanged after failed peek");
    ASSERT(out.data[3] == 0x44,         "out.data[3] unchanged after failed peek");
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(void) {
    printf("=== ringbuf tests ===\n");
    test_push_pop();
    test_empty();
    test_full();
    test_push_full();
    test_pop_empty();
    test_wraparound();
    test_peek();
    test_count();
    test_independent_types();
    test_multi_wraparound_cycles();
    test_push_full_preserves_fifo();
    test_pop_empty_output_unchanged();
    test_peek_empty_output_unchanged();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
