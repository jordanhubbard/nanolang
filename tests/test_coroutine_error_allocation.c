/* I isolate actual scheduler error-text allocations from graph ownership hooks. */
#include "../src/coroutine.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t attempts, failures, live;
static bool fail;
static void *message_alloc(size_t bytes) {
    ++attempts;
    if (fail) { ++failures; return NULL; }
    void *result = malloc(bytes);
    if (result) ++live;
    return result;
}
static void message_free(void *pointer) {
    if (pointer) { assert(live); --live; }
    free(pointer);
}
#define malloc message_alloc
#define free message_free
#include "../src/coroutine.c"
#undef malloc
#undef free

static Value error_callback(void *argument, int id) {
    assert(argument == NULL);
    nano_coro_error("first message");
    nano_coro_error("second message");
    assert(!nano_coro_release(id) && !nano_coro_cancel(id));
    Value result = {0}; result.type = VAL_VOID; return result;
}
static void attempt(bool cancel, bool allocation_failure, bool await) {
    assert(live == 0);
    attempts = failures = 0; fail = allocation_failure;
    int id = nano_coro_spawn(error_callback, NULL);
    assert(id >= 0 && attempts == 0);
    if (cancel) assert(nano_coro_cancel(id));
    else if (await) (void)nano_coro_await_id(id);
    else assert(nano_scheduler_step());
    NanoCoroutine *task = coro_by_id(id);
    assert(task && !task->active && task->status == CORO_ERROR);
    assert(attempts == 1 && failures == (size_t)allocation_failure);
    if (allocation_failure) assert(task->error_msg == NULL && live == 0);
    else {
        assert(live == 1 && task->error_msg != NULL);
        assert(!strcmp(task->error_msg, cancel ? "I cancelled this pending task." : "first message"));
    }
    assert(nano_coro_release(id) && live == 0);
    assert(!nano_coro_release(id));
}
int main(void) {
    nano_scheduler_init();
    for (int route = 0; route < 3; ++route) {
        attempt(route == 0, false, route == 2);
        attempt(route == 0, true, route == 2);
        attempt(route == 0, false, route == 2);
    }
    puts("I checked scheduler error-text allocation, release and fresh recovery.");
    return 0;
}
