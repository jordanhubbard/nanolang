#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef __APPLE__
static int fail_allocation;
static void *dispatch_malloc(size_t size) {
    return fail_allocation ? NULL : malloc(size);
}
#define malloc dispatch_malloc
#endif
#include "../../modules/dispatch/dispatch.c"
#ifdef __APPLE__
#undef malloc
#endif

int main(void) {
#ifdef __APPLE__
    fail_allocation = 1;
    assert(!nl_queue_serial("nano.fail"));
    assert(!nl_queue_concurrent("nano.fail"));
    assert(!nl_group_create());
    fail_allocation = 0;
    void *serial = nl_queue_serial("nano.recover");
    void *concurrent = nl_queue_concurrent("nano.recover");
    void *group = nl_group_create();
    assert(serial && concurrent && group);
    nl_queue_destroy(serial);
    nl_queue_destroy(concurrent);
    nl_group_destroy(group);
    puts("I passed dispatch wrapper allocation failure and recovery checks.");
#else
    assert(!nl_dispatch_available());
#endif
    return 0;
}
