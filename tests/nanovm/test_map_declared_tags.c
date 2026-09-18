/* I reuse the ordinary VM API acceptance under a focused sanitizer entry. */
#define main nanolang_full_vm_main
#include "test_vm.c"
#undef main

int main(void) {
    test_hashmap_declared_write_tags();
    printf("I passed %d/%d declared map ownership checks.\n", tests_passed, tests_run);
    return tests_failed != 0;
}
