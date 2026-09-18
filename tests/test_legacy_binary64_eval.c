/* I isolate conversion acceptance from the separately retained full evaluator
 * shadow-gate failure. This driver does not claim that broader gate passes. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main
int main(void) {
    test_eval_binary64_prefix_and_strict_cast();
    return 0;
}
