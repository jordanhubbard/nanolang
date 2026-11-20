/* I reuse the real evaluator fixture with a bounded ownership-only entry. */
#define main nano_original_eval_main
#include "test_eval.c"
#undef main

int main(void) {
    test_eval_global_binding_scalar_ownership();
    puts("I retain global binding identity and caller STRING ownership.");
    return 0;
}
