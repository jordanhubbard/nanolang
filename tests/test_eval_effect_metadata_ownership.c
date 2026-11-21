/* I execute every original effect/handler method without changing assertions. */
#define main nano_original_eval_main
#include "test_eval.c"
#undef main

int main(void) {
    ASSERT(setvbuf(stdout, NULL, _IONBF, 0) == 0);
    test_eval_effects_basic();
    test_eval_unqualified_effect_handler();
    test_eval_nested_effect_handlers();
    test_eval_effect_argument_lists();
    test_eval_handler_return_destination();
    test_eval_handler_return_expression_order();
    test_eval_handler_final_value_resumes();
    test_eval_handler_return_nested_and_string();
    test_eval_handler_return_recursive_activation();
    test_eval_handler_return_partial_literal_cleanup();
    test_eval_handler_return_higher_order();
    test_eval_handler_return_async_calls();
    puts("I executed my original effect and handler ownership controls.");
    return 0;
}
