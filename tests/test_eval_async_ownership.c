/* I execute my original async-handler assertions without changing selection. */
#define main nano_original_eval_main
#include "test_eval.c"
#undef main
int main(void) { test_eval_handler_return_async_calls(); return 0; }
