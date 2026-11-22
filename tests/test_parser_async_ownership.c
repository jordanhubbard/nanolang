/* I isolate the same async/await tree regression with leak checking. */
#define main nano_original_parser_main
#include "test_parser.c"
#undef main
int main(void) { test_free_ast_async_and_await(); return 0; }
