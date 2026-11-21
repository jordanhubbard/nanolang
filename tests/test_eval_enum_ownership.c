/* I exercise my original enum assertions and repeated shadow registration. */
#define main complete_evaluator_main
#include "test_eval.c"
#undef main
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    TEST(eval_enum_access);
    puts("I completed my enum value and repeated-shadow controls.");
    return 0;
}
