/* CLI runtime main wrapper - defines g_argc/g_argv and wraps nl_main */
#include <stdint.h>

/* Defined by transpiled code */
extern int64_t nl_main(void);

/* CLI runtime globals */
int g_argc = 0;
char **g_argv = NULL;

/* Main entry point - initializes globals and calls transpiled main */
int main(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
    return (int)nl_main();
}
