/* =============================================================================
 * Command-Line Argument Support - C Implementation
 * =============================================================================
 * Provides access to argc/argv for nanolang programs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global storage for argc/argv (set by runtime) */
static int g_argc = 0;
static char **g_argv = NULL;

/* Initialize CLI args (called by generated C main) */
void nl_cli_args_init(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
}

/* Get argument count */
int get_argc(void) {
    return g_argc;
}

/* Get argument by index */
char* get_argv(int index) {
    if (index < 0 || index >= g_argc || !g_argv) {
        return "";
    }
    return g_argv[index];
}
