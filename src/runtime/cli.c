/* =============================================================================
 * Command-Line Argument Runtime Support
 * =============================================================================
 * Provides argc/argv access to transpiled NanoLang programs
 */

#include <stdint.h>

/* These are set by main.c during program startup */
extern int g_argc;
extern char **g_argv;

/* Get number of command-line arguments */
int64_t get_argc(void) {
    return (int64_t)g_argc;
}

/* Get command-line argument by index */
const char* get_argv(int64_t index) {
    if (index < 0 || index >= g_argc) {
        return "";
    }
    return g_argv[index];
}

/* Wrapper for system() to avoid conflicts with stdlib.h declaration */
int64_t nl_system_exec(const char* command) {
    extern int system(const char*);  /* Forward declaration */
    return (int64_t)system(command);
}
