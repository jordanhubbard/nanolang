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

#include <stdlib.h>

/* Wrapper for system() to avoid conflicts with stdlib.h declaration */
int64_t nl_os_system(const char* command) {
    return (int64_t)system(command);
}

/* Wrapper for getenv() */
const char* nl_os_getenv(const char* name) {
    const char* val = getenv(name);
    return val ? val : "";
}

/* Wrapper for setenv() */
int64_t nl_os_setenv(const char* name, const char* value, int64_t overwrite) {
    return (int64_t)setenv(name, value, (int)overwrite);
}

/* Wrapper for unsetenv() */
int64_t nl_os_unsetenv(const char* name) {
    return (int64_t)unsetenv(name);
}
