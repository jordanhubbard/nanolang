/* repl_main.c — Entry point for the standalone nanorepl binary.
 *
 * The REPL logic lives in repl.c; this file only provides main() and the
 * project-root resolver that mirrors nano_main.c.
 */

#define _POSIX_C_SOURCE 200809L

#include "nanolang.h"
#include <limits.h>
#include <string.h>
#include <unistd.h>

/* Globals declared in repl.c / shared with the rest of the interpreter */
int   g_argc = 0;
char **g_argv = NULL;
char  g_project_root[PATH_MAX] = "";

const char *get_project_root(void) {
    return g_project_root[0] ? g_project_root : ".";
}

static void resolve_project_root(const char *argv0) {
    char exe_path[PATH_MAX];
    if (realpath(argv0, exe_path) == NULL) {
        if (getcwd(g_project_root, sizeof(g_project_root)) == NULL)
            strcpy(g_project_root, ".");
        return;
    }
    char *slash = strrchr(exe_path, '/');
    if (slash) {
        *slash = '\0';
        slash = strrchr(exe_path, '/');
        if (slash) *slash = '\0';
    }
    strncpy(g_project_root, exe_path, sizeof(g_project_root) - 1);
    g_project_root[sizeof(g_project_root) - 1] = '\0';
}

/* Declared in repl.c */
int run_repl(void);

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = argv;
    resolve_project_root(argv[0]);

    if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("nanorepl — nanolang interactive REPL\n\n");
        printf("Usage: %s\n\n", argv[0]);
        printf("Starts an interactive read-eval-print loop.\n");
        printf("Variables and functions persist across inputs.\n");
        printf("Readline line-editing and history are supported.\n");
        return 0;
    }

    return run_repl();
}
