/*
 * NanoISA Forth REPL. sdl_forth_ide forks this as bin/forth --interactive
 * with FORTH_RAW_IO=1, so I read lines with fgets and leave termios alone.
 */

#include "forth/forth_session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int g_argc = 0;
char **g_argv = NULL;

static void strip_line(char *line) {
    size_t n;
    if (!line) return;
    n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
}

static void flush_output(ForthSession *session) {
    const char *text = forth_output(session);
    if (text != NULL && text[0] != '\0') {
        fputs(text, stdout);
    }
    forth_output_clear(session);
    fflush(stdout);
}

static int run_repl(ForthSession *session) {
    char line[FORTH_TIB_SIZE];

    fputs("Nano Forth\n", stdout);
    for (;;) {
        fputs("forth> ", stdout);
        fflush(stdout);
        if (fgets(line, (int)sizeof(line), stdin) == NULL) return 0;
        strip_line(line);
        if (!forth_interpret(session, (const uint8_t *)line, (uint32_t)strlen(line))) {
            fputs(" ?\n", stdout);
        }
        flush_output(session);
        if (forth_exit_requested(session)) return 0;
    }
}

int main(int argc, char **argv) {
    ForthSession *session;
    int interactive = 0;
    int i;
    int rc = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--interactive") == 0) {
            interactive = 1;
        }
    }

    g_argc = argc;
    g_argv = argv;

    session = forth_session_create();
    if (session == NULL) {
        fputs("forth: session create failed\n", stderr);
        return 1;
    }

    if (interactive || isatty(0) || getenv("FORTH_RAW_IO") != NULL) {
        rc = run_repl(session);
    } else {
        fputs("usage: forth --interactive\n", stderr);
        rc = 1;
    }

    forth_session_destroy(session);
    return rc;
}
