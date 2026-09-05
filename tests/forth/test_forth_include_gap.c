/*
 * FIND INCLUDE / INCLUDED / OPEN-FILE must fail: those are File Access
 * words, not Core, and they are not in the NanoISA session dictionary.
 * FIND DUP must succeed. C file-source helpers exist; they are not Forth
 * words. This is the Jackson INCLUDE gap, not a Core pass.
 */

#include "forth/forth_session.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;

static int expect_missing(ForthSession *session, const char *name) {
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool immediate = false;
    bool found;

    found = forth_find(session, name, (uint32_t)strlen(name), &nt, &xt, &immediate);
    if (found) {
        printf("  FAIL  FIND %s succeeded; File Access words must stay absent\n",
               name);
        return 1;
    }
    printf("  PASS  FIND %s is empty\n", name);
    return 0;
}

static int expect_present(ForthSession *session, const char *name) {
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool immediate = false;
    bool found;

    found = forth_find(session, name, (uint32_t)strlen(name), &nt, &xt, &immediate);
    if (!found) {
        printf("  FAIL  FIND %s failed\n", name);
        return 1;
    }
    printf("  PASS  FIND %s\n", name);
    return 0;
}

int main(int argc, char **argv) {
    ForthSession *session;
    int fail = 0;
    uint32_t fileid = 0;
    const char *words_path;
    FILE *fp;

    g_argc = argc;
    g_argv = argv;

    session = forth_session_create();
    if (session == NULL) {
        fprintf(stderr, "forth_session_create failed\n");
        return 1;
    }

    fail |= expect_missing(session, "INCLUDE");
    fail |= expect_missing(session, "INCLUDED");
    fail |= expect_missing(session, "OPEN-FILE");
    fail |= expect_present(session, "DUP");
    fail |= expect_present(session, "EVALUATE");
    fail |= expect_present(session, "SOURCE");

    if (forth_interpret(session, (const uint8_t *)"INCLUDED", 8)) {
        printf("  FAIL  interpret INCLUDED succeeded\n");
        fail = 1;
    } else {
        printf("  PASS  interpret INCLUDED fails\n");
    }

    /* C helpers exist; they are not INCLUDE. */
    if (!forth_file_open(session, "/dev/null", "r", &fileid)) {
        printf("  FAIL  forth_file_open(/dev/null) failed\n");
        fail = 1;
    } else if (!forth_source_push_file(session, fileid)) {
        printf("  FAIL  forth_source_push_file failed\n");
        fail = 1;
    } else {
        printf("  PASS  C forth_file_open + forth_source_push_file\n");
        forth_source_pop(session);
        forth_file_close(session, fileid);
    }

    words_path = argc > 1 ? argv[1] : "tests/forth/forth2012_core_words.txt";
    fp = fopen(words_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "missing %s\n", words_path);
        forth_session_destroy(session);
        return 1;
    }
    printf("COREFIND\n");
    for (;;) {
        char line[128];
        size_t n;
        ForthNt nt = 0;
        ForthXt xt = 0;
        bool immediate = false;
        bool found;

        if (fgets(line, (int)sizeof(line), fp) == NULL) break;
        n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[--n] = '\0';
        }
        if (n == 0) continue;
        found = forth_find(session, line, (uint32_t)n, &nt, &xt, &immediate);
        printf("%s %s\n", found ? "PRESENT" : "MISSING", line);
    }
    fclose(fp);

    forth_session_destroy(session);
    return fail;
}
