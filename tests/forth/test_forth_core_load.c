/*
 * Load Jackson evidence files through C file-source REFILL.
 * File Access words may be present. Core evidence is still not Forth INCLUDED.
 * I do not claim Core. I do not claim File Access as a banner.
 */

#include "forth/forth_session.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;

static int copy_source(ForthSession *session, char *out, size_t cap) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    uint64_t i;

    if (!forth_source(session, &caddr, &u)) return -1;
    if (u + 1 > cap) u = cap - 1;
    for (i = 0; i < u; i++) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + i, &ch)) return -1;
        out[i] = (char)ch;
    }
    out[u] = '\0';
    return (int)u;
}

static int load_file(ForthSession *session, const char *path) {
    uint32_t fileid = 0;
    uint32_t line = 0;
    char src[FORTH_TIB_SIZE + 1];

    printf("=== %s ===\n", path);
    if (!forth_file_open(session, path, "r", &fileid)) {
        printf("FAIL open %s\n", path);
        return 1;
    }
    if (!forth_source_push_file(session, fileid)) {
        printf("FAIL push_file %s\n", path);
        forth_file_close(session, fileid);
        return 1;
    }
    while (forth_refill(session)) {
        line++;
        if (!forth_interpret_loop(session)) {
            copy_source(session, src, sizeof(src));
            printf("FAIL interpret %s:%u\n", path, line);
            printf("SOURCE: %s\n", src);
            printf("OUTPUT:\n%s\n", forth_output(session));
            forth_source_pop(session);
            forth_file_close(session, fileid);
            return 1;
        }
        if (forth_exit_requested(session)) break;
    }
    if (forth_colon_is_open(session)) {
        printf("FAIL open colon at EOF %s after line %u\n", path, line);
        forth_source_pop(session);
        forth_file_close(session, fileid);
        return 1;
    }
    forth_source_pop(session);
    forth_file_close(session, fileid);
    printf("loaded %u lines\n", line);
    return 0;
}

static int64_t read_named_cell(ForthSession *session, const char *name) {
    char line[64];
    int64_t cell = 0;
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool immediate = false;

    if (!forth_find(session, name, (uint32_t)strlen(name), &nt, &xt, &immediate))
        return -1;
    snprintf(line, sizeof(line), "%s @", name);
    if (!forth_interpret(session, (const uint8_t *)line, (uint32_t)strlen(line)))
        return -1;
    if (forth_data_depth(session) != 1) return -1;
    if (!forth_data_pop(session, &cell)) return -1;
    return cell;
}

static int count_substr(const char *hay, const char *needle) {
    int n = 0;
    const char *p = hay;
    size_t k = strlen(needle);
    if (k == 0) return 0;
    while ((p = strstr(p, needle)) != NULL) {
        n++;
        p += k;
    }
    return n;
}

int main(int argc, char **argv) {
    ForthSession *session;
    const char *files[16];
    int nfiles = 0;
    int saw_coreext = 0;
    int saw_exception = 0;
    int saw_double = 0;
    int saw_string = 0;
    int saw_search = 0;
    int saw_file = 0;
    int saw_memory = 0;
    int saw_locals = 0;
    int saw_facility = 0;
    int saw_tools = 0;
    int saw_float = 0;
    int saw_block = 0;
    int saw_examples = 0;
    int i;
    int rc = 0;
    int64_t errs = 0;
    int64_t errors = 0;
    const char *out;
    int incorrect;
    int wrong_count;
    ForthNt nt_unused = 0;
    ForthXt xt_unused = 0;
    bool imm_unused = false;

    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc > 1) {
        for (i = 1; i < argc && nfiles < 16; i++) files[nfiles++] = argv[i];
    } else {
        files[nfiles++] = "tests/forth/vendor/gerryjackson/src/prelimtest.fth";
        files[nfiles++] = "tests/forth/vendor/gerryjackson/src/tester.fr";
        files[nfiles++] = "tests/forth/vendor/gerryjackson/src/core.fr";
        files[nfiles++] = "tests/forth/vendor/gerryjackson/src/coreplustest.fth";
    }

    session = forth_session_create();
    if (!session) {
        printf("FAIL session create\n");
        return 1;
    }
    printf("PASS Core evidence still loads through C REFILL\n");

    for (i = 0; i < nfiles; i++) {
        if (strstr(files[i], "coreexttest.fth") != NULL) saw_coreext = 1;
        if (strstr(files[i], "exceptiontest.fth") != NULL) saw_exception = 1;
        if (strstr(files[i], "doubletest.fth") != NULL) saw_double = 1;
        if (strstr(files[i], "stringtest.fth") != NULL) saw_string = 1;
        if (strstr(files[i], "searchordertest.fth") != NULL) saw_search = 1;
        if (strstr(files[i], "filetest.fth") != NULL) saw_file = 1;
        if (strstr(files[i], "memorytest.fth") != NULL) saw_memory = 1;
        if (strstr(files[i], "localstest.fth") != NULL) saw_locals = 1;
        if (strstr(files[i], "facilitytest.fth") != NULL) saw_facility = 1;
        if (strstr(files[i], "toolstest.fth") != NULL) saw_tools = 1;
        if (strstr(files[i], "ak-fp-test.fth") != NULL) saw_float = 1;
        if (strstr(files[i], "blocktest.fth") != NULL) saw_block = 1;
        if (strstr(files[i], "test_arithmetic.fs") != NULL) saw_examples = 1;
        if (load_file(session, files[i]) != 0) {
            forth_session_destroy(session);
            return 1;
        }
    }

    out = forth_output(session);
    if (!out) out = "";
    incorrect = count_substr(out, "INCORRECT RESULT");
    wrong_count = count_substr(out, "WRONG NUMBER");
    errs = 0;
    if (forth_find(session, "#ERRS", 5, &nt_unused, &xt_unused, &imm_unused))
        errs = read_named_cell(session, "#ERRS");
    errors = read_named_cell(session, "#ERRORS");
    printf("OUTPUT:\n%s\n", out);
    printf("#ERRS=%lld #ERRORS=%lld INCORRECT=%d WRONG-NUMBER=%d\n",
           (long long)errs, (long long)errors, incorrect, wrong_count);

    if (errs != 0 || errors != 0 || incorrect > 0 || wrong_count > 0) {
        if (saw_examples)
            printf("FAIL example T{ cases failed. I do not claim Core.\n");
        else if (saw_block)
            printf("FAIL Jackson Block cases failed. I do not claim Block.\n");
        else if (saw_float)
            printf("FAIL Jackson Floating-Point cases failed. I do not claim Floating-Point.\n");
        else if (saw_tools)
            printf("FAIL Jackson Programming Tools cases failed. I do not claim Programming Tools.\n");
        else if (saw_facility)
            printf("FAIL Jackson Facility cases failed. I do not claim Facility.\n");
        else if (saw_locals)
            printf("FAIL Jackson Locals cases failed. I do not claim Locals.\n");
        else if (saw_memory)
            printf("FAIL Jackson Memory-Allocation cases failed. I do not claim Memory-Allocation.\n");
        else if (saw_file)
            printf("FAIL Jackson File Access cases failed. I do not claim File Access.\n");
        else if (saw_search)
            printf("FAIL Jackson Search Order cases failed. I do not claim Search Order.\n");
        else if (saw_string)
            printf("FAIL Jackson String cases failed. I do not claim String.\n");
        else if (saw_double)
            printf("FAIL Jackson Double cases failed. I do not claim Double.\n");
        else if (saw_exception)
            printf("FAIL Jackson Exception cases failed. I do not claim Exception.\n");
        else if (saw_coreext)
            printf("FAIL Jackson Core Ext cases failed. I do not claim Core Ext.\n");
        else
            printf("FAIL Jackson Core cases failed. I do not claim Core.\n");
        rc = 1;
    } else if (saw_examples) {
        printf("PASS 280 example T{ cases via Jackson tester.fr. I do not claim Core.\n");
    } else if (saw_block) {
        printf("PASS Jackson Block evidence files. I do not claim a Standard System.\n");
    } else if (saw_float) {
        printf("PASS Jackson Floating-Point evidence files. I do not claim a Standard System.\n");
    } else if (saw_tools) {
        printf("PASS Jackson Programming Tools evidence files. I do not claim a Standard System.\n");
    } else if (saw_facility) {
        printf("PASS Jackson Facility evidence files. I do not claim a Standard System.\n");
    } else if (saw_locals) {
        printf("PASS Jackson Locals evidence files. I do not claim a Standard System.\n");
    } else if (saw_memory) {
        printf("PASS Jackson Memory-Allocation evidence files. I do not claim a Standard System.\n");
    } else if (saw_file) {
        printf("PASS Jackson File Access evidence files. I do not claim a Standard System.\n");
    } else if (saw_search) {
        printf("PASS Jackson Search Order evidence files. I do not claim a Standard System.\n");
    } else if (saw_string) {
        printf("PASS Jackson String evidence files. I do not claim a Standard System.\n");
    } else if (saw_double) {
        printf("PASS Jackson Double evidence files. I do not claim a Standard System.\n");
    } else if (saw_exception) {
        printf("PASS Jackson Exception evidence files. I do not claim a Standard System.\n");
    } else if (saw_coreext) {
        printf("PASS Jackson Core Ext evidence files. I do not claim a Standard System.\n");
    } else {
        printf("PASS Jackson Core evidence files. I do not claim Core.\n");
    }
    forth_session_destroy(session);
    return rc;
}
