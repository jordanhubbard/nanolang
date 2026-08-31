/*
 * CLI tests for `nanoisa dump`.
 *
 * argv[1] is the path to the dump binary.
 */

#include "nanoisa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

static int tests_run;
static int tests_passed;
static int tests_failed;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        tests_failed++; \
        return; \
    } \
    tests_passed++; \
} while (0)

static int capture_cmd(const char *cmd, char *output, size_t output_size,
                       int *status) {
    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        return -1;
    }

    size_t used = 0;
    while (used + 1 < output_size) {
        size_t n = fread(output + used, 1, output_size - used - 1, pipe);
        if (n == 0) {
            break;
        }
        used += n;
    }
    output[used] = '\0';
    int wait_status = pclose(pipe);
    if (WIFEXITED(wait_status)) {
        *status = WEXITSTATUS(wait_status);
    } else {
        *status = 127;
    }
    return 0;
}

static char *quote_path(const char *path) {
    size_t len = strlen(path);
    char *quoted = malloc(len + 3);
    if (!quoted) {
        return NULL;
    }
    quoted[0] = '\'';
    memcpy(quoted + 1, path, len);
    quoted[len + 1] = '\'';
    quoted[len + 2] = '\0';
    return quoted;
}

static void test_dump_canonical(const char *cli) {
    const char *src =
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 7\n"
        "  RET\n"
        ".end\n";
    NanoisaErr err;
    NvmModule *mod = nanoisa_assemble_text(src, &err);
    ASSERT(mod != NULL, err.message);

    const char *nvm_path = "/tmp/nanolang_nanoisa_dump_canonical.nvm";
    ASSERT(nanoisa_save_file(mod, nvm_path, &err) == NANOISA_OK,
           "fixture saves");
    nvm_module_free(mod);

    char *cli_q = quote_path(cli);
    char *nvm_q = quote_path(nvm_path);
    ASSERT(cli_q && nvm_q, "quoted paths");
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s dump %s 2>&1", cli_q, nvm_q);

    char output[4096];
    int status = -1;
    ASSERT(capture_cmd(cmd, output, sizeof(output), &status) == 0,
           "canonical dump ran");
    ASSERT(status == 0, "canonical dump exits 0");
    ASSERT(strstr(output, "PUSH_I64 7") != NULL, "canonical listing has PUSH_I64");
    ASSERT(strstr(output, "RET") != NULL, "canonical listing has RET");
    ASSERT(strstr(output, "NVM module") == NULL,
           "canonical listing has no pretty preamble");

    free(cli_q);
    free(nvm_q);
    remove(nvm_path);
}

static void test_dump_roundtrip(const char *cli) {
    const char *src =
        ".flag needs_extern\n"
        ".entry 0\n"
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 9\n"
        "  JMP done\n"
        "  PUSH_I64 0\n"
        "done:\n"
        "  RET\n"
        ".end\n";
    NanoisaErr err;
    NvmModule *mod = nanoisa_assemble_text(src, &err);
    ASSERT(mod != NULL, err.message);

    const char *nvm_path = "/tmp/nanolang_nanoisa_dump_roundtrip.nvm";
    ASSERT(nanoisa_save_file(mod, nvm_path, &err) == NANOISA_OK,
           "round-trip fixture saves");
    nvm_module_free(mod);

    char *cli_q = quote_path(cli);
    char *nvm_q = quote_path(nvm_path);
    ASSERT(cli_q && nvm_q, "quoted paths");

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s dump %s 2>&1", cli_q, nvm_q);

    char first[4096];
    int status = -1;
    ASSERT(capture_cmd(cmd, first, sizeof(first), &status) == 0,
           "first dump ran");
    ASSERT(status == 0, "first dump exits 0");
    ASSERT(strstr(first, "[0000|") == NULL,
           "dump listing has no offset prefixes");
    ASSERT(strstr(first, ".flag needs_extern") != NULL,
           "dump listing emits needs_extern");

    NvmModule *reassembled = nanoisa_assemble_text(first, &err);
    ASSERT(reassembled != NULL, "dump listing reassembles");
    ASSERT(nanoisa_save_file(reassembled, nvm_path, &err) == NANOISA_OK,
           "reassembled module saves");
    nvm_module_free(reassembled);

    char second[4096];
    status = -1;
    ASSERT(capture_cmd(cmd, second, sizeof(second), &status) == 0,
           "second dump ran");
    ASSERT(status == 0, "second dump exits 0");
    ASSERT(strcmp(first, second) == 0, "dump listing is a text fixed point");

    free(cli_q);
    free(nvm_q);
    remove(nvm_path);
}

static void test_dump_pretty(const char *cli) {
    const char *src =
        ".function helper 0 0 0 void 0\n"
        "  RET\n"
        ".end\n";
    NanoisaErr err;
    NvmModule *mod = nanoisa_assemble_text(src, &err);
    ASSERT(mod != NULL, err.message);

    const char *nvm_path = "/tmp/nanolang_nanoisa_dump_pretty.nvm";
    ASSERT(nanoisa_save_file(mod, nvm_path, &err) == NANOISA_OK,
           "pretty fixture saves");
    nvm_module_free(mod);

    char *cli_q = quote_path(cli);
    char *nvm_q = quote_path(nvm_path);
    ASSERT(cli_q && nvm_q, "quoted paths");
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s dump --pretty %s 2>&1", cli_q, nvm_q);

    char output[4096];
    int status = -1;
    ASSERT(capture_cmd(cmd, output, sizeof(output), &status) == 0,
           "pretty dump ran");
    ASSERT(status == 0, "pretty dump exits 0");
    ASSERT(strstr(output, "NVM module") != NULL, "pretty dump has heading");
    ASSERT(strstr(output, "magic: NVM\\x01") != NULL, "pretty dump has magic");
    ASSERT(strstr(output, "RET") != NULL, "pretty dump has instruction listing");

    free(cli_q);
    free(nvm_q);
    remove(nvm_path);
}

static void test_dump_missing_file(const char *cli) {
    char *cli_q = quote_path(cli);
    ASSERT(cli_q != NULL, "quoted cli path");
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s dump /tmp/nanolang_nanoisa_dump_missing.nvm 2>&1", cli_q);

    char output[1024];
    int status = -1;
    ASSERT(capture_cmd(cmd, output, sizeof(output), &status) == 0,
           "missing-file dump ran");
    ASSERT(status != 0, "missing-file dump exits nonzero");
    ASSERT(output[0] != '\0', "missing-file dump reports an error");

    free(cli_q);
}

static void test_dump_invalid_magic(const char *cli) {
    const char *path = "/tmp/nanolang_nanoisa_dump_bad.nvm";
    FILE *file = fopen(path, "wb");
    ASSERT(file != NULL, "bad-magic fixture opens");
    unsigned char blob[32];
    memset(blob, 0, sizeof(blob));
    memcpy(blob, "BAD!", 4);
    ASSERT(fwrite(blob, 1, sizeof(blob), file) == sizeof(blob),
           "bad-magic fixture writes");
    fclose(file);

    char *cli_q = quote_path(cli);
    char *path_q = quote_path(path);
    ASSERT(cli_q && path_q, "quoted paths");
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s dump %s 2>&1", cli_q, path_q);

    char output[1024];
    int status = -1;
    ASSERT(capture_cmd(cmd, output, sizeof(output), &status) == 0,
           "bad-magic dump ran");
    ASSERT(status != 0, "bad-magic dump exits nonzero");
    ASSERT(strstr(output, "magic") != NULL, "bad-magic dump mentions magic");

    free(cli_q);
    free(path_q);
    remove(path);
}

static void test_dump_usage(const char *cli) {
    char *cli_q = quote_path(cli);
    ASSERT(cli_q != NULL, "quoted cli path");
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s 2>&1", cli_q);

    char output[1024];
    int status = -1;
    ASSERT(capture_cmd(cmd, output, sizeof(output), &status) == 0,
           "usage dump ran");
    ASSERT(status != 0, "bare invocation exits nonzero");
    ASSERT(strstr(output, "dump") != NULL, "usage mentions dump");

    free(cli_q);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <nanoisa-dump-binary>\n", argv[0]);
        return 2;
    }

    printf("=== NanoISA dump CLI tests ===\n");
    test_dump_canonical(argv[1]);
    test_dump_roundtrip(argv[1]);
    test_dump_pretty(argv[1]);
    test_dump_missing_file(argv[1]);
    test_dump_invalid_magic(argv[1]);
    test_dump_usage(argv[1]);
    printf("=== Results: %d passed, %d failed, %d total ===\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}
