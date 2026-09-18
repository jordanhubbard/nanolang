/* I publish LLVM output only after complete verification and translation. */
#include "nanoisa.h"
#include "nvm2llvm.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const char *input = NULL, *output = NULL, *entry = "main";
    int entry_seen = 0, target_seen = 0;
    NvmLlvmTarget target = NVM_LLVM_NATIVE;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help")) {
            puts("I translate verified scalar NanoISA v2 to LLVM IR.\nUsage: nvm2llvm input.nvm [-o output.ll] [--entry-name main|nano_NAME] [--runtime-target native|wasm32]\nI refuse unsupported profiles; I do not embed NanoVM."); return 0;
        }
        if (!strcmp(argv[i], "-o") && i + 1 < argc && !output) output = argv[++i];
        else if (!strcmp(argv[i], "--entry-name") && i + 1 < argc && !entry_seen) {
            entry = argv[++i]; entry_seen = 1;
        }
        else if (!strcmp(argv[i], "--runtime-target") && i + 1 < argc && !target_seen) {
            const char *name = argv[++i]; target_seen = 1;
            if (!strcmp(name, "wasm32")) target = NVM_LLVM_WASM32;
            else if (strcmp(name, "native")) { fputs("I require native or wasm32 runtime target\n", stderr); return 2; }
        }
        else if (argv[i][0] == '-' || input) { fputs("I require one input and an optional -o output\n", stderr); return 2; }
        else input = argv[i];
    }
    if (!input) { fputs("I require a NanoISA input\n", stderr); return 2; }
    struct stat a, b;
    if (stat(input, &a) || (output && !stat(output, &b) && a.st_dev == b.st_dev && a.st_ino == b.st_ino)) {
        fputs("I require readable input distinct from my output\n", stderr); return 1;
    }
    FILE *in = fopen(input, "rb");
    unsigned char magic[4];
    if (!in) return 1;
    size_t read = fread(magic, 1, sizeof magic, in); fclose(in);
    if (read != sizeof magic || memcmp(magic, "NVM\002", 4)) {
        fputs("I require a v2 NanoISA module\n", stderr); return 1;
    }
    NanoisaErr load_error;
    NvmModule *m = nanoisa_load_file(input, &load_error);
    if (!m) { fprintf(stderr, "I cannot load my module: %s\n", load_error.message); return 1; }
    char *temporary = NULL;
    FILE *stream = NULL;
    if (output) {
        size_t length = strlen(output);
        if (length > SIZE_MAX - 12) { nvm_module_free(m); return 1; }
        temporary = malloc(length + 12);
        if (temporary) {
            snprintf(temporary, length + 12, "%s.XXXXXX", output);
            int fd = mkstemp(temporary);
            if (fd >= 0) { stream = fdopen(fd, "w"); if (!stream) close(fd); }
        }
    } else stream = tmpfile();
    if (!stream) {
        fputs("I cannot create my temporary output\n", stderr);
        if (temporary) unlink(temporary);
        free(temporary); nvm_module_free(m); return 1;
    }
    char error[512];
    int ok = nvm2llvm_emit_target(m, stream, error, sizeof error, entry, target);
    nvm_module_free(m);
    if (!ok) fprintf(stderr, "%s\n", error);
    if (ok && fflush(stream)) ok = 0;
    if (output) {
        if (ok && fsync(fileno(stream))) ok = 0;
        if (fclose(stream)) ok = 0;
        if (ok && rename(temporary, output)) ok = 0;
        if (!ok) unlink(temporary);
    } else {
        if (ok && fseek(stream, 0, SEEK_SET)) ok = 0;
        char buffer[4096]; size_t n;
        while (ok && (n = fread(buffer, 1, sizeof buffer, stream)))
            if (fwrite(buffer, 1, n, stdout) != n) ok = 0;
        if (ferror(stream) || fflush(stdout)) ok = 0;
        if (fclose(stream)) ok = 0;
    }
    free(temporary);
    if (!ok) fputs("I did not publish LLVM output\n", stderr);
    return ok ? 0 : 1;
}
