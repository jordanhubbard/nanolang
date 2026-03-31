/*
 * fmt_main.c — nano-fmt: nanolang code formatter CLI entry point
 *
 * Usage:
 *   nano-fmt file.nano                  print formatted source to stdout
 *   nano-fmt --write file.nano          reformat in place
 *   nano-fmt --check file.nano          exit 1 if file would change
 *   nano-fmt --indent 2 file.nano       use 2-space indent
 *   nano-fmt --indent 4 *.nano          format multiple files
 *   nano-fmt -                          read from stdin
 *
 * Exit codes:
 *   0  success (or no changes needed with --check)
 *   1  error
 *   2  file(s) would be reformatted (--check only)
 */

#include "fmt.h"
#include "nanolang.h"
#include <stdio.h>

/* Required by eval.c / runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define VERSION "0.1.0"

static void usage(const char *prog) {
    fprintf(stderr,
        "nano-fmt %s — nanolang code formatter\n\n"
        "Usage: %s [options] <file.nano> ...\n\n"
        "Options:\n"
        "  --write, -w     Reformat files in place\n"
        "  --check, -c     Exit 1 if any file would change (CI mode)\n"
        "  --indent <n>    Spaces per indent level (default: 4)\n"
        "  --verbose, -v   Print names of reformatted files\n"
        "  --help, -h      Show this help\n"
        "  --version       Show version\n"
        "  -               Read from stdin, write to stdout\n\n"
        "Examples:\n"
        "  nano-fmt src/main.nano               # preview\n"
        "  nano-fmt --write src/*.nano           # format in place\n"
        "  nano-fmt --check src/*.nano           # CI check\n"
        "  cat prog.nano | nano-fmt -            # pipe\n",
        VERSION, prog);
}

int main(int argc, char **argv) {
    FmtOptions opts = {
        .indent_size    = 4,
        .write_in_place = false,
        .check_only     = false,
        .verbose        = false,
    };

    /* Collect file arguments */
    const char *files[256];
    int nfiles = 0;
    bool read_stdin = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); return 0;
        } else if (strcmp(argv[i], "--version") == 0) {
            printf("nano-fmt %s\n", VERSION); return 0;
        } else if (strcmp(argv[i], "--write") == 0 || strcmp(argv[i], "-w") == 0) {
            opts.write_in_place = true;
        } else if (strcmp(argv[i], "--check") == 0 || strcmp(argv[i], "-c") == 0) {
            opts.check_only = true;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            opts.verbose = true;
        } else if ((strcmp(argv[i], "--indent") == 0) && i + 1 < argc) {
            opts.indent_size = atoi(argv[++i]);
            if (opts.indent_size < 1 || opts.indent_size > 16) opts.indent_size = 4;
        } else if (strcmp(argv[i], "-") == 0) {
            read_stdin = true;
        } else if (argv[i][0] != '-') {
            if (nfiles < 256) files[nfiles++] = argv[i];
        } else {
            fprintf(stderr, "nano-fmt: unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    /* Read from stdin */
    if (read_stdin || nfiles == 0) {
        char *buf = NULL; size_t blen = 0; size_t bcap = 0;
        char tmp[4096]; size_t n;
        while ((n = fread(tmp, 1, sizeof(tmp), stdin)) > 0) {
            if (blen + n + 1 > bcap) {
                bcap = bcap ? bcap * 2 : 8192;
                while (bcap < blen + n + 1) bcap *= 2;
                buf = realloc(buf, bcap);
            }
            memcpy(buf + blen, tmp, n);
            blen += n;
        }
        if (buf) buf[blen] = '\0';

        char *out = fmt_source(buf ? buf : "", "<stdin>", &opts);
        free(buf);
        if (!out) { fprintf(stderr, "nano-fmt: formatting failed\n"); return 1; }
        fputs(out, stdout);
        free(out);
        return 0;
    }

    /* Format each file */
    int rc = 0;
    for (int i = 0; i < nfiles; i++) {
        int r = fmt_file(files[i], &opts);
        if (r > rc) rc = r;
    }
    return rc;
}
