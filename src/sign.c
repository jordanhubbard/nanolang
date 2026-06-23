/*
 * sign.c — stub implementation (no OpenSSL)
 *
 * Provides the public API from sign.h as no-ops so that bin/nano and
 * bin/nanoc can be built on systems without libssl-dev.  Signing and
 * verification are not required for the interpreter or for running
 * .nano programs.
 */

#include <stdio.h>
#include "sign.h"

int wasm_sign_file(const char *wasm_path, const char *key_path)
{
    (void)wasm_path;
    (void)key_path;
    fprintf(stderr, "sign: signing not available (built without OpenSSL)\n");
    return -1;
}

int nanoc_sign_cmd(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    fprintf(stderr, "sign: signing not available (built without OpenSSL)\n");
    return 1;
}

int nanoc_verify_cmd(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    fprintf(stderr, "sign: verification not available (built without OpenSSL)\n");
    return 1;
}
