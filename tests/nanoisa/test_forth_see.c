#include "../../modules/forth_see/forth_see.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <forth-interpreter.nvm>\n", argv[0]);
        return 2;
    }

    const char *output = nl_forth_see("dup", argv[1]);
    if (!strstr(output, "ISA implementation of Forth word: dup")) {
        fprintf(stderr, "unexpected SEE output:\n%s", output);
        return 1;
    }
    if (!strstr(output, "NanoISA block:")) {
        fprintf(stderr, "SEE output has no bytecode block:\n%s", output);
        return 1;
    }
    return 0;
}
