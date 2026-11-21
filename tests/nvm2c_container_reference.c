/* I expose the existing C container decision as a differential test oracle. */
#include "../src/nanoisa/nvm_format_v2.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        FILE *input = fopen(argv[i], "rb");
        if (!input) return 2;
        if (fseek(input, 0, SEEK_END)) return 2;
        long length = ftell(input);
        if (length < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(length ? (size_t)length : 1);
        if (!bytes) return 2;
        if (fread(bytes, 1, (size_t)length, input) != (size_t)length) return 2;
        if (fclose(input)) return 2;
        printf("%d\n", (int)nvm_v2_validate(bytes, (size_t)length));
        free(bytes);
    }
    return 0;
}
