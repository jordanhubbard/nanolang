#define _POSIX_C_SOURCE 200809L
#include "runtime/shadow_timing.h"

int main(int argc, char **argv) {
    int count = argc > 1 && !strcmp(argv[1], "cap") ? 4100 : 2;
    for (int i = 0; i < count; ++i) {
        errno = EDOM;
        nl_shadow_timing(i % 2 ? "test_end" : "test_start", 7, 11, i, i % 2, 0);
        if (errno != EDOM) return 37;
    }
    return 0;
}
