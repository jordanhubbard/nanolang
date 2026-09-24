/* I check sanitizer detection after repeated fake-stack allocations. */
#include <limits.h>
#include <stdio.h>
#include <string.h>

static volatile long *escaped;

__attribute__((noinline)) static long work(long n) {
    long local[4] = {n, n + 1, n + 2, n + 3};
    escaped = local;
    return local[0];
}

int main(int argc, char **argv) {
    long total = 0;
    for (long i = 0; i < 100000; ++i) total += work(i);
    if (total != 4999950000L) return 2;
    if (argc != 2) return 3;
    if (!strcmp(argv[1], "uar")) return (int)*escaped;
    if (!strcmp(argv[1], "overflow")) {
        volatile long long maximum = LLONG_MAX;
        return (int)(maximum + argc);
    }
    if (strcmp(argv[1], "valid")) return 4;
    puts("I completed 100000 valid calls.");
    return 0;
}
