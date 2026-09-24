#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static volatile long *escaped;
__attribute__((noinline)) static long work(long n) {
    long local[4] = {n, n + 1, n + 2, n + 3};
    escaped = local;
    return local[0];
}
int main(int argc, char **argv) {
    long count = argc > 1 ? strtol(argv[1], NULL, 10) : 100000;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    long total = 0;
    for (long i = 0; i < count; ++i) total += work(i);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("calls=%ld sum=%ld elapsed=%.6f\n", count, total,
           end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) / 1e9);
    fflush(stdout);
    if (argc > 2) return (int)*escaped;
    return total == count * (count - 1) / 2 ? 0 : 2;
}
