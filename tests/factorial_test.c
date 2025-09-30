#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

int64_t factorial(int64_t n);
int main();

int64_t factorial(int64_t n) {
    if ((n <= 1LL)) {
        return 1LL;
    }
    else {
        return (n * factorial((n - 1LL)));
    }
}

int main() {
    printf("%s\n", "Factorials from 0 to 10:");
    int64_t i = 0LL;
    while ((i <= 10LL)) {
        printf("%s\n", "factorial(");
        printf("%lld\n", (long long)i);
        printf("%s\n", ") = ");
        printf("%lld\n", (long long)factorial(i));
        i = (i + 1LL);
    }
    return 0LL;
}

