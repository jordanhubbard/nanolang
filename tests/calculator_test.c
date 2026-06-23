#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

int64_t add(int64_t a, int64_t b);
int64_t subtract(int64_t a, int64_t b);
int64_t multiply(int64_t a, int64_t b);
int64_t divide(int64_t a, int64_t b);
int64_t modulo(int64_t a, int64_t b);
int64_t abs(int64_t n);
int64_t max(int64_t a, int64_t b);
int64_t min(int64_t a, int64_t b);
int main();

int64_t add(int64_t a, int64_t b) {
    return (a + b);
}

int64_t subtract(int64_t a, int64_t b) {
    return (a - b);
}

int64_t multiply(int64_t a, int64_t b) {
    return (a * b);
}

int64_t divide(int64_t a, int64_t b) {
    return (a / b);
}

int64_t modulo(int64_t a, int64_t b) {
    return (a % b);
}

int64_t abs(int64_t n) {
    if ((n < 0LL)) {
        return (0LL - n);
    }
    else {
        return n;
    }
}

int64_t max(int64_t a, int64_t b) {
    if ((a > b)) {
        return a;
    }
    else {
        return b;
    }
}

int64_t min(int64_t a, int64_t b) {
    if ((a < b)) {
        return a;
    }
    else {
        return b;
    }
}

int main() {
    printf("%s\n", "Calculator Demo");
    printf("%s\n", "");
    int64_t a = 15LL;
    int64_t b = 4LL;
    printf("%s\n", "a = 15, b = 4");
    printf("%s\n", "");
    printf("%s\n", "add(a, b) = ");
    printf("%lld\n", (long long)add(a, b));
    printf("%s\n", "subtract(a, b) = ");
    printf("%lld\n", (long long)subtract(a, b));
    printf("%s\n", "multiply(a, b) = ");
    printf("%lld\n", (long long)multiply(a, b));
    printf("%s\n", "divide(a, b) = ");
    printf("%lld\n", (long long)divide(a, b));
    printf("%s\n", "modulo(a, b) = ");
    printf("%lld\n", (long long)modulo(a, b));
    printf("%s\n", "abs(-42) = ");
    printf("%lld\n", (long long)abs(-42LL));
    printf("%s\n", "max(a, b) = ");
    printf("%lld\n", (long long)max(a, b));
    printf("%s\n", "min(a, b) = ");
    printf("%lld\n", (long long)min(a, b));
    return 0LL;
}

