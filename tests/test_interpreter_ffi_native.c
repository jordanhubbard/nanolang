/* I expose actual native signatures for interpreter ABI tests. */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static int64_t observed;

double ffi_test_mixed(int64_t n, double d, bool b, const char *s, void *p) {
    if (!b || strcmp(s, "native") || !p) return -1;
    return n + d + *(int64_t *)p;
}

bool ffi_test_bool(bool b, int64_t n, double d) {
    return b && n == 42 && d == 1.25;
}

void ffi_test_void(double d, int64_t n, bool b) {
    observed = d == 1.25 && b ? n : -1;
}

int64_t ffi_test_observed(void) { return observed; }
void *ffi_test_pointer(void *p) { return p; }
const char *ffi_test_string(void) { return "native"; }

double ffi_test_ten_doubles(double a, double b, double c, double d, double e,
                            double f, double g, double h, double i, double j) {
    return a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g + 8*h + 9*i + 10*j;
}

int64_t ffi_test_ten_ints(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e,
                         int64_t f, int64_t g, int64_t h, int64_t i, int64_t j) {
    return a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g + 8*h + 9*i + 10*j;
}
