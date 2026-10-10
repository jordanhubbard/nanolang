#include <stdio.h>
#include <stdlib.h>
static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#include "socket_values_cases.h"
int main(void) {
    vs_real_lifetime(NL_SOCKET_IPV4);
    vs_real_lifetime(NL_SOCKET_IPV6);
    vs_errors_and_capacity();
    vs_overlap_controls();
    vs_endpoint_domains();
    printf("PASS %u separately linked Socket value checks\n", checks);
    return 0;
}
