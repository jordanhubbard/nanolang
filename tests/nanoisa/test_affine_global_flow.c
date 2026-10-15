#include "affine_global_flow.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

static unsigned checks;
#define CHECK(c) do { checks++; assert(c); } while (0)

static bool edge(uint8_t relation, unsigned from, unsigned to) {
    return (relation & (1u << (2 * from + to))) != 0;
}

/* I enumerate individual executions as an oracle for the compact relations. */
static void pair(NvmAffineGlobalFlow a, NvmAffineGlobalFlow b) {
    NvmAffineGlobalFlow got = nvm_affine_global_then(a, b);
    for (unsigned input = 0; input < 2; input++) {
        bool safe = (a.allowed & (1u << input)) != 0;
        bool normal[2] = {false, false}, returned[2] = {false, false};
        for (unsigned middle = 0; middle < 2; middle++) {
            returned[middle] |= edge(a.lexical_return, input, middle);
            if (!edge(a.normal, input, middle)) continue;
            if (!(b.allowed & (1u << middle))) safe = false;
            for (unsigned end = 0; end < 2; end++) {
                normal[end] |= edge(b.normal, middle, end);
                returned[end] |= edge(b.lexical_return, middle, end);
            }
        }
        CHECK(((got.allowed & (1u << input)) != 0) == safe);
        for (unsigned end = 0; end < 2; end++) {
            CHECK(edge(got.normal, input, end) == (safe && normal[end]));
            CHECK(edge(got.lexical_return, input, end) == (safe && returned[end]));
        }
    }
    NvmAffineGlobalFlow joined = nvm_affine_global_join(a, b);
    for (unsigned input = 0; input < 2; input++) {
        bool safe = (a.allowed & b.allowed & (1u << input)) != 0;
        CHECK(((joined.allowed & (1u << input)) != 0) == safe);
        for (unsigned end = 0; end < 2; end++) {
            CHECK(edge(joined.normal, input, end) ==
                (safe && (edge(a.normal, input, end) || edge(b.normal, input, end))));
            CHECK(edge(joined.lexical_return, input, end) ==
                (safe && (edge(a.lexical_return, input, end) || edge(b.lexical_return, input, end))));
        }
    }
}

int main(void) {
    NvmAffineGlobalFlow identity = nvm_affine_global_flow(3, 9, 0);
    NvmAffineGlobalFlow read = nvm_affine_global_flow(2, 8, 0);
    NvmAffineGlobalFlow initialize = nvm_affine_global_flow(1, 2, 0);
    NvmAffineGlobalFlow write = nvm_affine_global_flow(3, 10, 0);
    NvmAffineGlobalFlow leave = nvm_affine_global_flow(3, 0, 9);
    CHECK(nvm_affine_global_then(initialize, read).allowed == 1);
    CHECK(nvm_affine_global_then(initialize, initialize).allowed == 0);
    CHECK(nvm_affine_global_then(write, read).allowed == 3);
    CHECK(nvm_affine_global_then(identity, read).allowed == 2);
    NvmAffineGlobalFlow early = nvm_affine_global_then(initialize, leave);
    CHECK(early.normal == 0 && early.lexical_return == 2);
    CHECK(nvm_affine_global_then(early, initialize).allowed == 1);
    NvmAffineGlobalFlow maybe = nvm_affine_global_join(identity, write);
    CHECK(nvm_affine_global_then(maybe, read).allowed == 2);
    NvmAffineGlobalFlow arm = nvm_affine_global_join(initialize, early);
    NvmAffineGlobalFlow continued = nvm_affine_global_then(arm, read);
    CHECK(continued.allowed == 1 && continued.normal == 2 && continued.lexical_return == 2);
    /* I check every encoded input, including empty/divergent continuations. */
    for (unsigned a = 0; a < 1024; a++)
        for (unsigned b = 0; b < 1024; b++)
            pair(nvm_affine_global_flow(a & 3u, (a >> 2) & 15u, a >> 6),
                 nvm_affine_global_flow(b & 3u, (b >> 2) & 15u, b >> 6));
    printf("PASS %u global transfer checks; handler integration remains separate\n", checks);
    return 0;
}
