/* I distinguish scalar storage views from exact producers before a copy.
 * This characterizes the unchanged graph API, not a repaired C translator. */
#include "../../src/nanoisa/nvm2c_shape.h"
#include <stdio.h>

static unsigned passed, failed;
#define CHECK(condition) do { \
    if (condition) ++passed; \
    else { ++failed; fprintf(stderr, "I failed line %d: %s\n", __LINE__, #condition); } \
} while (0)

static void observe(NvmShapeKind kind, int exact_copy, int early_element, int mismatch) {
    NvmShapeGraph graph = {0};
    NvmShapeId element = nvm_shape_new(&graph, NVM_SHAPE_UNKNOWN);
    NvmShapeId read = nvm_shape_new(&graph, NVM_SHAPE_UNKNOWN);
    NvmShapeId local = nvm_shape_new(&graph, NVM_SHAPE_UNKNOWN);
    NvmShapeId initial = nvm_shape_new(&graph, kind);
    NvmShapeKind actual = mismatch ? (kind == NVM_SHAPE_BOOL ? NVM_SHAPE_FLOAT : NVM_SHAPE_BOOL) : kind;
    CHECK(nvm_shape_convert(&graph, initial, local));
    if (early_element) CHECK(nvm_shape_unify(&graph, element, nvm_shape_new(&graph, actual)));
    CHECK(nvm_shape_array_read(&graph, element, read));
    /* I retain the observed current STORE_LOCAL edge while both IDs are unknown. */
    CHECK(nvm_shape_unify(&graph, read, local));
    NvmShapeId copied_fact = nvm_shape_new(&graph, kind);
    if (exact_copy) CHECK(nvm_shape_unify(&graph, local, copied_fact));
    else CHECK(nvm_shape_convert(&graph, copied_fact, local));
    if (!early_element) CHECK(nvm_shape_unify(&graph, element, nvm_shape_new(&graph, actual)));
    int ok = nvm_shape_solve_conversions(&graph);
    CHECK(ok == (!exact_copy && !mismatch));
    if (ok) {
        CHECK(nvm_shape_kind(&graph, initial) == kind);
        CHECK(nvm_shape_kind(&graph, copied_fact) == kind);
        CHECK(nvm_shape_kind(&graph, element) == kind);
        CHECK(nvm_shape_kind(&graph, local) == NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_kind(&graph, nvm_shape_lookup(&graph, local, 0)) == kind);
        CHECK(nvm_shape_root(&graph, read) == nvm_shape_root(&graph, local));
        CHECK(nvm_shape_root(&graph, element) != nvm_shape_root(&graph, local));
        size_t nodes = graph.count, conversions = graph.conversion_count;
        CHECK(nvm_shape_solve_conversions(&graph));
        CHECK(graph.count == nodes && graph.conversion_count == conversions);
    } else {
        CHECK(graph.error != NULL);
        /* I preserve poisoning after an exact or mismatched-payload refusal. */
        CHECK(!nvm_shape_new(&graph, NVM_SHAPE_UNKNOWN));
    }
    nvm_shape_destroy(&graph);
}

int main(void) {
    const NvmShapeKind kinds[] = {NVM_SHAPE_BOOL, NVM_SHAPE_INT, NVM_SHAPE_FLOAT, NVM_SHAPE_STRING};
    for (unsigned k = 0; k < sizeof kinds / sizeof *kinds; ++k)
        for (int exact = 0; exact < 2; ++exact)
            for (int early = 0; early < 2; ++early)
                for (int mismatch = 0; mismatch < 2; ++mismatch)
                    observe(kinds[k], exact, early, mismatch);
    printf("Inferred scalar copy graphs: %u passed, %u failed\n", passed, failed);
    return failed != 0;
}
