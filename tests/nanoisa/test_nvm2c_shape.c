#include "nvm2c_shape.h"
#include <stdio.h>
#include <string.h>

static int passed, failed;
#define CHECK(condition) do { if (condition) ++passed; else { \
    ++failed; fprintf(stderr, "I failed %s at line %d\n", #condition, __LINE__); \
} } while (0)

static NvmShapeId recursive_record(NvmShapeGraph *g) {
    NvmShapeId record = nvm_shape_new(g, NVM_SHAPE_RECORD);
    NvmShapeId array = nvm_shape_new(g, NVM_SHAPE_ARRAY);
    CHECK(nvm_shape_unify(g, nvm_shape_child(g, record, 0), array));
    CHECK(nvm_shape_unify(g, nvm_shape_child(g, array, 0), record));
    return record;
}

static void test_cycles_and_shared_children(void) {
    NvmShapeGraph g = {0};
    NvmShapeId a = recursive_record(&g), b = recursive_record(&g);
    CHECK(nvm_shape_unify(&g, a, b));
    CHECK(nvm_shape_root(&g, a) == nvm_shape_root(&g, b));
    NvmShapeId array = nvm_shape_child(&g, a, 0);
    CHECK(nvm_shape_kind(&g, array) == NVM_SHAPE_ARRAY);
    CHECK(nvm_shape_root(&g, nvm_shape_child(&g, array, 0)) == nvm_shape_root(&g, a));
    NvmShapeId first = nvm_shape_child(&g, a, 1), second = nvm_shape_child(&g, b, 2);
    CHECK(nvm_shape_unify(&g, first, second));
    CHECK(nvm_shape_unify(&g, first, nvm_shape_new(&g, NVM_SHAPE_STRING)));
    CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, b, 1)) == NVM_SHAPE_STRING);
    CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, a, 2)) == NVM_SHAPE_STRING);
    CHECK(nvm_shape_unify(&g, a, b)); /* Repeated unification is idempotent. */
    nvm_shape_destroy(&g);
    CHECK(!g.nodes && !g.count && !g.error);
}

static void test_deep_graph(int conflict) {
    NvmShapeGraph g = {0};
    NvmShapeId roots[2];
    for (int side = 0; side < 2; ++side) {
        NvmShapeId node = roots[side] = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        for (int depth = 0; depth < 10000; ++depth) {
            NvmShapeId next = nvm_shape_new(&g, NVM_SHAPE_RECORD);
            if (!nvm_shape_unify(&g, nvm_shape_child(&g, node, 0), next)) break;
            node = next;
        }
        NvmShapeId leaf = nvm_shape_new(&g, side && conflict ? NVM_SHAPE_STRING : NVM_SHAPE_INT);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, node, 65534), leaf));
    }
    CHECK(nvm_shape_unify(&g, roots[0], roots[1]) == !conflict);
    if (conflict) {
        CHECK(g.error && strstr(g.error, "conflicting"));
        CHECK(nvm_shape_new(&g, NVM_SHAPE_INT) == 0);
        CHECK(nvm_shape_root(&g, roots[0]) == 0);
    } else {
        CHECK(nvm_shape_root(&g, roots[0]) == nvm_shape_root(&g, roots[1]));
        CHECK(!g.error);
    }
    nvm_shape_destroy(&g);
}

static void test_wide_worklist(void) {
    NvmShapeGraph g = {0};
    NvmShapeId roots[2] = {nvm_shape_new(&g, NVM_SHAPE_RECORD),
                           nvm_shape_new(&g, NVM_SHAPE_RECORD)};
    for (int side = 0; side < 2; ++side) {
        for (uint32_t field = 0; field < 300; ++field) {
            NvmShapeId value = nvm_shape_new(&g, field % 2 ? NVM_SHAPE_STRING : NVM_SHAPE_INT);
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, roots[side], field), value));
        }
    }
    CHECK(nvm_shape_unify(&g, roots[0], roots[1]));
    size_t nodes = g.count;
    for (uint32_t field = 0; field < 300; ++field) {
        CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, roots[0], field)) ==
              (field % 2 ? NVM_SHAPE_STRING : NVM_SHAPE_INT));
    }
    CHECK(g.count == nodes);
    CHECK(nvm_shape_child(&g, roots[0], 0) != nvm_shape_child(&g, roots[0], 1));
    nvm_shape_destroy(&g);
}

static void test_projection_conflicts(void) {
    for (int which = 0; which < 4; ++which) {
        NvmShapeGraph g = {0};
        NvmShapeId variable = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_child(&g, variable, 3) != 0);
        if (which < 2) {
            NvmShapeId target = nvm_shape_new(&g, which ? NVM_SHAPE_ARRAY : NVM_SHAPE_INT);
            CHECK(!nvm_shape_unify(&g, variable, target));
        } else if (which == 2) CHECK(!nvm_shape_child(&g, 0, 0));
        else CHECK(!nvm_shape_root(&g, UINT32_MAX));
        CHECK(g.error != NULL);
        CHECK(!nvm_shape_unify(&g, variable, variable));
        nvm_shape_destroy(&g);
    }
    NvmShapeGraph g = {0};
    NvmShapeId a = recursive_record(&g), b = recursive_record(&g);
    CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, a, 1), nvm_shape_new(&g, NVM_SHAPE_INT)));
    CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, b, 1), nvm_shape_new(&g, NVM_SHAPE_STRING)));
    CHECK(!nvm_shape_unify(&g, a, b));
    CHECK(g.error != NULL);
    nvm_shape_destroy(&g);
}

static void test_lookup_without_constraints(void) {
    NvmShapeGraph g = {0};
    NvmShapeId record = recursive_record(&g);
    size_t count = g.count;
    CHECK(nvm_shape_lookup(&g, record, 123) == 0);
    CHECK(g.count == count && !g.error);
    NvmShapeId array = nvm_shape_lookup(&g, record, 0);
    CHECK(nvm_shape_kind(&g, array) == NVM_SHAPE_ARRAY);
    CHECK(nvm_shape_lookup(&g, array, 0) == nvm_shape_root(&g, record));
    CHECK(g.count == count);
    NvmShapeId alias = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
    CHECK(nvm_shape_unify(&g, alias, record));
    CHECK(nvm_shape_lookup(&g, alias, 0) == array);
    NvmShapeId empty = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
    count = g.count;
    CHECK(nvm_shape_lookup(&g, empty, 0) == 0);
    CHECK(g.count == count && !g.error);
    CHECK(nvm_shape_lookup(&g, empty, 1) == 0);
    CHECK(g.error != NULL);
    nvm_shape_destroy(&g);
}

static void test_map_shapes(void) {
    for (int conflict = 0; conflict < 2; ++conflict) {
        NvmShapeGraph g = {0};
        NvmShapeId maps[2] = {nvm_shape_new(&g, NVM_SHAPE_MAP), nvm_shape_new(&g, NVM_SHAPE_MAP)};
        for (int i = 0; i < 2; ++i) {
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, maps[i], 0), nvm_shape_new(&g, NVM_SHAPE_STRING)));
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, maps[i], 1),
                nvm_shape_new(&g, conflict && i ? NVM_SHAPE_STRING : NVM_SHAPE_INT)));
        }
        CHECK(nvm_shape_unify(&g, maps[0], maps[1]) == !conflict);
        if (!conflict) {
            CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, maps[0], 0)) == NVM_SHAPE_STRING);
            CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, maps[0], 1)) == NVM_SHAPE_INT);
            CHECK(!nvm_shape_child(&g, maps[0], 2));
            CHECK(g.error != NULL);
        }
        nvm_shape_destroy(&g);
    }
    NvmShapeGraph g = {0};
    CHECK(!nvm_shape_unify(&g, nvm_shape_new(&g, NVM_SHAPE_MAP), nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
    nvm_shape_destroy(&g);
}

int main(void) {
    {
        NvmShapeGraph g = {0};
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        NvmShapeId string = nvm_shape_new(&g, NVM_SHAPE_STRING);
        CHECK(!nvm_shape_unify(&g, optional, string));
        CHECK(g.error == g.error_detail);
        CHECK(strstr(g.error, "optional/string") != NULL);
        CHECK(strstr(g.error, "nodes 1/2") != NULL);
        char saved[160];
        snprintf(saved, sizeof saved, "%s", g.error);
        CHECK(nvm_shape_new(&g, NVM_SHAPE_INT) == 0);
        CHECK(strcmp(saved, g.error) == 0);
        nvm_shape_destroy(&g);
        CHECK(g.error == NULL && g.error_detail[0] == '\0');
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId boolean = nvm_shape_new(&g, NVM_SHAPE_BOOL);
        CHECK(nvm_shape_kind(&g, boolean) == NVM_SHAPE_BOOL);
        CHECK(nvm_shape_unify(&g, boolean, nvm_shape_new(&g, NVM_SHAPE_BOOL)));
        CHECK(!nvm_shape_unify(&g, boolean, nvm_shape_new(&g, NVM_SHAPE_INT)));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
    for (int conflict = 0; conflict < 2; ++conflict) {
        NvmShapeGraph g = {0};
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        NvmShapeId value = nvm_shape_child(&g, optional, 0);
        CHECK(nvm_shape_unify(&g, value, nvm_shape_new(&g, NVM_SHAPE_STRING)));
        CHECK(nvm_shape_kind(&g, optional) == NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_kind(&g, value) == NVM_SHAPE_STRING);
        if (conflict) CHECK(!nvm_shape_unify(&g, optional, value));
        else CHECK(!nvm_shape_child(&g, optional, 1));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
    test_map_shapes();
    test_lookup_without_constraints();
    test_cycles_and_shared_children();
    test_deep_graph(0);
    test_deep_graph(1);
    test_wide_worklist();
    test_projection_conflicts();
    printf("Shape constraints: %d passed, %d failed\n", passed, failed);
    return failed != 0;
}
