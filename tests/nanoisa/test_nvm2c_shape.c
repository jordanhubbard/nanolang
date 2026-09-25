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

static void test_copy_into_source_descendant(void) {
    for (int depth = 1; depth <= 3; depth += 2) {
        for (int unknown = 0; unknown < 2; ++unknown) {
            NvmShapeGraph g = {0};
            NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_RECORD);
            NvmShapeId node = source;
            for (int i = 1; i < depth; ++i) {
                NvmShapeId next = nvm_shape_new(&g, NVM_SHAPE_RECORD);
                CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, node, 0), next));
                node = next;
            }
            NvmShapeId destination = nvm_shape_new(&g, unknown ? NVM_SHAPE_UNKNOWN : NVM_SHAPE_RECORD);
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, node, 0), destination));
            CHECK(nvm_shape_convert(&g, source, destination));
            CHECK(nvm_shape_solve_conversions(&g));
            CHECK(g.count < 32);
            CHECK(nvm_shape_root(&g, source) != nvm_shape_root(&g, destination));
            node = destination;
            for (int i = 0; i < depth; ++i) {
                CHECK(nvm_shape_kind(&g, node) == NVM_SHAPE_RECORD);
                node = nvm_shape_lookup(&g, node, 0);
            }
            CHECK(nvm_shape_root(&g, node) == nvm_shape_root(&g, destination));
            size_t count = g.count;
            CHECK(nvm_shape_solve_conversions(&g));
            CHECK(g.count == count);
            nvm_shape_destroy(&g);
        }
    }
}

static void test_recursive_conversion_chain(void) {
    static const unsigned orders[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for (size_t order = 0; order < 6; ++order) {
        NvmShapeGraph g = {0};
        NvmShapeId a = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        NvmShapeId b = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        NvmShapeId c = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        NvmShapeId wrapper = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        NvmShapeId leaf = a;
        for (int depth = 0; depth < 3; ++depth) {
            leaf = nvm_shape_child(&g, leaf, 0);
            if (depth < 2) CHECK(nvm_shape_unify(&g, leaf, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        }
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, wrapper, 0), b));
        NvmShapeId sources[] = {a, c, wrapper}, targets[] = {b, a, c};
        for (size_t i = 0; i < 3; ++i)
            CHECK(nvm_shape_convert(&g, sources[orders[order][i]], targets[orders[order][i]]));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(g.count < 100);
        CHECK(nvm_shape_root(&g, a) != nvm_shape_root(&g, b));
        CHECK(nvm_shape_root(&g, b) != nvm_shape_root(&g, c));
        CHECK(nvm_shape_kind(&g, leaf) == NVM_SHAPE_RECORD);
        size_t count = g.count;
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(g.count == count);
        nvm_shape_destroy(&g);
    }
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

static void test_directed_conversions(void) {
    const NvmShapeKind kinds[] = {NVM_SHAPE_STRING, NVM_SHAPE_INT, NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT};
    for (unsigned kind = 0; kind < sizeof kinds / sizeof *kinds; ++kind) {
        for (int reverse = 0; reverse < 2; ++reverse) {
            for (int conflict = 0; conflict < 2; ++conflict) {
                NvmShapeGraph g = {0};
                NvmShapeId plain = recursive_record(&g), maybe = recursive_record(&g);
                NvmShapeId result = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
                NvmShapeId text = nvm_shape_new(&g, kinds[kind]);
                NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
                CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, plain, 1), text));
                CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, maybe, 1), optional));
                NvmShapeId payload = conflict ? nvm_shape_new(&g,
                    kinds[kind] == NVM_SHAPE_FLOAT ? NVM_SHAPE_INT : NVM_SHAPE_FLOAT) : text;
                CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, optional, 0), payload));
                CHECK(nvm_shape_convert(&g, reverse ? maybe : plain, result));
                CHECK(nvm_shape_convert(&g, reverse ? plain : maybe, result));
                CHECK(nvm_shape_solve_conversions(&g) == !conflict);
                if (!conflict) {
                    CHECK(nvm_shape_kind(&g, text) == kinds[kind]);
                    CHECK(nvm_shape_kind(&g, optional) == NVM_SHAPE_OPTIONAL);
                    CHECK(nvm_shape_root(&g, nvm_shape_child(&g, optional, 0)) == nvm_shape_root(&g, text));
                    NvmShapeId field = nvm_shape_child(&g, result, 1);
                    CHECK(nvm_shape_kind(&g, field) == NVM_SHAPE_OPTIONAL);
                    CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, field, 0)) == kinds[kind]);
                    CHECK(nvm_shape_root(&g, field) != nvm_shape_root(&g, nvm_shape_child(&g, field, 0)));
                    NvmShapeId array = nvm_shape_child(&g, result, 0);
                    CHECK(nvm_shape_root(&g, nvm_shape_child(&g, array, 0)) == nvm_shape_root(&g, result));
                    size_t count = g.count;
                    CHECK(nvm_shape_solve_conversions(&g));
                    CHECK(g.count == count);
                }
                nvm_shape_destroy(&g);
                CHECK(g.conversions == NULL && g.conversion_count == 0);
            }
        }
        {
            NvmShapeGraph g = {0};
            NvmShapeId exact = nvm_shape_new(&g, kinds[kind]);
            NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
            CHECK(nvm_shape_convert(&g, optional, exact));
            CHECK(!nvm_shape_solve_conversions(&g));
            CHECK(strstr(g.error, "exactly constrained") != NULL);
            nvm_shape_destroy(&g);
        }
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId first = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId second = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId third = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_convert(&g, second, third));
        CHECK(nvm_shape_convert(&g, first, second));
        CHECK(nvm_shape_unify(&g, first, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, first, 0), nvm_shape_new(&g, NVM_SHAPE_BOOL)));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, third, 0)) == NVM_SHAPE_BOOL);
        CHECK(nvm_shape_root(&g, first) != nvm_shape_root(&g, second));
        nvm_shape_destroy(&g);
    }
}

static void test_array_optional_conversion(void) {
    for (int conflict = 0; conflict < 2; ++conflict) {
        NvmShapeGraph g = {0};
        NvmShapeId array = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId element = nvm_shape_child(&g, array, 0);
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        NvmShapeId payload = nvm_shape_child(&g, optional, 0);
        CHECK(nvm_shape_convert(&g, array, optional));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, array) == NVM_SHAPE_ARRAY);
        CHECK(nvm_shape_kind(&g, payload) == NVM_SHAPE_ARRAY);
        CHECK(nvm_shape_kind(&g, optional) == NVM_SHAPE_OPTIONAL);
        /* I retain later-discovered element facts inside the wrapper. */
        CHECK(nvm_shape_unify(&g, element, nvm_shape_new(&g, NVM_SHAPE_STRING)));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, payload, 0),
                              nvm_shape_new(&g, conflict ? NVM_SHAPE_INT : NVM_SHAPE_STRING)));
        CHECK(nvm_shape_solve_conversions(&g) == !conflict);
        nvm_shape_destroy(&g);
    }
}

static void test_container_optional_payload_to_exact_scalar(void) {
    const NvmShapeKind kinds[] = {
        NVM_SHAPE_STRING, NVM_SHAPE_INT, NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT
    };
    for (size_t i = 0; i < sizeof kinds / sizeof *kinds; ++i) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, source, 0), optional));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, optional, 0),
                              nvm_shape_new(&g, kinds[i])));
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId exact = nvm_shape_child(&g, target, 0);
        CHECK(nvm_shape_unify(&g, exact, nvm_shape_new(&g, kinds[i])));
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, optional) == NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_kind(&g, exact) == kinds[i]);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, source, 0), optional));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, optional, 0),
                              nvm_shape_new(&g, NVM_SHAPE_INT)));
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, target, 0),
                              nvm_shape_new(&g, NVM_SHAPE_STRING)));
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(!nvm_shape_solve_conversions(&g));
        CHECK(strstr(g.error, "optional container payload int to exact string") != NULL);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, source, 0),
                              nvm_shape_new(&g, NVM_SHAPE_OPTIONAL)));
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, target, 0),
                              nvm_shape_new(&g, NVM_SHAPE_STRING)));
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(!nvm_shape_solve_conversions(&g));
        CHECK(strstr(g.error, "proved payload") != NULL);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId source_record = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, source, 0), source_record));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, source_record, 0), optional));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, optional, 0),
                              nvm_shape_new(&g, NVM_SHAPE_STRING)));
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId target_record = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, target, 0), target_record));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, target_record, 0),
                              nvm_shape_new(&g, NVM_SHAPE_STRING)));
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(!nvm_shape_solve_conversions(&g));
        CHECK(strstr(g.error, "exactly constrained") != NULL);
        nvm_shape_destroy(&g);
    }
}

static void test_numeric_union_payload(void) {
    for (int reverse = 0; reverse < 2; ++reverse) {
        NvmShapeGraph g = {0};
        NvmShapeId integer = nvm_shape_new(&g, NVM_SHAPE_INT);
        NvmShapeId floating = nvm_shape_new(&g, NVM_SHAPE_FLOAT);
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        NvmShapeId payload = nvm_shape_child(&g, optional, 0);
        CHECK(nvm_shape_unify(&g, payload, nvm_shape_new(&g, NVM_SHAPE_NUMERIC)));
        CHECK(nvm_shape_convert(&g, reverse ? floating : integer, optional));
        CHECK(nvm_shape_convert(&g, reverse ? integer : floating, optional));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, integer) == NVM_SHAPE_INT);
        CHECK(nvm_shape_kind(&g, floating) == NVM_SHAPE_FLOAT);
        CHECK(nvm_shape_kind(&g, payload) == NVM_SHAPE_NUMERIC);
        CHECK(nvm_shape_solve_conversions(&g));
        nvm_shape_destroy(&g);
    }
    for (int kind = NVM_SHAPE_INT; kind <= NVM_SHAPE_FLOAT; ++kind) {
        NvmShapeGraph g = {0};
        NvmShapeId numeric = nvm_shape_new(&g, NVM_SHAPE_NUMERIC);
        NvmShapeId other = nvm_shape_new(&g, (NvmShapeKind)kind);
        CHECK(!nvm_shape_unify(&g, numeric, other));
        nvm_shape_destroy(&g);
        numeric = nvm_shape_new(&g, NVM_SHAPE_NUMERIC);
        other = nvm_shape_new(&g, (NvmShapeKind)kind);
        CHECK(nvm_shape_convert(&g, other, numeric));
        CHECK(nvm_shape_solve_conversions(&g) == (kind == NVM_SHAPE_INT || kind == NVM_SHAPE_FLOAT));
        nvm_shape_destroy(&g);
    }
    for (int kind = NVM_SHAPE_INT; kind <= NVM_SHAPE_FLOAT; kind += NVM_SHAPE_FLOAT - NVM_SHAPE_INT) {
        NvmShapeGraph g = {0};
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, optional, 0), nvm_shape_new(&g, (NvmShapeKind)kind)));
        CHECK(nvm_shape_convert(&g, nvm_shape_new(&g, NVM_SHAPE_NUMERIC), optional));
        CHECK(!nvm_shape_solve_conversions(&g));
        nvm_shape_destroy(&g);
    }
    NvmShapeGraph g = {0};
    CHECK(!nvm_shape_child(&g, nvm_shape_new(&g, NVM_SHAPE_NUMERIC), 0));
    nvm_shape_destroy(&g);
}

static void test_explicit_variant_scalar_storage(void) {
    const NvmShapeKind members[] = {NVM_SHAPE_INT, NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT, NVM_SHAPE_STRING};
    for (size_t i = 0; i < sizeof members / sizeof members[0]; ++i) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, members[i]);
        NvmShapeId storage = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        NvmShapeId payload = nvm_shape_child(&g, storage, 0);
        CHECK(nvm_shape_unify(&g, payload, nvm_shape_new(&g, NVM_SHAPE_VARIANT_SCALAR)));
        CHECK(nvm_shape_convert(&g, source, storage));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, source) == members[i]);
        CHECK(nvm_shape_kind(&g, payload) == NVM_SHAPE_VARIANT_SCALAR);
        CHECK(nvm_shape_root(&g, source) != nvm_shape_root(&g, payload));
        nvm_shape_destroy(&g);
        source = nvm_shape_new(&g, members[i]);
        payload = nvm_shape_new(&g, NVM_SHAPE_VARIANT_SCALAR);
        CHECK(!nvm_shape_unify(&g, source, payload));
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId late = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId exact = nvm_shape_new(&g, NVM_SHAPE_STRING);
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_VARIANT_SCALAR);
        CHECK(nvm_shape_convert(&g, late, target));
        CHECK(nvm_shape_convert(&g, exact, late));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, late) == NVM_SHAPE_STRING);
        CHECK(nvm_shape_kind(&g, target) == NVM_SHAPE_VARIANT_SCALAR);
        nvm_shape_destroy(&g);
    }
    const NvmShapeKind excluded[] = {NVM_SHAPE_ARRAY, NVM_SHAPE_MAP, NVM_SHAPE_RECORD, NVM_SHAPE_NUMERIC, NVM_SHAPE_UNKNOWN};
    for (size_t i = 0; i < sizeof excluded / sizeof excluded[0]; ++i) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, excluded[i]);
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_VARIANT_SCALAR);
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(!nvm_shape_solve_conversions(&g));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
}

static void test_finite_variant_integer_array(void) {
    const NvmShapeKind members[] = {NVM_SHAPE_INT, NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT,
        NVM_SHAPE_STRING, NVM_SHAPE_VARIANT_SCALAR, NVM_SHAPE_VARIANT_INT_ARRAY};
    for (size_t i = 0; i < sizeof members / sizeof members[0]; ++i) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, members[i]);
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_VARIANT_INT_ARRAY);
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, source) == members[i]);
        nvm_shape_destroy(&g);
    }
    const NvmShapeKind elements[] = {NVM_SHAPE_INT, NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT,
        NVM_SHAPE_STRING, NVM_SHAPE_RECORD, NVM_SHAPE_ARRAY, NVM_SHAPE_UNKNOWN};
    for (size_t i = 0; i < sizeof elements / sizeof elements[0]; ++i) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId element = nvm_shape_child(&g, source, 0);
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_VARIANT_INT_ARRAY);
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(nvm_shape_unify(&g, element, nvm_shape_new(&g, elements[i])));
        CHECK(nvm_shape_solve_conversions(&g) == (elements[i] == NVM_SHAPE_INT));
        if (!g.error) {
            CHECK(nvm_shape_kind(&g, source) == NVM_SHAPE_ARRAY);
            CHECK(nvm_shape_kind(&g, element) == NVM_SHAPE_INT);
        }
        nvm_shape_destroy(&g);
    }
    const NvmShapeKind excluded[] = {NVM_SHAPE_RECORD, NVM_SHAPE_MAP, NVM_SHAPE_UNKNOWN};
    for (size_t i = 0; i < sizeof excluded / sizeof excluded[0]; ++i) {
        NvmShapeGraph g = {0};
        CHECK(nvm_shape_convert(&g, nvm_shape_new(&g, excluded[i]),
            nvm_shape_new(&g, NVM_SHAPE_VARIANT_INT_ARRAY)));
        CHECK(!nvm_shape_solve_conversions(&g));
        nvm_shape_destroy(&g);
    }
    NvmShapeGraph g = {0};
    CHECK(!nvm_shape_unify(&g, nvm_shape_new(&g, NVM_SHAPE_ARRAY),
        nvm_shape_new(&g, NVM_SHAPE_VARIANT_INT_ARRAY)));
    nvm_shape_destroy(&g);
}

static void test_nested_variant_payload_widening(void) {
    const NvmShapeKind members[] = {NVM_SHAPE_INT, NVM_SHAPE_BOOL,
        NVM_SHAPE_FLOAT, NVM_SHAPE_STRING};
    for (size_t i = 0; i < sizeof members / sizeof members[0]; ++i) {
        for (int exact = 0; exact < 2; ++exact) {
            for (int reverse = 0; reverse < 2; ++reverse) {
                NvmShapeGraph g = {0};
                NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
                NvmShapeId payload = nvm_shape_child(&g, source, 0);
                CHECK(nvm_shape_unify(&g, payload,
                    nvm_shape_new(&g, NVM_SHAPE_VARIANT_SCALAR)));
                NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
                NvmShapeId dest = nvm_shape_child(&g, target, 0);
                NvmShapeId scalar = nvm_shape_new(&g, members[i]);
                if (exact) CHECK(nvm_shape_unify(&g, dest, scalar));
                if (reverse) CHECK(nvm_shape_convert(&g, source, target));
                if (!exact) CHECK(nvm_shape_convert(&g, scalar, dest));
                if (!reverse) CHECK(nvm_shape_convert(&g, source, target));
                CHECK(nvm_shape_solve_conversions(&g) == !exact);
                if (!exact) {
                    CHECK(nvm_shape_kind(&g, dest) == NVM_SHAPE_VARIANT_SCALAR);
                    CHECK(nvm_shape_kind(&g, scalar) == members[i]);
                    CHECK(nvm_shape_root(&g, scalar) != nvm_shape_root(&g, dest));
                }
                nvm_shape_destroy(&g);
            }
        }
    }
}

static void test_aggregate_consumer_copies(void) {
    for (int reverse = 0; reverse < 2; ++reverse) {
        for (int array = 0; array < 2; ++array) {
            NvmShapeGraph g = {0};
            NvmShapeKind kind = array ? NVM_SHAPE_ARRAY : NVM_SHAPE_RECORD;
            NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
            NvmShapeId local = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
            NvmShapeId target = nvm_shape_new(&g, kind);
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, target, 0),
                                  nvm_shape_new(&g, NVM_SHAPE_STRING)));
            CHECK(nvm_shape_convert(&g, reverse ? local : source, reverse ? target : local));
            CHECK(nvm_shape_convert(&g, reverse ? source : local, reverse ? local : target));
            CHECK(nvm_shape_solve_conversions(&g));
            CHECK(nvm_shape_kind(&g, source) == kind);
            CHECK(nvm_shape_kind(&g, local) == kind);
            CHECK(nvm_shape_root(&g, source) != nvm_shape_root(&g, target));
            CHECK(nvm_shape_lookup(&g, source, 0) == 0);
            /* A consumer never rewrites a producer's explicit field type. */
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, source, 0),
                                  nvm_shape_new(&g, NVM_SHAPE_INT)));
            CHECK(!nvm_shape_solve_conversions(&g));
            nvm_shape_destroy(&g);
        }
    }
    for (int optional = 0; optional < 2; ++optional) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId target = nvm_shape_new(&g, optional ? NVM_SHAPE_OPTIONAL : NVM_SHAPE_INT);
        CHECK(nvm_shape_convert(&g, source, target));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, source) == NVM_SHAPE_UNKNOWN);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_convert(&g, source, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        CHECK(nvm_shape_convert(&g, source, nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
        CHECK(!nvm_shape_solve_conversions(&g));
        nvm_shape_destroy(&g);
    }
}

/* I model full constructor payloads, including records with different layouts. */
static NvmShapeId tagged_record(NvmShapeGraph *g, uint32_t tag, NvmShapeKind leaf) {
    NvmShapeId sum = nvm_shape_new(g, NVM_SHAPE_VARIANT);
    NvmShapeId record = nvm_shape_child(g, sum, tag);
    CHECK(nvm_shape_unify(g, record, nvm_shape_new(g, NVM_SHAPE_RECORD)));
    CHECK(nvm_shape_unify(g, nvm_shape_child(g, record, 0), nvm_shape_new(g, leaf)));
    return sum;
}

static void test_constructor_indexed_storage(void) {
    for (int reverse = 0; reverse < 2; ++reverse) {
        NvmShapeGraph g = {0};
        NvmShapeId numbered = tagged_record(&g, 0, NVM_SHAPE_INT);
        NvmShapeId named = tagged_record(&g, UINT16_MAX, NVM_SHAPE_STRING);
        NvmShapeId named_record = nvm_shape_child(&g, named, UINT16_MAX);
        NvmShapeId items = nvm_shape_child(&g, named_record, 1);
        CHECK(nvm_shape_unify(&g, items, nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, items, 0),
                              nvm_shape_new(&g, NVM_SHAPE_STRING)));
        NvmShapeId local = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId returned = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        /* I solve the consumer first, then discover both producers. */
        CHECK(nvm_shape_convert(&g, local, returned));
        CHECK(nvm_shape_convert(&g, reverse ? named : numbered, local));
        CHECK(nvm_shape_convert(&g, reverse ? numbered : named, local));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, returned) == NVM_SHAPE_VARIANT);
        NvmShapeId number_copy = nvm_shape_lookup(&g, returned, 0);
        NvmShapeId name_copy = nvm_shape_lookup(&g, returned, UINT16_MAX);
        CHECK(number_copy && name_copy && number_copy != name_copy);
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, number_copy, 0)) == NVM_SHAPE_INT);
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, name_copy, 0)) == NVM_SHAPE_STRING);
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g,
              nvm_shape_lookup(&g, name_copy, 1), 0)) == NVM_SHAPE_STRING);
        CHECK(!nvm_shape_lookup(&g, numbered, UINT16_MAX));
        CHECK(!nvm_shape_lookup(&g, named, 0));
        CHECK(!nvm_shape_lookup(&g, returned, 7));
        size_t nodes = g.count;
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(g.count == nodes);
        /* A later producer cannot change the contract of an existing tag. */
        NvmShapeId conflict = tagged_record(&g, 0, NVM_SHAPE_STRING);
        CHECK(nvm_shape_convert(&g, conflict, local));
        CHECK(!nvm_shape_solve_conversions(&g));
        CHECK(g.error && strstr(g.error, "string to int"));
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId left = tagged_record(&g, 0, NVM_SHAPE_INT);
        NvmShapeId right = tagged_record(&g, 1, NVM_SHAPE_STRING);
        CHECK(nvm_shape_unify(&g, left, right));
        CHECK(nvm_shape_lookup(&g, left, 0) != nvm_shape_lookup(&g, left, 1));
        CHECK(!nvm_shape_unify(&g, left, tagged_record(&g, 0, NVM_SHAPE_STRING)));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
}

static void test_variant_shape_boundaries(void) {
    for (int inferred = 0; inferred < 2; ++inferred) {
        NvmShapeGraph g = {0};
        NvmShapeId sum = nvm_shape_new(&g, inferred ? NVM_SHAPE_UNKNOWN : NVM_SHAPE_VARIANT);
        if (inferred) {
            CHECK(nvm_shape_child(&g, sum, (uint32_t)UINT16_MAX + 1) != 0);
            CHECK(!nvm_shape_unify(&g, sum, nvm_shape_new(&g, NVM_SHAPE_VARIANT)));
        } else CHECK(!nvm_shape_child(&g, sum, (uint32_t)UINT16_MAX + 1));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId producer = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId consumer = tagged_record(&g, 3, NVM_SHAPE_STRING);
        CHECK(nvm_shape_convert(&g, producer, consumer));
        CHECK(nvm_shape_kind(&g, producer) == NVM_SHAPE_UNKNOWN);
        CHECK(!nvm_shape_lookup(&g, producer, 3));
        CHECK(!g.error);
        CHECK(!nvm_shape_solve_conversions(&g));
        CHECK(g.error && strstr(g.error, "proved producers"));
        nvm_shape_destroy(&g);
    }
    for (int flow = 0; flow < 2; ++flow) {
        NvmShapeGraph g = {0};
        NvmShapeId sum = tagged_record(&g, 0, NVM_SHAPE_INT);
        NvmShapeId record = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        if (flow) {
            CHECK(nvm_shape_convert(&g, sum, record));
            CHECK(!nvm_shape_solve_conversions(&g));
        } else CHECK(!nvm_shape_unify(&g, sum, record));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
}

static void test_recursive_variant_copy(void) {
    NvmShapeGraph g = {0};
    NvmShapeId sum = nvm_shape_new(&g, NVM_SHAPE_VARIANT);
    NvmShapeId payload = nvm_shape_child(&g, sum, 1);
    CHECK(nvm_shape_unify(&g, payload, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
    NvmShapeId array = nvm_shape_child(&g, payload, 0);
    CHECK(nvm_shape_unify(&g, array, nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
    CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, array, 0), sum));
    CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, payload, 1), array));
    NvmShapeId copy = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
    CHECK(nvm_shape_convert(&g, sum, copy));
    CHECK(nvm_shape_solve_conversions(&g));
    CHECK(nvm_shape_kind(&g, copy) == NVM_SHAPE_VARIANT);
    NvmShapeId copied_payload = nvm_shape_lookup(&g, copy, 1);
    NvmShapeId copied_array = nvm_shape_lookup(&g, copied_payload, 0);
    CHECK(nvm_shape_root(&g, nvm_shape_lookup(&g, copied_array, 0)) ==
          nvm_shape_root(&g, copy));
    CHECK(copied_array == nvm_shape_lookup(&g, copied_payload, 1));
    CHECK(nvm_shape_root(&g, sum) != nvm_shape_root(&g, copy));
    size_t nodes = g.count;
    CHECK(nvm_shape_solve_conversions(&g));
    CHECK(g.count == nodes);
    /* I retain two explicit consumer views even when their producer is shared. */
    NvmShapeId viewed = nvm_shape_new(&g, NVM_SHAPE_VARIANT);
    NvmShapeId viewed_payload = nvm_shape_child(&g, viewed, 1);
    CHECK(nvm_shape_unify(&g, viewed_payload, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
    NvmShapeId first_view = nvm_shape_child(&g, viewed_payload, 0);
    NvmShapeId second_view = nvm_shape_child(&g, viewed_payload, 1);
    CHECK(nvm_shape_unify(&g, first_view, nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
    CHECK(nvm_shape_unify(&g, second_view, nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
    CHECK(nvm_shape_convert(&g, sum, viewed));
    CHECK(nvm_shape_solve_conversions(&g));
    CHECK(nvm_shape_root(&g, first_view) != nvm_shape_root(&g, second_view));
    CHECK(nvm_shape_root(&g, nvm_shape_lookup(&g, first_view, 0)) ==
          nvm_shape_root(&g, viewed));
    CHECK(nvm_shape_root(&g, nvm_shape_lookup(&g, second_view, 0)) ==
          nvm_shape_root(&g, viewed));
    NvmShapeId selected = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
    CHECK(nvm_shape_select_variant(&g, sum, 1, selected));
    CHECK(nvm_shape_solve_conversions(&g));
    NvmShapeId selected_array = nvm_shape_lookup(&g, selected, 0);
    NvmShapeId selected_sum = nvm_shape_lookup(&g, selected_array, 0);
    CHECK(nvm_shape_kind(&g, selected) == NVM_SHAPE_RECORD);
    CHECK(nvm_shape_root(&g, nvm_shape_lookup(&g, selected_sum, 1)) ==
          nvm_shape_root(&g, selected));
    nvm_shape_destroy(&g);
}

static void test_deferred_variant_selection(void) {
    for (int reverse = 0; reverse < 2; ++reverse) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId number = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId name = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId result = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId absent = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_select_variant(&g, source, 0, number));
        CHECK(nvm_shape_select_variant(&g, source, 1, name));
        CHECK(nvm_shape_select_variant(&g, source, 2, absent));
        CHECK(nvm_shape_convert(&g, name, result));
        NvmShapeId a = tagged_record(&g, 0, NVM_SHAPE_INT);
        NvmShapeId b = tagged_record(&g, 1, NVM_SHAPE_STRING);
        CHECK(nvm_shape_convert(&g, reverse ? a : b, source));
        CHECK(nvm_shape_convert(&g, reverse ? b : a, source));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, number, 0)) == NVM_SHAPE_INT);
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, result, 0)) == NVM_SHAPE_STRING);
        CHECK(nvm_shape_kind(&g, absent) == NVM_SHAPE_UNKNOWN);
        CHECK(!nvm_shape_lookup(&g, source, 2));
        CHECK(nvm_shape_root(&g, name) != nvm_shape_root(&g, result));
        size_t count = g.count;
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(g.count == count);
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, number, 1),
                              nvm_shape_new(&g, NVM_SHAPE_BOOL)));
        CHECK(!nvm_shape_lookup(&g, nvm_shape_lookup(&g, source, 0), 1));
        nvm_shape_destroy(&g);
        CHECK(!g.selections && !g.selection_count && !g.selection_capacity);
    }
    for (int reverse = 0; reverse < 2; ++reverse) {
        NvmShapeGraph g = {0};
        NvmShapeId known = tagged_record(&g, 0, NVM_SHAPE_INT);
        NvmShapeId unknown = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId joined = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId selected = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_convert(&g, reverse ? known : unknown, joined));
        CHECK(nvm_shape_convert(&g, reverse ? unknown : known, joined));
        CHECK(nvm_shape_select_variant(&g, joined, 0, selected));
        /* One known caller must not hide another caller's missing contract. */
        CHECK(!nvm_shape_solve_conversions(&g));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId outer = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId selected = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId inner = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId final = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        /* I solve the inner selection before its enclosing caller arrives. */
        CHECK(nvm_shape_select_variant(&g, inner, 7, final));
        CHECK(nvm_shape_select_variant(&g, outer, 3, selected));
        CHECK(nvm_shape_unify(&g, selected, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        CHECK(nvm_shape_convert(&g, nvm_shape_child(&g, selected, 0), inner));
        NvmShapeId producer = nvm_shape_new(&g, NVM_SHAPE_VARIANT);
        NvmShapeId payload = nvm_shape_child(&g, producer, 3);
        CHECK(nvm_shape_unify(&g, payload, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, payload, 0),
                              tagged_record(&g, 7, NVM_SHAPE_FLOAT)));
        CHECK(nvm_shape_convert(&g, producer, outer));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, final, 0)) == NVM_SHAPE_FLOAT);
        nvm_shape_destroy(&g);
    }
    for (int error = 0; error < 6; ++error) {
        NvmShapeGraph g = {0};
        NvmShapeId source = error == 0 ? nvm_shape_new(&g, NVM_SHAPE_UNKNOWN) :
                            error == 1 ? nvm_shape_new(&g, NVM_SHAPE_RECORD) :
                            tagged_record(&g, 0, NVM_SHAPE_INT);
        NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        if (error == 2) {
            CHECK(nvm_shape_unify(&g, target, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, target, 0),
                                  nvm_shape_new(&g, NVM_SHAPE_STRING)));
        }
        if (error == 3) CHECK(!nvm_shape_select_variant(&g, source, 65536, target));
        else if (error == 4) CHECK(!nvm_shape_select_variant(&g, 0, 0, target));
        else {
            uint32_t tag = error == 5 ? 9 : 0;
            if (error == 5) CHECK(nvm_shape_child(&g, source, tag) != 0);
            CHECK(nvm_shape_select_variant(&g, source, tag, target));
            CHECK(!nvm_shape_solve_conversions(&g));
        }
        CHECK(g.error != NULL);
        CHECK(!nvm_shape_select_variant(&g, source, 0, target));
        nvm_shape_destroy(&g);
    }
}

int main(void) {
    test_deferred_variant_selection();
    test_constructor_indexed_storage();
    test_variant_shape_boundaries();
    test_recursive_variant_copy();
    test_nested_variant_payload_widening();
    test_finite_variant_integer_array();
    test_explicit_variant_scalar_storage();
    test_numeric_union_payload();
    {
        NvmShapeGraph g = {0};
        NvmShapeId floating = nvm_shape_new(&g, NVM_SHAPE_FLOAT);
        CHECK(nvm_shape_kind(&g, floating) == NVM_SHAPE_FLOAT);
        CHECK(nvm_shape_unify(&g, floating, nvm_shape_new(&g, NVM_SHAPE_FLOAT)));
        CHECK(!nvm_shape_unify(&g, floating, nvm_shape_new(&g, NVM_SHAPE_INT)));
        nvm_shape_destroy(&g);
    }

    test_directed_conversions();
    test_array_optional_conversion();
    test_container_optional_payload_to_exact_scalar();
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
    test_aggregate_consumer_copies();
    test_map_shapes();
    test_lookup_without_constraints();
    test_cycles_and_shared_children();
    test_copy_into_source_descendant();
    test_recursive_conversion_chain();
    test_deep_graph(0);
    test_deep_graph(1);
    test_wide_worklist();
    test_projection_conflicts();
    printf("Shape constraints: %d passed, %d failed\n", passed, failed);
    return failed != 0;
}
