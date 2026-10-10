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

static void test_directed_conversions(void) {
    const NvmShapeKind kinds[] = {NVM_SHAPE_STRING, NVM_SHAPE_INT, NVM_SHAPE_BOOL};
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
                NvmShapeId payload = conflict ? nvm_shape_new(&g, NVM_SHAPE_FLOAT) : text;
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

static void test_deferred_array_reads(void) {
    const NvmShapeKind kinds[] = {NVM_SHAPE_INT, NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT, NVM_SHAPE_STRING, NVM_SHAPE_FUNCTION};
    for (size_t i = 0; i < sizeof kinds / sizeof *kinds; ++i) {
        for (int exact = 0; exact < 2; ++exact) {
            NvmShapeGraph g = {0};
            NvmShapeId element = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
            NvmShapeId result = nvm_shape_new(&g, exact ? kinds[i] : NVM_SHAPE_UNKNOWN);
            CHECK(nvm_shape_array_read(&g, element, result));
            CHECK(nvm_shape_solve_conversions(&g));
            CHECK(nvm_shape_kind(&g, element) == NVM_SHAPE_UNKNOWN);
            CHECK(nvm_shape_convert(&g, nvm_shape_new(&g, kinds[i]), element));
            CHECK(nvm_shape_solve_conversions(&g) == !exact);
            if (!exact) {
                CHECK(nvm_shape_kind(&g, element) == kinds[i]);
                CHECK(nvm_shape_kind(&g, result) == NVM_SHAPE_OPTIONAL);
                CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, result, 0)) == kinds[i]);
                size_t count = g.count, conversions = g.conversion_count;
                CHECK(nvm_shape_solve_conversions(&g));
                CHECK(g.count == count && g.conversion_count == conversions);
            } else CHECK(g.error != NULL);
            nvm_shape_destroy(&g);
            CHECK(!g.array_reads && !g.array_read_count);
        }
    }
    NvmShapeGraph g = {0};
    NvmShapeId element = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
    NvmShapeId result = nvm_shape_new(&g, NVM_SHAPE_RECORD);
    NvmShapeId field = nvm_shape_child(&g, result, 0);
    CHECK(nvm_shape_convert(&g, nvm_shape_new(&g, NVM_SHAPE_INT), field));
    CHECK(nvm_shape_array_read(&g, element, result));
    CHECK(nvm_shape_solve_conversions(&g));
    CHECK(nvm_shape_kind(&g, element) == NVM_SHAPE_RECORD);
    NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_RECORD);
    NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
    CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, optional, 0),
                         nvm_shape_new(&g, NVM_SHAPE_INT)));
    CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, source, 0), optional));
    CHECK(nvm_shape_convert(&g, source, element));
    CHECK(nvm_shape_solve_conversions(&g));
    CHECK(nvm_shape_kind(&g, field) == NVM_SHAPE_OPTIONAL);
    CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, field, 0)) == NVM_SHAPE_INT);
    CHECK(nvm_shape_root(&g, element) != nvm_shape_root(&g, result));
    nvm_shape_destroy(&g);
}

static void check_function_targets(NvmShapeGraph *g, NvmShapeId shape,
                                   const uint32_t *expected, size_t count) {
    CHECK(nvm_shape_kind(g, shape) == NVM_SHAPE_FUNCTION);
    CHECK(nvm_shape_function_count(g, shape) == count);
    for (size_t i = 0; i < count; ++i) {
        uint32_t target = UINT32_MAX;
        CHECK(nvm_shape_function_target(g, shape, i, &target));
        CHECK(target == expected[i]);
    }
}

static void test_nested_array_write_facts(void) {
    NvmShapeGraph g = {0};
    NvmShapeId holders[2], arrays[2], fields[2], producers[2];
    NvmShapeId callee = nvm_shape_new(&g, NVM_SHAPE_RECORD);
    for (size_t i = 0; i < 2; ++i) {
        holders[i] = nvm_shape_new(&g, NVM_SHAPE_RECORD);
        arrays[i] = nvm_shape_child(&g, holders[i], 0);
        CHECK(nvm_shape_unify(&g, arrays[i], nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
        NvmShapeId element = nvm_shape_child(&g, arrays[i], 0);
        CHECK(nvm_shape_unify(&g, element, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        fields[i] = nvm_shape_child(&g, element, 0);
        producers[i] = nvm_shape_new(&g, NVM_SHAPE_FUNCTION);
        CHECK(nvm_shape_function_add(&g, producers[i], (uint32_t)i + 2));
        CHECK(nvm_shape_convert(&g, producers[i], fields[i]));
        CHECK(nvm_shape_convert(&g, holders[i], callee));
    }
    CHECK(nvm_shape_solve_conversions(&g));
    const uint32_t first[] = {2}, second[] = {3}, read_join[] = {2, 3};
    check_function_targets(&g, fields[0], first, 1);
    check_function_targets(&g, fields[1], second, 1);
    NvmShapeId callee_array = nvm_shape_child(&g, callee, 0);
    NvmShapeId read_field = nvm_shape_child(&g, nvm_shape_child(&g, callee_array, 0), 0);
    check_function_targets(&g, read_field, read_join, 2);
    /* I add a write after initial convergence, through another alias hop.
     * Each caller gains that write but never the other caller's read facts. */
    NvmShapeId forwarded = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
    CHECK(nvm_shape_array_alias(&g, callee_array, forwarded));
    NvmShapeId written = nvm_shape_new(&g, NVM_SHAPE_RECORD);
    CHECK(nvm_shape_function_add(&g, nvm_shape_child(&g, written, 0), 9));
    CHECK(nvm_shape_array_write(&g, forwarded, written));
    CHECK(nvm_shape_solve_conversions(&g));
    const uint32_t first_written[] = {2, 9}, second_written[] = {3, 9};
    check_function_targets(&g, fields[0], first_written, 2);
    check_function_targets(&g, fields[1], second_written, 2);
    check_function_targets(&g, producers[0], first, 1);
    check_function_targets(&g, producers[1], second, 1);
    CHECK(nvm_shape_unify(&g, callee_array, forwarded));
    CHECK(nvm_shape_solve_conversions(&g));
    check_function_targets(&g, fields[0], first_written, 2);
    check_function_targets(&g, fields[1], second_written, 2);
    size_t count = g.count, conversions = g.conversion_count;
    size_t aliases = g.array_alias_count, writes = g.array_write_count;
    CHECK(nvm_shape_solve_conversions(&g));
    CHECK(g.count == count && g.conversion_count == conversions);
    CHECK(g.array_alias_count == aliases && g.array_write_count == writes);
    nvm_shape_destroy(&g);
    CHECK(!g.array_aliases && !g.array_writes);
}

static void test_array_write_alias_cycle(void) {
    NvmShapeGraph g = {0};
    NvmShapeId arrays[128];
    for (size_t i = 0; i < 128; ++i) arrays[i] = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
    for (size_t i = 0; i < 128; ++i)
        CHECK(nvm_shape_array_alias(&g, arrays[i], arrays[(i + 1) % 128]));
    NvmShapeId written = nvm_shape_new(&g, NVM_SHAPE_RECORD);
    CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, written, 0), nvm_shape_new(&g, NVM_SHAPE_STRING)));
    CHECK(nvm_shape_array_write(&g, arrays[0], written));
    CHECK(nvm_shape_solve_conversions(&g));
    for (size_t i = 0; i < 128; ++i) {
        NvmShapeId element = nvm_shape_child(&g, arrays[i], 0);
        CHECK(nvm_shape_kind(&g, element) == NVM_SHAPE_RECORD);
        CHECK(nvm_shape_kind(&g, nvm_shape_child(&g, element, 0)) == NVM_SHAPE_STRING);
    }
    CHECK(g.array_write_count == 128);
    size_t count = g.count, conversions = g.conversion_count;
    CHECK(nvm_shape_solve_conversions(&g));
    CHECK(g.count == count && g.conversion_count == conversions && g.array_write_count == 128);
    nvm_shape_destroy(&g);
}

static void test_function_targets(void) {
    for (int reverse = 0; reverse < 2; ++reverse) {
        NvmShapeGraph g = {0};
        NvmShapeId first = nvm_shape_new(&g, NVM_SHAPE_FUNCTION);
        NvmShapeId second = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_function_count(&g, first) == 0);
        CHECK(nvm_shape_function_add(&g, first, 19));
        CHECK(nvm_shape_function_add(&g, first, 0));
        CHECK(nvm_shape_function_add(&g, first, 19));
        CHECK(nvm_shape_function_add(&g, second, UINT32_MAX));
        /* I force root rank and node-array growth before joining the sets. */
        CHECK(nvm_shape_unify(&g, second, nvm_shape_new(&g, NVM_SHAPE_UNKNOWN)));
        for (int i = 0; i < 128; ++i) CHECK(nvm_shape_new(&g, NVM_SHAPE_INT));
        CHECK(nvm_shape_unify(&g, reverse ? second : first, reverse ? first : second));
        const uint32_t expected[] = {0, 19, UINT32_MAX};
        check_function_targets(&g, first, expected, 3);
        check_function_targets(&g, second, expected, 3);
        CHECK(nvm_shape_function_add(&g, first, 7));
        const uint32_t extended[] = {0, 7, 19, UINT32_MAX};
        check_function_targets(&g, second, extended, 4);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId first = nvm_shape_new(&g, NVM_SHAPE_FUNCTION);
        NvmShapeId second = nvm_shape_new(&g, NVM_SHAPE_FUNCTION);
        NvmShapeId joined = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId result = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_function_add(&g, first, 2));
        CHECK(nvm_shape_function_add(&g, second, 9));
        /* I order consumers before producers and retain a conversion cycle. */
        CHECK(nvm_shape_convert(&g, result, joined));
        CHECK(nvm_shape_convert(&g, joined, result));
        CHECK(nvm_shape_convert(&g, second, joined));
        CHECK(nvm_shape_convert(&g, first, joined));
        CHECK(nvm_shape_solve_conversions(&g));
        const uint32_t both[] = {2, 9}, only_first[] = {2}, only_second[] = {9};
        check_function_targets(&g, result, both, 2);
        check_function_targets(&g, first, only_first, 1);
        check_function_targets(&g, second, only_second, 1);
        /* I propagate new targets without changing either unrelated producer. */
        CHECK(nvm_shape_function_add(&g, second, 13));
        CHECK(nvm_shape_solve_conversions(&g));
        const uint32_t late[] = {2, 9, 13};
        check_function_targets(&g, result, late, 3);
        check_function_targets(&g, first, only_first, 1);
        CHECK(nvm_shape_solve_conversions(&g));
        check_function_targets(&g, joined, late, 3);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId source = recursive_record(&g), storage = recursive_record(&g);
        NvmShapeId producer = nvm_shape_child(&g, source, 1);
        CHECK(nvm_shape_function_add(&g, producer, 41));
        CHECK(nvm_shape_convert(&g, source, storage));
        CHECK(nvm_shape_solve_conversions(&g));
        NvmShapeId stored = nvm_shape_lookup(&g, storage, 1);
        const uint32_t target[] = {41};
        check_function_targets(&g, stored, target, 1);
        CHECK(nvm_shape_function_add(&g, stored, 42));
        CHECK(nvm_shape_solve_conversions(&g));
        check_function_targets(&g, producer, target, 1);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        NvmShapeId array = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId element = nvm_shape_child(&g, array, 0);
        CHECK(nvm_shape_convert(&g, source, element));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, element) == NVM_SHAPE_UNKNOWN);
        uint32_t targets[128];
        for (uint32_t i = 128; i > 0; --i) {
            targets[i - 1] = i * 3;
            CHECK(nvm_shape_function_add(&g, source, targets[i - 1]));
        }
        CHECK(nvm_shape_solve_conversions(&g));
        check_function_targets(&g, element, targets, 128);
        CHECK(nvm_shape_function_add(&g, source, targets[0]));
        CHECK(nvm_shape_solve_conversions(&g));
        check_function_targets(&g, element, targets, 128);
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId function = nvm_shape_new(&g, NVM_SHAPE_FUNCTION);
        NvmShapeId optional = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_function_add(&g, function, 9));
        CHECK(nvm_shape_convert(&g, function, optional));
        CHECK(nvm_shape_solve_conversions(&g));
        const uint32_t target[] = {9};
        check_function_targets(&g, nvm_shape_lookup(&g, optional, 0), target, 1);
        CHECK(nvm_shape_kind(&g, optional) == NVM_SHAPE_OPTIONAL);
        nvm_shape_destroy(&g);
    }
    for (int operation = 0; operation < 8; ++operation) {
        NvmShapeGraph g = {0};
        NvmShapeId function = nvm_shape_new(&g, NVM_SHAPE_FUNCTION);
        NvmShapeId integer = nvm_shape_new(&g, NVM_SHAPE_INT);
        CHECK(nvm_shape_function_add(&g, function, 0));
        uint32_t sentinel = 123;
        if (operation == 0) CHECK(!nvm_shape_unify(&g, function, integer));
        if (operation == 1 || operation == 2) {
            CHECK(nvm_shape_convert(&g, operation == 1 ? integer : function,
                                       operation == 1 ? function : integer));
            CHECK(!nvm_shape_solve_conversions(&g));
        }
        if (operation == 3) CHECK(!nvm_shape_function_add(&g, integer, 1));
        if (operation == 4) CHECK(!nvm_shape_child(&g, function, 0));
        if (operation == 5) CHECK(!nvm_shape_function_target(&g, function, 1, &sentinel));
        if (operation == 6) CHECK(!nvm_shape_function_target(&g, function, 0, NULL));
        if (operation == 7) CHECK(!nvm_shape_function_count(&g, integer));
        CHECK(g.error != NULL);
        CHECK(sentinel == 123);
        CHECK(!nvm_shape_function_add(&g, function, 8));
        nvm_shape_destroy(&g);
    }
    {
        NvmShapeGraph g = {0};
        NvmShapeId projected = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_child(&g, projected, 0));
        CHECK(!nvm_shape_function_add(&g, projected, 1));
        nvm_shape_destroy(&g);
    }
}

static void test_nested_record_array_views(void) {
    for (int array = 0; array < 2; ++array) {
        for (int wrong = 0; wrong < 2; ++wrong) {
            NvmShapeGraph g = {0};
            NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_RECORD);
            NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_RECORD);
            NvmShapeId from = source, to = target;
            if (array) {
                from = nvm_shape_child(&g, from, 7);
                to = nvm_shape_child(&g, to, 7);
                CHECK(nvm_shape_unify(&g, from, nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
                CHECK(nvm_shape_unify(&g, to, nvm_shape_new(&g, NVM_SHAPE_ARRAY)));
                from = nvm_shape_child(&g, from, 0);
                to = nvm_shape_child(&g, to, 0);
                CHECK(nvm_shape_unify(&g, from, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
                CHECK(nvm_shape_unify(&g, to, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
            }
            NvmShapeId boxed = nvm_shape_child(&g, from, 2);
            NvmShapeId exact = nvm_shape_child(&g, to, 2);
            CHECK(nvm_shape_unify(&g, boxed, nvm_shape_new(&g, NVM_SHAPE_OPTIONAL)));
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, boxed, 0),
                                  nvm_shape_new(&g, wrong ? NVM_SHAPE_INT : NVM_SHAPE_STRING)));
            CHECK(nvm_shape_unify(&g, exact, nvm_shape_new(&g, NVM_SHAPE_STRING)));
            CHECK(nvm_shape_convert(&g, source, target));
            CHECK(nvm_shape_solve_conversions(&g) == (array && !wrong));
            if (array && !wrong) {
                CHECK(nvm_shape_kind(&g, exact) == NVM_SHAPE_STRING);
                CHECK(nvm_shape_kind(&g, boxed) == NVM_SHAPE_OPTIONAL);
            }
            nvm_shape_destroy(&g);
        }
    }
}

static void test_shared_array_views(void) {
    for (int reverse = 0; reverse < 2; ++reverse) {
        NvmShapeGraph g = {0};
        NvmShapeId caller = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId callee = nvm_shape_new(&g, NVM_SHAPE_ARRAY);
        NvmShapeId stored = nvm_shape_child(&g, caller, 0);
        NvmShapeId written = nvm_shape_child(&g, callee, 0);
        CHECK(nvm_shape_unify(&g, stored, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        CHECK(nvm_shape_unify(&g, written, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        if (reverse) CHECK(nvm_shape_alias_view(&g, callee, caller));
        CHECK(nvm_shape_convert(&g, caller, callee));
        if (!reverse) CHECK(nvm_shape_alias_view(&g, callee, caller));
        CHECK(nvm_shape_solve_conversions(&g));
        /* I add the nested write after the first solve to exercise late facts. */
        NvmShapeId nested = nvm_shape_child(&g, written, 3);
        CHECK(nvm_shape_unify(&g, nested, nvm_shape_new(&g, NVM_SHAPE_RECORD)));
        CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, nested, 0),
                              nvm_shape_new(&g, NVM_SHAPE_STRING)));
        CHECK(nvm_shape_solve_conversions(&g));
        NvmShapeId retained = nvm_shape_lookup(&g, stored, 3);
        CHECK(nvm_shape_kind(&g, retained) == NVM_SHAPE_RECORD);
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, retained, 0)) == NVM_SHAPE_STRING);
        nvm_shape_destroy(&g);
    }
    const NvmShapeKind scalars[] = {NVM_SHAPE_STRING, NVM_SHAPE_INT, NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT};
    for (size_t i = 0; i < sizeof scalars / sizeof scalars[0]; ++i) {
        for (int wrong = 0; wrong < 2; ++wrong) {
            NvmShapeGraph g = {0};
            NvmShapeId exact = nvm_shape_new(&g, scalars[i]);
            NvmShapeId tagged = nvm_shape_new(&g, NVM_SHAPE_OPTIONAL);
            CHECK(nvm_shape_unify(&g, nvm_shape_child(&g, tagged, 0),
                                  nvm_shape_new(&g, wrong ? NVM_SHAPE_RECORD : scalars[i])));
            CHECK(nvm_shape_alias_view(&g, tagged, exact));
            CHECK(nvm_shape_solve_conversions(&g) == !wrong);
            if (!wrong) {
                CHECK(nvm_shape_kind(&g, exact) == scalars[i]);
                CHECK(nvm_shape_kind(&g, tagged) == NVM_SHAPE_OPTIONAL);
            }
            nvm_shape_destroy(&g);
        }
    }
}

int main(void) {
    {
        NvmShapeGraph g = {0};
        NvmShapeId byte = nvm_shape_new(&g, NVM_SHAPE_U8);
        NvmShapeId read = nvm_shape_new(&g, NVM_SHAPE_UNKNOWN);
        CHECK(nvm_shape_array_read(&g, byte, read));
        CHECK(nvm_shape_solve_conversions(&g));
        CHECK(nvm_shape_kind(&g, read) == NVM_SHAPE_OPTIONAL);
        CHECK(nvm_shape_kind(&g, nvm_shape_lookup(&g, read, 0)) == NVM_SHAPE_U8);
        CHECK(!nvm_shape_unify(&g, byte, nvm_shape_new(&g, NVM_SHAPE_INT)));
        CHECK(g.error != NULL);
        nvm_shape_destroy(&g);
    }
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

    test_nested_record_array_views();
    test_shared_array_views();
    test_directed_conversions();
    test_array_optional_conversion();
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
    test_array_write_alias_cycle();
    test_nested_array_write_facts();
    test_function_targets();
    test_deferred_array_reads();
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
