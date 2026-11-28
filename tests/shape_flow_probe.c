/* I measure actual directional flow, including stale IDs and repeated pairs. */
#include <assert.h>
#include <inttypes.h>

static uint64_t digest = UINT64_C(1469598103934665603);
static int strict_budget, over_budget;
static void mix(uint64_t value) {
    digest = (digest ^ value) * UINT64_C(1099511628211);
}

static void graph_digest(NvmShapeGraph *g, int result, int changed) {
    mix((unsigned)result); mix((unsigned)changed); mix(g->count);
    const char *error = g->error ? g->error : "";
    while (*error) mix((unsigned char)*error++);
    for (size_t i = 0; i < g->count; ++i) {
        NvmShapeNode *node = &g->nodes[i];
        mix(node->parent); mix(node->rank); mix(node->kind);
        mix((unsigned)node->conversion_kind); mix(node->value_tags);
        mix(node->count); mix(node->function_count);
        for (size_t j = 0; j < node->count; ++j) {
            mix(node->edges[j].index); mix(node->edges[j].child);
        }
        for (size_t j = 0; j < node->function_count; ++j) mix(node->functions[j]);
    }
    mix(g->array_alias_count); mix(g->array_write_count);
}

static void semantics(void) {
    unsigned cases = 0;
    for (int from = NVM_SHAPE_UNKNOWN; from <= NVM_SHAPE_BSTRING; ++from)
    for (int to = NVM_SHAPE_UNKNOWN; to <= NVM_SHAPE_BSTRING; ++to)
    for (int exact = 0; exact < 2; ++exact)
    for (int alias = 0; alias < 2; ++alias)
    for (int inferred = 0; inferred < 2; ++inferred) {
        NvmShapeGraph g = {0};
        NvmShapeId source = nvm_shape_new(&g, (NvmShapeKind)from);
        NvmShapeId target = nvm_shape_new(&g, (NvmShapeKind)to);
        g.nodes[target - 1].conversion_kind = inferred;
        assert(nvm_shape_tags_add(&g, source, 5));
        /* I also exercise payload flags below optional containers. */
        if (from == NVM_SHAPE_OPTIONAL || from == NVM_SHAPE_ARRAY || from == NVM_SHAPE_MAP)
            assert(nvm_shape_unify(&g, nvm_shape_child(&g, source, 0), nvm_shape_new(&g, NVM_SHAPE_INT)));
        if (to == NVM_SHAPE_OPTIONAL || to == NVM_SHAPE_ARRAY || to == NVM_SHAPE_MAP)
            assert(nvm_shape_unify(&g, nvm_shape_child(&g, target, 0), nvm_shape_new(&g, NVM_SHAPE_INT)));
        NvmShapeConversion conversion = {source, target, exact, alias};
        int changed = 0;
        int result = flow_one(&g, conversion, &changed, 1);
        graph_digest(&g, result, changed);
        if (!result) {
            assert(g.error);
            assert(!nvm_shape_new(&g, NVM_SHAPE_INT));
            assert(!nvm_shape_root(&g, source));
        }
        nvm_shape_destroy(&g); assert(!g.nodes && !g.count && !g.error);
        ++cases;
    }
    printf("{\"semantic_cases\":%u,\"digest\":\"%016" PRIx64 "\"}\n", cases, digest);
}

static void chain(size_t n, int cycle, size_t fail_at) {
    NvmShapeGraph g = {0};
    NvmShapeId *ids = malloc(2 * (n + 1) * sizeof *ids); assert(ids);
    for (size_t side = 0; side < 2; ++side) {
        NvmShapeId *nodes = ids + side * (n + 1);
        for (size_t i = 0; i <= n; ++i)
            nodes[i] = nvm_shape_new(&g, i < n || cycle ? NVM_SHAPE_RECORD : NVM_SHAPE_INT);
        for (size_t i = 0; i < n; ++i)
            assert(nvm_shape_unify(&g, nvm_shape_child(&g, nodes[i], 0), nodes[i + 1]));
        if (cycle)
            assert(nvm_shape_unify(&g, nvm_shape_child(&g, nodes[n], 0), nodes[0]));
        if (!side)
            for (size_t i = 0; i <= n; ++i) assert(nvm_shape_tags_add(&g, nodes[i], 7));
    }
    size_t before_nodes = g.count;
    flow_test_roots = 0; flow_test_allocations = 0; flow_test_fail_at = fail_at;
    flow_test_armed = 1;
    int changed = 0;
    int result = flow_one(&g, (NvmShapeConversion){ids[0], ids[n + 1], 0, 0}, &changed, 0);
    flow_test_armed = 0;
    size_t roots = flow_test_roots, allocations = flow_test_allocations;
    if (!fail_at && roots > 40 * (n + 1)) over_budget = 1;
    assert(result && !g.error && changed && g.count == before_nodes);
    for (size_t i = 0; i <= n; ++i) {
        assert(nvm_shape_tags(&g, ids[n + 1 + i]) == 7);
        assert(nvm_shape_kind(&g, ids[n + 1 + i]) == (i < n || cycle ? NVM_SHAPE_RECORD : NVM_SHAPE_INT));
    }
    changed = 0;
    assert(flow_one(&g, (NvmShapeConversion){ids[0], ids[n + 1], 0, 0}, &changed, 0));
    assert(!changed && g.count == before_nodes);
    printf("{\"nodes\":%zu,\"cycle\":%d,\"wide\":0,\"failure\":%zu,\"root_calls\":%zu,\"optional_allocations\":%zu}\n",
           n, cycle, fail_at, roots, allocations);
    free(ids); nvm_shape_destroy(&g); assert(!g.nodes && !g.count && !g.error);
}

static void shared_wide(size_t n) {
    NvmShapeGraph g = {0};
    NvmShapeId source = nvm_shape_new(&g, NVM_SHAPE_RECORD);
    NvmShapeId target = nvm_shape_new(&g, NVM_SHAPE_RECORD);
    NvmShapeId from = nvm_shape_new(&g, NVM_SHAPE_STRING);
    NvmShapeId to = nvm_shape_new(&g, NVM_SHAPE_STRING);
    for (uint32_t i = 0; i < n; ++i) {
        assert(nvm_shape_unify(&g, nvm_shape_child(&g, source, i), from));
        assert(nvm_shape_unify(&g, nvm_shape_child(&g, target, i), to));
    }
    assert(nvm_shape_tags_add(&g, from, 7));
    size_t nodes = g.count;
    flow_test_roots = 0; flow_test_allocations = 0; flow_test_fail_at = 0;
    flow_test_armed = 1;
    int changed = 0;
    assert(flow_one(&g, (NvmShapeConversion){source, target, 0, 0}, &changed, 0));
    flow_test_armed = 0;
    size_t roots = flow_test_roots;
    if (roots > 40 * (n + 1)) over_budget = 1;
    assert(changed && g.count == nodes && nvm_shape_tags(&g, to) == 7);
    for (uint32_t i = 0; i < n; ++i)
        assert(nvm_shape_root(&g, nvm_shape_lookup(&g, target, i)) == nvm_shape_root(&g, to));
    printf("{\"nodes\":%zu,\"cycle\":0,\"wide\":1,\"failure\":0,\"root_calls\":%zu,\"optional_allocations\":%zu}\n",
           n, roots, flow_test_allocations);
    nvm_shape_destroy(&g); assert(!g.nodes && !g.count && !g.error);
}

int main(int argc, char **argv) {
    assert(argc == 1 || (argc == 2 && !strcmp(argv[1], "--strict")));
    strict_budget = argc == 2;
    semantics();
    const size_t sizes[] = {32, 128, 1024, 4096};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; ++i) chain(sizes[i], 0, 0);
    chain(128, 1, 0);
    shared_wide(128);
    return strict_budget && over_budget ? 7 : 0;
}
