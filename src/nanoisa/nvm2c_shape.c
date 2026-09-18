#include "nvm2c_shape.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct { uint32_t index; NvmShapeId child; } ShapeEdge;
struct NvmShapeNode {
    NvmShapeId parent;
    uint32_t rank;
    NvmShapeKind kind;
    int conversion_kind;
    ShapeEdge *edges;
    size_t count, capacity;
};
typedef struct { NvmShapeId a, b; } ShapePair;

static const char *kind_name(NvmShapeKind kind) {
    static const char *names[] = {"unknown", "int", "string", "array", "record", "map", "optional", "bool", "float"};
    return names[kind];
}

static int fail(NvmShapeGraph *g, const char *message) {
    if (!g->error) g->error = message;
    return 0;
}

static void *grow(NvmShapeGraph *g, void *data, size_t *capacity,
                size_t needed, size_t width) {
    if (g->error) return NULL;
    if (needed <= *capacity) return data;
    if (needed > SIZE_MAX / width) {
        fail(g, "I cannot represent shape storage");
        return NULL;
    }
    size_t capacity_new = *capacity ? *capacity : 8;
    while (capacity_new < needed) {
        if (capacity_new > SIZE_MAX / 2 / width) {
            capacity_new = needed;
            break;
        }
        capacity_new *= 2;
    }
    void *next = realloc(data, capacity_new * width);
    if (!next) {
        fail(g, "I cannot allocate shape storage");
        return NULL;
    }
    *capacity = capacity_new;
    return next;
}

void nvm_shape_destroy(NvmShapeGraph *g) {
    for (size_t i = 0; i < g->count; ++i) free(g->nodes[i].edges);
    free(g->nodes);
    free(g->conversions);
    memset(g, 0, sizeof *g);
}

NvmShapeId nvm_shape_new(NvmShapeGraph *g, NvmShapeKind kind) {
    if (g->error) return 0;
    if (kind < NVM_SHAPE_UNKNOWN || kind > NVM_SHAPE_FLOAT)
        return fail(g, "I cannot create an invalid shape kind");
    if (g->count >= UINT32_MAX)
        return fail(g, "I cannot represent another shape ID");
    NvmShapeNode *nodes = grow(g, g->nodes, &g->capacity, g->count + 1, sizeof *g->nodes);
    if (!nodes) return 0;
    g->nodes = nodes;
    NvmShapeId id = (NvmShapeId)++g->count;
    g->nodes[id - 1] = (NvmShapeNode){.parent = id, .kind = kind};
    return id;
}

NvmShapeId nvm_shape_root(NvmShapeGraph *g, NvmShapeId id) {
    if (g->error) return 0;
    if (!id || id > g->count) return fail(g, "I cannot resolve an invalid shape ID");
    NvmShapeId root = id;
    while (g->nodes[root - 1].parent != root) root = g->nodes[root - 1].parent;
    while (id != root) {
        NvmShapeId next = g->nodes[id - 1].parent;
        g->nodes[id - 1].parent = root;
        id = next;
    }
    return root;
}

NvmShapeKind nvm_shape_kind(NvmShapeGraph *g, NvmShapeId id) {
    NvmShapeId root = nvm_shape_root(g, id);
    return root ? g->nodes[root - 1].kind : NVM_SHAPE_UNKNOWN;
}

static int allows_edge(NvmShapeKind kind, uint32_t index) {
    return kind == NVM_SHAPE_UNKNOWN || kind == NVM_SHAPE_RECORD ||
           (kind == NVM_SHAPE_MAP && index < 2) ||
           (kind == NVM_SHAPE_OPTIONAL && index == 0) ||
           (kind == NVM_SHAPE_ARRAY && index == 0);
}

NvmShapeId nvm_shape_lookup(NvmShapeGraph *g, NvmShapeId id, uint32_t index) {
    NvmShapeId root = nvm_shape_root(g, id);
    if (!root) return 0;
    NvmShapeNode *node = &g->nodes[root - 1];
    if (!allows_edge(node->kind, index)) return fail(g, "I found an invalid shape projection");
    for (size_t i = 0; i < node->count; ++i)
        if (node->edges[i].index == index) return nvm_shape_root(g, node->edges[i].child);
    return 0;
}

NvmShapeId nvm_shape_child(NvmShapeGraph *g, NvmShapeId id, uint32_t index) {
    NvmShapeId found = nvm_shape_lookup(g, id, index);
    if (found || g->error) return found;
    NvmShapeId root = nvm_shape_root(g, id);
    NvmShapeId child = nvm_shape_new(g, NVM_SHAPE_UNKNOWN);
    if (!child) return 0;
    NvmShapeNode *node = &g->nodes[root - 1]; /* Adding a node can relocate the node array. */
    ShapeEdge *edges = grow(g, node->edges, &node->capacity, node->count + 1, sizeof *node->edges);
    if (!edges) return 0;
    node->edges = edges;
    node->edges[node->count++] = (ShapeEdge){index, child};
    return child;
}

int nvm_shape_unify(NvmShapeGraph *g, NvmShapeId a, NvmShapeId b) {
    size_t count = 0, capacity = 0;
    ShapePair *pending = grow(g, NULL, &capacity, 1, sizeof *pending);
    if (!pending) return 0;
    pending[count++] = (ShapePair){a, b};
    while (count && !g->error) {
        ShapePair pair = pending[--count];
        NvmShapeId left = nvm_shape_root(g, pair.a), right = nvm_shape_root(g, pair.b);
        if (!left || !right) break;
        if (left == right) continue;
        NvmShapeNode *x = &g->nodes[left - 1], *y = &g->nodes[right - 1];
        if (x->kind != NVM_SHAPE_UNKNOWN && y->kind != NVM_SHAPE_UNKNOWN && x->kind != y->kind) {
            snprintf(g->error_detail, sizeof g->error_detail,
                     "I found conflicting aggregate shape kinds %s/%s at nodes %u/%u",
                     kind_name(x->kind), kind_name(y->kind), left, right);
            fail(g, g->error_detail);
            break;
        }
        if (x->rank < y->rank) {
            NvmShapeId swap = left; left = right; right = swap;
            x = &g->nodes[left - 1]; y = &g->nodes[right - 1];
        }
        NvmShapeKind kind = x->kind == NVM_SHAPE_UNKNOWN ? y->kind : x->kind;
        for (size_t i = 0; i < x->count; ++i)
            if (!allows_edge(kind, x->edges[i].index)) fail(g, "I found conflicting shape projections");
        for (size_t i = 0; i < y->count; ++i)
            if (!allows_edge(kind, y->edges[i].index)) fail(g, "I found conflicting shape projections");
        if (g->error) break;
        /* Join roots before visiting children: cycles then converge rather
         * than recurse. A later conflict poisons this whole graph. */
        y->parent = left;
        x->kind = kind;
        if (x->rank == y->rank) ++x->rank;
        for (size_t i = 0; i < y->count && !g->error; ++i) {
            ShapeEdge edge = y->edges[i];
            size_t at = 0;
            while (at < x->count && x->edges[at].index != edge.index) ++at;
            if (at < x->count) {
                ShapePair *next = grow(g, pending, &capacity, count + 1, sizeof *pending);
                if (!next) break;
                pending = next;
                pending[count++] = (ShapePair){x->edges[at].child, edge.child};
            } else {
                ShapeEdge *edges = grow(g, x->edges, &x->capacity, x->count + 1, sizeof *x->edges);
                if (!edges) break;
                x->edges = edges;
                x->edges[x->count++] = edge;
            }
        }
        free(y->edges);
        y->edges = NULL;
        y->count = y->capacity = 0;
    }
    free(pending);
    return !g->error;
}

int nvm_shape_convert(NvmShapeGraph *g, NvmShapeId source, NvmShapeId target) {
    if (!nvm_shape_root(g, source) || !nvm_shape_root(g, target)) return 0;
    NvmShapeConversion *next = grow(g, g->conversions, &g->conversion_capacity,
                                    g->conversion_count + 1, sizeof *next);
    if (!next) return 0;
    g->conversions = next;
    g->conversions[g->conversion_count++] = (NvmShapeConversion){source, target};
    return 1;
}

typedef struct { NvmShapeId source, target; int exact; } FlowPair;

static int flow_kind(NvmShapeGraph *g, NvmShapeId target, NvmShapeKind kind, int *changed) {
    NvmShapeNode *node = &g->nodes[target - 1];
    for (size_t i = 0; i < node->count; ++i)
        if (!allows_edge(kind, node->edges[i].index))
            return fail(g, "I found incompatible projections during storage conversion");
    if (node->kind != kind) { node->kind = kind; node->conversion_kind = 1; *changed = 1; }
    return 1;
}

static int flow_one(NvmShapeGraph *g, NvmShapeConversion conversion, int *changed) {
    size_t count = 0, capacity = 0, cursor = 0;
    FlowPair *queue = grow(g, NULL, &capacity, 1, sizeof *queue);
    if (!queue) return 0;
    queue[count++] = (FlowPair){conversion.source, conversion.target, 0};
    while (cursor < count && !g->error) {
        FlowPair pair = queue[cursor++];
        NvmShapeId source = nvm_shape_root(g, pair.source), target = nvm_shape_root(g, pair.target);
        if (!source || !target) break;
        if (source == target) continue;
        int seen = 0;
        for (size_t i = 0; i + 1 < cursor; ++i)
            if (nvm_shape_root(g, queue[i].source) == source &&
                nvm_shape_root(g, queue[i].target) == target && queue[i].exact == pair.exact) seen = 1;
        if (seen) continue;
        NvmShapeKind from = g->nodes[source - 1].kind, to = g->nodes[target - 1].kind;
        if (from == NVM_SHAPE_UNKNOWN) continue;
        if (to == NVM_SHAPE_UNKNOWN) {
            if (!flow_kind(g, target, from, changed)) break;
            to = from;
        }
        if (!pair.exact && from == NVM_SHAPE_OPTIONAL &&
            (to == NVM_SHAPE_STRING || to == NVM_SHAPE_INT || to == NVM_SHAPE_BOOL)) {
            if (!g->nodes[target - 1].conversion_kind) {
                snprintf(g->error_detail, sizeof g->error_detail,
                         "I cannot widen an exactly constrained %s destination at nodes %u/%u (conversion %u/%u)",
                         kind_name(to), source, target, conversion.source, conversion.target);
                fail(g, g->error_detail); break;
            }
            /* I widen only inferred destination storage. Its old scalar kind
             * belongs to a fresh payload, never to the wrapper itself. */
            if (!flow_kind(g, target, NVM_SHAPE_OPTIONAL, changed)) break;
            NvmShapeId payload = nvm_shape_child(g, target, 0);
            if (!payload || !flow_kind(g, payload, to, changed)) break;
            to = NVM_SHAPE_OPTIONAL;
        }
        if (!pair.exact && (from == NVM_SHAPE_STRING || from == NVM_SHAPE_INT ||
                            from == NVM_SHAPE_BOOL || from == NVM_SHAPE_FLOAT || from == NVM_SHAPE_ARRAY ||
                            from == NVM_SHAPE_MAP) && to == NVM_SHAPE_OPTIONAL) {
            NvmShapeId payload = nvm_shape_child(g, target, 0);
            FlowPair *next = grow(g, queue, &capacity, count + 1, sizeof *queue);
            if (!next || !payload) break;
            queue = next; queue[count++] = (FlowPair){source, payload, 1};
            continue;
        }
        if (from != to) {
            snprintf(g->error_detail, sizeof g->error_detail,
                     "I cannot convert aggregate storage %s to %s at nodes %u/%u",
                     kind_name(from), kind_name(to), source, target);
            fail(g, g->error_detail); break;
        }
        size_t edges = g->nodes[source - 1].count;
        for (size_t i = 0; i < edges && !g->error; ++i) {
            ShapeEdge edge = g->nodes[source - 1].edges[i];
            NvmShapeId child_source = nvm_shape_root(g, edge.child);
            NvmShapeId child_target = nvm_shape_lookup(g, target, edge.index);
            if (!child_target && !g->error) {
                /* I preserve cycles and sharing when creating missing target
                 * edges, without equating an existing destination view. */
                NvmShapeKind child_kind = nvm_shape_kind(g, child_source);
                for (size_t j = 0; j < cursor; ++j)
                    if ((child_kind == NVM_SHAPE_RECORD || child_kind == NVM_SHAPE_ARRAY ||
                         child_kind == NVM_SHAPE_MAP || child_kind == NVM_SHAPE_OPTIONAL) &&
                        nvm_shape_root(g, queue[j].source) == child_source &&
                        nvm_shape_kind(g, queue[j].target) == child_kind) {
                        child_target = nvm_shape_root(g, queue[j].target); break;
                    }
                if (!child_target) child_target = nvm_shape_new(g, NVM_SHAPE_UNKNOWN);
                if (!child_target) break;
                NvmShapeNode *node = &g->nodes[target - 1];
                ShapeEdge *next = grow(g, node->edges, &node->capacity, node->count + 1, sizeof *next);
                if (!next) break;
                node->edges = next; node->edges[node->count++] = (ShapeEdge){edge.index, child_target};
                *changed = 1;
            }
            FlowPair *next = grow(g, queue, &capacity, count + 1, sizeof *queue);
            if (!next) break;
            queue = next;
            queue[count++] = (FlowPair){child_source, child_target,
                pair.exact || from == NVM_SHAPE_OPTIONAL || from == NVM_SHAPE_MAP};
        }
    }
    free(queue);
    return !g->error;
}

int nvm_shape_solve_conversions(NvmShapeGraph *g) {
    int changed;
    do {
        changed = 0;
        for (size_t i = 0; i < g->conversion_count && !g->error; ++i)
            if (!flow_one(g, g->conversions[i], &changed)) return 0;
    } while (changed && !g->error);
    return !g->error;
}
