#include "nvm2c_shape.h"
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t index; NvmShapeId child; } ShapeEdge;
struct NvmShapeNode {
    NvmShapeId parent;
    uint32_t rank;
    NvmShapeKind kind;
    ShapeEdge *edges;
    size_t count, capacity;
};
typedef struct { NvmShapeId a, b; } ShapePair;

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
    memset(g, 0, sizeof *g);
}

NvmShapeId nvm_shape_new(NvmShapeGraph *g, NvmShapeKind kind) {
    if (g->error) return 0;
    if (kind < NVM_SHAPE_UNKNOWN || kind > NVM_SHAPE_RECORD)
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
            fail(g, "I found conflicting aggregate shape kinds");
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
