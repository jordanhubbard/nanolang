#ifndef NVM2C_SHAPE_H
#define NVM2C_SHAPE_H

#include <stddef.h>
#include <stdint.h>

/* Private AOT constraint graph. IDs are stable across growth; zero is invalid.
 * A failed graph is poisoned and must be discarded, not queried or reused.
 * Record edges are field indices; an array's edge zero is its element shape.
 * Missing edges mean unconstrained, not absent fields or a proved width. */
typedef uint32_t NvmShapeId;
typedef enum {
    NVM_SHAPE_UNKNOWN, NVM_SHAPE_INT, NVM_SHAPE_STRING,
    NVM_SHAPE_ARRAY, NVM_SHAPE_RECORD
} NvmShapeKind;
typedef struct NvmShapeNode NvmShapeNode;
typedef struct {
    NvmShapeNode *nodes;
    size_t count, capacity;
    const char *error;
} NvmShapeGraph;

void nvm_shape_destroy(NvmShapeGraph *graph);
NvmShapeId nvm_shape_new(NvmShapeGraph *graph, NvmShapeKind kind);
NvmShapeId nvm_shape_root(NvmShapeGraph *graph, NvmShapeId id);
NvmShapeKind nvm_shape_kind(NvmShapeGraph *graph, NvmShapeId id);
NvmShapeId nvm_shape_child(NvmShapeGraph *graph, NvmShapeId id, uint32_t index);
int nvm_shape_unify(NvmShapeGraph *graph, NvmShapeId a, NvmShapeId b);

#endif
