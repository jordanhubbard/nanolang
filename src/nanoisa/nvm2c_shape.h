#ifndef NVM2C_SHAPE_H
#define NVM2C_SHAPE_H

#include <stddef.h>
#include <stdint.h>

/* Private AOT constraint graph. IDs are stable across growth; zero is invalid.
 * A failed graph is poisoned and must be discarded, not queried or reused.
 * Record edges are field indices; an array's edge zero is its element shape.
 * A map's edges zero and one are its key and value shapes respectively.
 * An optional's edge zero is the present value shape; absence remains tagged.
 * NUMERIC is an explicit INT|FLOAT leaf, not an inferred exact-kind conflict.
 * Missing edges mean unconstrained, not absent fields or a proved width. */
typedef uint32_t NvmShapeId;
typedef enum {
    NVM_SHAPE_UNKNOWN, NVM_SHAPE_INT, NVM_SHAPE_STRING,
    NVM_SHAPE_ARRAY, NVM_SHAPE_RECORD, NVM_SHAPE_MAP, NVM_SHAPE_OPTIONAL,
    NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT, NVM_SHAPE_NUMERIC
} NvmShapeKind;
typedef struct NvmShapeNode NvmShapeNode;
typedef struct { NvmShapeId source, target; } NvmShapeConversion;
typedef struct {
    NvmShapeNode *nodes;
    size_t count, capacity;
    const char *error;
    char error_detail[160];
    NvmShapeConversion *conversions;
    size_t conversion_count, conversion_capacity;
} NvmShapeGraph;

void nvm_shape_destroy(NvmShapeGraph *graph);
NvmShapeId nvm_shape_new(NvmShapeGraph *graph, NvmShapeKind kind);
NvmShapeId nvm_shape_root(NvmShapeGraph *graph, NvmShapeId id);
NvmShapeKind nvm_shape_kind(NvmShapeGraph *graph, NvmShapeId id);
NvmShapeId nvm_shape_child(NvmShapeGraph *graph, NvmShapeId id, uint32_t index);
/* Unlike child, lookup does not create a missing edge. Zero with no error
 * means unconstrained. Root path compression may still update parent links. */
NvmShapeId nvm_shape_lookup(NvmShapeGraph *graph, NvmShapeId id, uint32_t index);
int nvm_shape_unify(NvmShapeGraph *graph, NvmShapeId a, NvmShapeId b);
/* Storage conversion does not equate source and destination nodes. I solve
 * these directed constraints after collecting the module's exact shapes. */
int nvm_shape_convert(NvmShapeGraph *graph, NvmShapeId source, NvmShapeId target);
int nvm_shape_solve_conversions(NvmShapeGraph *graph);

#endif
