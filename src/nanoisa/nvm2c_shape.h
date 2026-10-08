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
 * VARIANT_SCALAR is an explicit int/bool/float/string constructor payload set;
 * I never infer it by unifying unrelated exact kinds or arbitrary VALUE.
 * VARIANT_INT_ARRAY adds only exact ARRAY<INT> to that finite payload set.
 * Missing edges mean unconstrained, not absent fields or a proved width. */
typedef uint32_t NvmShapeId;
typedef enum {
    NVM_SHAPE_UNKNOWN, NVM_SHAPE_INT, NVM_SHAPE_STRING,
    NVM_SHAPE_ARRAY, NVM_SHAPE_RECORD, NVM_SHAPE_MAP, NVM_SHAPE_OPTIONAL,
    NVM_SHAPE_BOOL, NVM_SHAPE_FLOAT, NVM_SHAPE_NUMERIC, NVM_SHAPE_VARIANT_SCALAR, NVM_SHAPE_VARIANT_INT_ARRAY,
    NVM_SHAPE_FUNCTION
} NvmShapeKind;
typedef struct NvmShapeNode NvmShapeNode;
typedef struct { NvmShapeId source, target; } NvmShapeConversion;
typedef struct { NvmShapeId element, result; int resolved; } NvmShapeArrayRead;
typedef struct {
    NvmShapeNode *nodes;
    size_t count, capacity;
    const char *error;
    char error_detail[160];
    NvmShapeConversion *conversions;
    size_t conversion_count, conversion_capacity;
    NvmShapeArrayRead *array_reads;
    size_t array_read_count, array_read_capacity;
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
/* I retain function-table indices as a finite target set, never as integer
 * payloads or record edges. Empty means unresolved, not any callable target.
 * Exact joins union sets; storage conversions propagate only toward storage.
 * The caller validates each index against its module and checks signatures. */
int nvm_shape_function_add(NvmShapeGraph *graph, NvmShapeId id, uint32_t target);
size_t nvm_shape_function_count(NvmShapeGraph *graph, NvmShapeId id);
int nvm_shape_function_target(NvmShapeGraph *graph, NvmShapeId id,
                            size_t index, uint32_t *target);
/* Storage conversion does not equate source and destination nodes. I solve
 * these directed constraints after collecting the module's exact shapes. */
int nvm_shape_convert(NvmShapeGraph *graph, NvmShapeId source, NvmShapeId target);
/* I separate optional scalar reads from exact element storage, even when
 * the element kind becomes known only through later record conversions. */
int nvm_shape_array_read(NvmShapeGraph *graph, NvmShapeId element, NvmShapeId result);
int nvm_shape_solve_conversions(NvmShapeGraph *graph);

#endif
