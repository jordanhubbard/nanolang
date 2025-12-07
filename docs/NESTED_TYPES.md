# Nested Types in Nanolang

## Overview

Nanolang now supports **arbitrary nesting of array types**, enabling powerful data structures like matrices, 3D grids, and complex nested collections.

## Nested Arrays

### 2D Arrays (Matrices)

```nano
fn create_matrix() -> int {
    # Create a 2x3 matrix
    let mut matrix: array<array<int>> = []
    
    # Create rows
    let mut row1: array<int> = []
    set row1 (array_push row1 1)
    set row1 (array_push row1 2)
    set row1 (array_push row1 3)
    
    let mut row2: array<int> = []
    set row2 (array_push row2 4)
    set row2 (array_push row2 5)
    set row2 (array_push row2 6)
    
    # Build matrix
    set matrix (array_push matrix row1)
    set matrix (array_push matrix row2)
    
    # Access matrix[1][2] (value = 6)
    let row: array<int> = (at matrix 1)
    let value: int = (at row 2)
    
    return value  # Returns 6
}
```

### 3D Arrays (Cubes/Voxels)

```nano
fn create_cube() -> int {
    # Create a 2x2x2 cube
    let mut cube: array<array<array<int>>> = []
    
    # Layer 0
    let mut layer0: array<array<int>> = []
    let mut row00: array<int> = []
    set row00 (array_push row00 1)
    set row00 (array_push row00 2)
    let mut row01: array<int> = []
    set row01 (array_push row01 3)
    set row01 (array_push row01 4)
    set layer0 (array_push layer0 row00)
    set layer0 (array_push layer0 row01)
    
    # Layer 1
    let mut layer1: array<array<int>> = []
    let mut row10: array<int> = []
    set row10 (array_push row10 5)
    set row10 (array_push row10 6)
    let mut row11: array<int> = []
    set row11 (array_push row11 7)
    set row11 (array_push row11 8)
    set layer1 (array_push layer1 row10)
    set layer1 (array_push layer1 row11)
    
    # Build cube
    set cube (array_push cube layer0)
    set cube (array_push cube layer1)
    
    # Access cube[1][1][0] (value = 7)
    let layer: array<array<int>> = (at cube 1)
    let row: array<int> = (at layer 1)
    let value: int = (at row 0)
    
    return value  # Returns 7
}
```

## Implementation Details

### Type Declaration

Nested array types use angle bracket syntax with recursive nesting:

- `array<int>` - Simple array of integers
- `array<array<int>>` - 2D array (array of arrays of integers)
- `array<array<array<int>>>` - 3D array (array of arrays of arrays of integers)
- Arbitrary depth supported

### Empty Array Initialization

Empty nested arrays require proper type annotations:

```nano
# CORRECT - Type annotation required
let mut matrix: array<array<int>> = []

# WRONG - Type cannot be inferred
let mut matrix = []
```

### Access Patterns

Nested arrays require sequential access through each level:

```nano
# 2D array access
let row: array<int> = (at matrix 0)
let value: int = (at row 1)

# 3D array access
let layer: array<array<int>> = (at cube 0)
let row: array<int> = (at layer 0)
let value: int = (at row 0)
```

### Memory Management

Nested arrays are fully garbage collected:

- Inner arrays are tracked by GC
- Outer arrays reference inner arrays through GC-managed pointers
- No manual memory management required
- Tested with 100x10 nested arrays (1000 inner arrays)

### Compilation

Nested arrays work in **both interpreter and compiled modes**:

- Interpreter: Uses `builtin_array_push/pop/at`
- Transpiler: Generates `dyn_array_push_array/pop_array/get_array` calls
- C runtime: Uses `ELEM_ARRAY` element type for nested arrays

## Use Cases

### Graphics & Math

```nano
# 4x4 transformation matrix
let mut transform: array<array<float>> = []

# 3D vector array for mesh vertices
let mut vertices: array<array<float>> = []
```

### Game Development

```nano
# 2D tile map
let mut tilemap: array<array<int>> = []

# 3D voxel world
let mut world: array<array<array<int>>> = []
```

### Data Structures

```nano
# Adjacency list for graphs
let mut graph: array<array<int>> = []

# Tree represented as nested arrays
let mut tree: array<array<array<int>>> = []
```

## Performance Considerations

1. **Access Speed**: Each level of nesting requires one array lookup
   - 2D: Two lookups `(at (at matrix y) x)`
   - 3D: Three lookups `(at (at (at cube z) y) x)`

2. **Memory**: Each nested array is a separate DynArray object
   - Overhead: ~24 bytes per DynArray header
   - Data: Proportional to element count and type

3. **GC Impact**: Nested arrays create more GC objects
   - GC marks all nested arrays recursively
   - Stress tested with 1000+ nested arrays successfully

## Limitations

### Lists vs Arrays

Currently, **List types do not support nesting**:
- `List<int>` - Supported
- `List<Point>` - Supported (generic list)
- `List<List<int>>` - **NOT supported**

**Workaround**: Use nested arrays instead:
```nano
# Instead of List<List<int>>, use:
let mut data: array<array<int>> = []
```

Arrays now support dynamic operations (`array_push`, `array_pop`) just like Lists, making them suitable for most use cases.

### Function Types

Nested function types are **not yet implemented**:
- `fn(int) -> int` - Supported
- `fn() -> fn() -> int` - **NOT supported**

This is planned for a future release.

## Examples

See the test suite for complete working examples:

- `tests/test_nested_simple.nano` - Basic 2D arrays
- `tests/test_nested_complex.nano` - 2x3 matrix with full access
- `tests/test_nested_3d.nano` - 3D cube (2x2x2)
- `tests/test_nested_gc.nano` - GC stress test with 100x10 arrays
- `tests/test_types_comprehensive.nano` - Comprehensive type tests

## Future Enhancements

Planned features for nested types:

1. **Nested Lists**: Support `List<List<T>>` syntax
2. **Nested Functions**: Support `fn() -> fn() -> int` syntax
3. **Syntactic Sugar**: Simplified multi-dimensional access like `matrix[i][j]`
4. **Type Inference**: Better inference for nested empty arrays
5. **Performance**: Optimize nested array access patterns

## Summary

Nested arrays are a **fundamental language feature** in nanolang, enabling:

✅ Matrices and multi-dimensional arrays  
✅ Complex data structures with arbitrary nesting depth  
✅ Full garbage collection support  
✅ Works in both interpreter and compiled modes  
✅ Tested with 2D and 3D arrays  

This brings nanolang's type system to feature parity with modern languages!
