# Tuple Return Types - Full Implementation Guide

## Status: 95% Complete

### What's Been Accomplished ✅

1. **Parser Infrastructure**
   - Added `TypeInfo **type_info_out` parameter to `parse_type_with_element()`
   - Parser creates `TypeInfo` objects for tuple types with element information
   - `return_type_info` field added to AST function nodes

2. **Transpiler Infrastructure**  
   - Added `TupleTypeRegistry` (similar to `FunctionTypeRegistry`)
   - Functions for: `create_tuple_type_registry()`, `free_tuple_type_registry()`
   - Functions for: `tuple_types_equal()`, `register_tuple_type()`, `generate_tuple_typedef()`
   - Generates typedef names like: `Tuple_int_int_0`, `Tuple_float_bool_1`

3. **Type Generation**
   - Transpiler generates inline struct syntax for tuple returns
   - Applied to both forward declarations and definitions

### Remaining Work (5%)  

Need to integrate TupleTypeRegistry into main transpile function:

```c
// In transpile_program():
// 1. Create registry after FunctionTypeRegistry
TupleTypeRegistry *tuple_registry = create_tuple_type_registry();

// 2. Register all tuple return types (after line ~1950)
for (int i = 0; i < program->as.program.count; i++) {
    ASTNode *item = program->as.program.items[i];
    if (item->type == AST_FUNCTION && 
        item->as.function.return_type == TYPE_TUPLE && 
        item->as.function.return_type_info) {
        register_tuple_type(tuple_registry, item->as.function.return_type_info);
    }
}

// 3. Generate typedef declarations (after line ~2000, before forward declarations)
if (tuple_registry->count > 0) {
    sb_append(sb, "/* Tuple Type Typedefs */\n");
    for (int i = 0; i < tuple_registry->count; i++) {
        generate_tuple_typedef(sb, tuple_registry->tuples[i],
                             tuple_registry->typedef_names[i]);
    }
    sb_append(sb, "\n");
}

// 4. Use typedef names instead of inline struct (lines ~2082 and ~2157)
// Replace:
//     } else if (item->as.function.return_type == TYPE_TUPLE && item->as.function.return_type_info) {
//         TypeInfo *info = item->as.function.return_type_info;
//         sb_append(sb, "struct { ");
//         for (int i = 0; i < info->tuple_element_count; i++) {
//             if (i > 0) sb_append(sb, "; ");
//             sb_appendf(sb, "%s _%d", type_to_c(info->tuple_types[i]), i);
//         }
//         sb_append(sb, "; }");
//     }

// With:
//     } else if (item->as.function.return_type == TYPE_TUPLE && item->as.function.return_type_info) {
//         const char *typedef_name = register_tuple_type(tuple_registry, 
//                                                       item->as.function.return_type_info);
//         sb_append(sb, typedef_name);
//     }

// 5. Free registry at end (after line ~2380)
free_tuple_type_registry(tuple_registry);
```

### Testing

After completing the integration:
```bash
./bin/nanoc tests/tuple_basic.nano -o /tmp/test_tuple
/tmp/test_tuple  # Should execute successfully
```

### Generated C Code Example

**Before** (breaks compilation):
```c
// Forward declaration
struct { int64_t _0; int64_t _1; } nl_get_pair();

// Definition  
struct { int64_t _0; int64_t _1; } nl_get_pair() {  // ERROR: Different anonymous struct type!
    return (struct { int64_t _0; int64_t _1; }){._0 = 100LL, ._1 = 200LL};
}
```

**After** (compiles successfully):
```c
// Typedef
typedef struct { int64_t _0; int64_t _1; } Tuple_int_int_0;

// Forward declaration
Tuple_int_int_0 nl_get_pair();

// Definition
Tuple_int_int_0 nl_get_pair() {
    return (Tuple_int_int_0){._0 = 100LL, ._1 = 200LL};
}
```

### Benefits

✅ Tuples work in interpreter (100% complete)  
✅ Tuple variables work in compiler  
✅ Tuple literals work in compiler  
✅ Tuple index access works in compiler  
⚠️  Tuple return types need typedef integration (5% remaining)

### Files Modified

- `src/nanolang.h` - Added `return_type_info` to function AST
- `src/parser.c` - Capture TypeInfo for tuple returns  
- `src/transpiler.c` - Added TupleTypeRegistry, needs integration

### Integration Location

The main work needs to happen in `src/transpiler.c`, function `transpile_program()`, starting around line ~1950-2380.

All infrastructure is in place. Just needs the 4-5 integration points listed above.

