# Memory Management in Nanolang

## Overview

Nanolang uses a combination of manual memory management (C-style) and automatic garbage collection for different subsystems.

## Memory Subsystems

### 1. Garbage Collector (Runtime)

**Location**: `src/runtime/gc.c`, `src/runtime/gc.h`

The GC manages runtime heap objects using reference counting with cycle detection.

**Managed Types**:
- `DynArray` - Dynamic arrays
- `GCStruct` - Runtime structs
- Strings (when heap-allocated)
- Closures

**API**:
```c
void* gc_alloc(size_t size, GCObjectType type);  // Allocate GC-managed object
void gc_retain(void* ptr);                        // Increment ref count
void gc_release(void* ptr);                       // Decrement ref count (may free)
void gc_collect_cycles();                         // Run cycle detection
```

**Ownership Model**:
- Objects start with refcount=1
- Each reference increments refcount
- When refcount reaches 0, object is freed
- Cycle detector handles circular references

### 2. Compiler Memory (Manual Management)

**Location**: `src/transpiler.c`, `src/parser.c`, `src/typechecker.c`

Compiler data structures use manual `malloc/free`.

**Key Structures**:
- **AST Nodes** - Allocated during parsing, freed after transpilation
- **Symbol Tables** - Environment structures with chained scopes
- **String Builders** - Dynamic string buffers for code generation
- **Type Registries** - Function and tuple type tracking

**Ownership Model**:
- Parser owns AST until transpiler completes
- Environment owns symbols until typecheck completes
- StringBuilder owns buffer until sb_to_string()
- Caller must free returned strings

**Potential Optimizations**:
1. **Arena Allocators** - Bulk allocate/free for AST nodes
2. **String Interning** - Reuse common strings (keywords, built-in types)
3. **Pool Allocators** - Small object pools for frequent allocations

### 3. Module System Memory

**Location**: `src/module.c`, `src/module_builder.c`

Module metadata and build information.

**Managed Objects**:
- `ModuleBuildMetadata` - Loaded from module.json
- `ModuleBuildInfo` - Compilation results
- Dependency graphs

**Ownership Model**:
- Module loader owns metadata until module_metadata_free()
- Build info owned by caller
- Arrays (c_sources, headers, etc.) owned by metadata struct

## Current Memory Usage

### Typical Profiles

**Small Program (< 100 lines)**:
- Compiler: ~2-5 MB peak RSS
- Interpreter: ~1-3 MB peak RSS

**Self-Hosting (parser_mvp.nano ~2400 lines)**:
- Compiler: ~15-25 MB peak RSS
- Interpreter: ~8-15 MB peak RSS

**Large SDL Program (1000+ lines)**:
- Compiler: ~20-40 MB peak RSS
- Interpreter: ~10-20 MB peak RSS

### Memory Hotspots

1. **AST Allocation** - Parser creates many small nodes
2. **String Building** - Transpiler generates large C code strings
3. **Type Registry** - Stores all generic type instantiations
4. **Dynamic Arrays** - Runtime array growth

## Best Practices

### For Compiler Development

1. **Free AST After Use**
```c
AST *program = parse(tokens);
// Use program
free_ast(program);  // Always free when done
```

2. **Use StringBuilder for Code Gen**
```c
StringBuilder *sb = sb_create();
sb_append(sb, "code");
char *result = sb_to_string(sb);  // Caller must free result
sb_free(sb);
```

3. **Clean Up Environments**
```c
Environment *env = create_environment();
// Use env
free_environment(env);  // Frees all symbols and child scopes
```

### For Runtime Development

1. **Use GC for Heap Objects**
```c
DynArray *arr = gc_alloc(sizeof(DynArray), GC_TYPE_ARRAY);
// arr starts with refcount=1
gc_release(arr);  // Decrement when done
```

2. **Retain Shared References**
```c
DynArray *arr = some_function();  // Returns arr with refcount=1
gc_retain(arr);  // Increment if storing elsewhere
// ...
gc_release(arr);  // Release when done
```

3. **Avoid Cycles When Possible**
```c
// If creating circular references, rely on cycle detector
// or break cycles manually before releasing
```

## Memory Leak Prevention

### Checklist for New Features

- [ ] All `malloc()` calls have corresponding `free()`
- [ ] All `gc_alloc()` objects are eventually `gc_release()`d
- [ ] Strings returned from functions are documented for ownership
- [ ] Error paths also free allocated memory
- [ ] Test with AddressSanitizer (`make sanitize`)

### Debugging Memory Issues

**AddressSanitizer** (use-after-free, buffer overflow):
```bash
make sanitize
make test  # With ASAN_OPTIONS=detect_leaks=0 for exit-time leaks
```

**Valgrind** (memory leaks, Linux only):
```bash
make valgrind
```

**GC Statistics**:
```c
gc_print_stats();  // Print current memory usage
GCStats stats = gc_get_stats();
printf("Memory usage: %zu bytes\\n", stats.current_usage);
```

## Known Limitations

1. **No String Deduplication** - Identical strings are stored separately
2. **No Arena Allocators** - Many small allocations for AST nodes
3. **StringBuilder Reallocation** - Frequent grows during code generation
4. **Module Metadata Caching** - Re-parses module.json on each compilation

## Future Optimizations

### High Impact

1. **Arena Allocator for AST** - Bulk allocate/free entire parse tree
   - Estimated: 30-50% reduction in parser allocations
   - Complexity: Medium

2. **String Interning** - Reuse common strings
   - Estimated: 10-20% memory reduction for identifiers
   - Complexity: Low

3. **StringBuilder Preallocation** - Size hints based on input size
   - Estimated: Fewer reallocations, 5-10% speedup
   - Complexity: Low

### Medium Impact

4. **Module Metadata Caching** - Cache parsed module.json
   - Estimated: Faster multi-module compilation
   - Complexity: Medium

5. **Type Registry Optimization** - Hash table instead of linear search
   - Estimated: Better scaling for generic-heavy code
   - Complexity: Medium

6. **Pool Allocator for Small Objects** - Reuse Symbol, Token, etc.
   - Estimated: 15-25% reduction in allocator overhead
   - Complexity: High

## Measurement Tools

**Benchmark Suite**:
```bash
make benchmark  # Measures compiler memory usage
```

**Manual Profiling** (macOS):
```bash
/usr/bin/time -l ./bin/nanoc large_file.nano
# Check "maximum resident set size"
```

**Manual Profiling** (Linux):
```bash
/usr/bin/time -v ./bin/nanoc large_file.nano
# Check "Maximum resident set size"
```

**GC Profiling**:
```c
// In nanolang code
gc_stats()  // Built-in function (if enabled)
```

## Contributing

When adding features that allocate memory:

1. Document ownership in function comments
2. Add AddressSanitizer test to CI
3. Update this document if adding new subsystems
4. Profile memory usage for large inputs

For questions about memory management, see:
- GC implementation: `src/runtime/gc.c`
- Compiler memory: `src/transpiler.c`
- Module system: `src/module_builder.c`

