# nanolang Module System Analysis

## Current Implementation Status

### What We Have

1. **Source-based Module Loading**
   - Modules are loaded by reading `.nano` source files
   - Modules are parsed, type-checked, and their symbols are added to the environment
   - Type information exists **only in memory** during compilation/interpretation
   - Functions, structs, enums, and unions from modules are registered in the `Environment` structure

2. **Static Linking**
   - Modules can be compiled to `.o` object files
   - Object files are linked with the main program
   - **Problem**: No type information is stored in the `.o` files

3. **Basic FFI Support**
   - `extern fn` declarations allow calling C functions
   - Type mapping exists for basic types (int, string, bool, float)
   - SDL-specific type handling exists as a special case

### Critical Gaps

## 1. Metadata Storage in Binary Modules

**Current State**: ❌ **NO METADATA STORAGE**

When we compile a module to `.o`:
- We transpile the nanolang code to C
- We compile the C to an object file
- **We lose all type information** - the `.o` file only contains compiled machine code

**What Should Happen**:
- Type information (function signatures, struct definitions, enum definitions) should be serialized into the module
- This could be done via:
  1. **Embedded metadata section**: Store type information as static data structures in the C code
  2. **Separate metadata file**: Generate a `.nano.meta` file alongside `.o` files
  3. **DWARF debug info**: Use compiler debug information (but this is C-level, not nanolang-level)

**Current Workaround**:
- We re-read the source `.nano` file every time we need type information
- This works but defeats the purpose of compiled modules
- Source files must be available at compile time

## 2. Type Information Serialization

**Current State**: ❌ **NO SERIALIZATION**

The `Environment` structure contains:
- `Function` array with signatures
- `StructDef` array with struct definitions
- `EnumDef` array with enum definitions
- `UnionDef` array with union definitions

**None of this is serialized into compiled modules.**

**What Should Happen**:
```c
// Example: Serialized module metadata structure
typedef struct {
    char module_name[256];
    int function_count;
    FunctionSignature functions[MAX_FUNCTIONS];
    int struct_count;
    StructDef structs[MAX_STRUCTS];
    // ... etc
} ModuleMetadata;
```

This metadata should be:
1. Generated during module compilation
2. Embedded as static data in the `.o` file
3. Readable at import time to reconstruct the environment

## 3. Module Import Code

**Current State**: ⚠️ **PARTIALLY IMPLEMENTED**

`process_imports()` currently:
- Reads source `.nano` files
- Parses and type-checks them
- Adds symbols to environment

**What's Missing**:
- No code to read metadata from compiled `.o` files
- No deserialization of type information
- Cannot import from compiled modules without source

**What Should Happen**:
```c
// Pseudo-code for proper module import
bool import_compiled_module(const char *module_path, Environment *env) {
    // 1. Try to find compiled .o file
    // 2. Extract metadata section from .o file
    // 3. Deserialize ModuleMetadata
    // 4. Register all functions, structs, enums in environment
    // 5. Link the .o file
}
```

## 4. FFI Mechanism Audit

### Current FFI Capabilities

**What Works**:
- ✅ Basic type mapping: `int`, `string`, `bool`, `float`
- ✅ Function declarations: `extern fn function_name(...) -> return_type`
- ✅ SDL-specific type handling (hardcoded special cases)

**What's Missing**:
- ❌ **No C header parsing**: `nanoc-ffi` only generates templates
- ❌ **No automatic type mapping**: C types must be manually mapped
- ❌ **No C struct support**: Cannot use C structs as parameters/return types
- ❌ **No C enum support**: Cannot use C enums
- ❌ **No pointer type handling**: Cannot handle `int*`, `char*`, `void*` properly
- ❌ **No array type handling**: Cannot handle C arrays
- ❌ **No function pointer support**: Cannot pass function pointers to C

### What's Needed for Genuine FFI

1. **C Header Parser**
   - Parse C function declarations
   - Extract parameter types and return types
   - Handle C structs, enums, unions
   - Generate nanolang `extern fn` declarations

2. **Type Mapping System**
   - Map C types to nanolang types
   - Handle opaque types (e.g., `SDL_Window*` -> `int64_t` or custom type)
   - Support type aliases and typedefs

3. **Wrapper Generation**
   - Generate nanolang wrapper functions for C functions
   - Handle type conversions
   - Handle memory management

4. **Module Metadata for C Libraries**
   - Store C type information in module metadata
   - Include library paths, include paths, linker flags
   - Store function signatures with C types

## 5. Module System Trade-offs

### For nanolang-only Modules

**Current Simplicity**:
- ✅ No header files needed
- ✅ Import entire module with one line: `import "module.nano"`
- ✅ All symbols available immediately
- ✅ Type checking happens at import time

**Trade-offs**:
- ⚠️ Source files must be available (no true binary modules)
- ⚠️ Slower compilation (must parse source every time)
- ⚠️ No versioning or module boundaries

**What Programmers Get**:
- Simpler than C: no separate header/implementation files
- More explicit than Python: types are checked at compile time
- Less flexible than Go: no package versioning or module boundaries

### For C FFI Modules

**Current Complexity**:
- ❌ Must manually write `extern fn` declarations
- ❌ Must manually map C types
- ❌ Must understand C type system
- ❌ No tooling to automate this

**What Should Be**:
- ✅ Run `nanoc-ffi SDL.h -o sdl.nano`
- ✅ Get complete nanolang module with all SDL functions
- ✅ Use SDL types naturally in nanolang
- ✅ Compiler handles all type conversions

## Recommendations

### Immediate Fixes Needed

1. **Implement Metadata Serialization**
   ```c
   // In module.c
   bool serialize_module_metadata(ASTNode *module_ast, Environment *env, const char *output_file);
   ModuleMetadata *deserialize_module_metadata(const char *module_file);
   ```

2. **Update Module Import**
   ```c
   // Try compiled module first, fall back to source
   bool import_module(const char *module_path, Environment *env) {
       // 1. Try .nano.meta or embedded metadata
       // 2. If found, deserialize and register symbols
       // 3. If not found, read source and compile
   }
   ```

3. **Complete FFI Tool**
   - Implement C header parser in `nanoc-ffi`
   - Generate proper `extern fn` declarations
   - Generate type mapping code
   - Generate module metadata

### Long-term Improvements

1. **Module Versioning**: Support module versions and dependencies
2. **Module Caching**: Cache compiled modules and metadata
3. **Type System Extensions**: Support opaque types, function pointers
4. **Better Error Messages**: Clear errors when types don't match

## Conclusion

**Current State**: We have a **source-based module system** that works for nanolang-to-nanolang imports, but **not a true binary module system** and **not a complete FFI mechanism**.

**What's Needed**:
1. Metadata serialization/deserialization
2. Compiled module import support
3. Complete C header parser
4. Comprehensive type mapping system

The foundation is there, but we need to complete the metadata storage and FFI tooling to make this a genuine module system that can handle both nanolang modules and C library bindings.

