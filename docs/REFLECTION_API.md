# Struct Reflection API

## Overview

NanoLang provides **compile-time struct introspection** through auto-generated reflection functions. This enables runtime inspection of struct fields, types, and metadata without manual maintenance.

**Use Cases:**
- JSON serializers/deserializers
- Database ORMs
- Configuration parsers
- Debug printers
- Validation frameworks
- Test frameworks

---

## How It Works

### Automatic Generation

For every struct in your program, the compiler automatically generates 5 reflection functions:

```nano
struct Point {
    x: int,
    y: int,
    label: string
}

/* Compiler auto-generates these functions: */

// 1. Get field count
fn ___reflect_Point_field_count() -> int

// 2. Get field name by index
fn ___reflect_Point_field_name(index: int) -> string

// 3. Get field type by index
fn ___reflect_Point_field_type(index: int) -> string

// 4. Check if field exists
fn ___reflect_Point_has_field(name: string) -> bool

// 5. Get field type by name
fn ___reflect_Point_field_type_by_name(name: string) -> string
```

---

## API Reference

### Field Count

```nano
extern fn ___reflect_<StructName>_field_count() -> int
```

Returns the number of fields in the struct.

**Example:**
```nano
extern fn ___reflect_Point_field_count() -> int

fn main() -> int {
    let count: int = (___reflect_Point_field_count)
    (println (int_to_string count))  // Output: 3
    return 0
}
```

---

### Field Name by Index

```nano
extern fn ___reflect_<StructName>_field_name(index: int) -> string
```

Returns the name of the field at the given index (0-based).

**Example:**
```nano
extern fn ___reflect_Point_field_name(index: int) -> string

fn main() -> int {
    let name0: string = (___reflect_Point_field_name 0)
    (println name0)  // Output: x
    
    let name1: string = (___reflect_Point_field_name 1)
    (println name1)  // Output: y
    
    return 0
}
```

**Returns:** Field name as string, or `""` if index out of bounds.

---

### Field Type by Index

```nano
extern fn ___reflect_<StructName>_field_type(index: int) -> string
```

Returns the type of the field at the given index as a string.

**Type Representations:**
- Primitives: `"int"`, `"float"`, `"bool"`, `"string"`, `"void"`
- Structs: `"StructName"` (e.g., `"Point"`)
- Arrays: `"array<T>"` (e.g., `"array<int>"`)

**Example:**
```nano
extern fn ___reflect_Point_field_type(index: int) -> string

fn main() -> int {
    let type0: string = (___reflect_Point_field_type 0)
    (println type0)  // Output: int
    
    let type2: string = (___reflect_Point_field_type 2)
    (println type2)  // Output: string
    
    return 0
}
```

---

### Has Field

```nano
extern fn ___reflect_<StructName>_has_field(name: string) -> bool
```

Checks if a field with the given name exists.

**Example:**
```nano
extern fn ___reflect_Point_has_field(name: string) -> bool

fn main() -> int {
    let has_x: bool = (___reflect_Point_has_field "x")
    let has_z: bool = (___reflect_Point_has_field "z")
    
    (println (if has_x { "Has x" } else { "No x" }))  // Output: Has x
    (println (if has_z { "Has z" } else { "No z" }))  // Output: No z
    
    return 0
}
```

---

### Field Type by Name

```nano
extern fn ___reflect_<StructName>_field_type_by_name(name: string) -> string
```

Returns the type of a field given its name.

**Example:**
```nano
extern fn ___reflect_Point_field_type_by_name(name: string) -> string

fn main() -> int {
    let x_type: string = (___reflect_Point_field_type_by_name "x")
    (println x_type)  // Output: int
    
    let label_type: string = (___reflect_Point_field_type_by_name "label")
    (println label_type)  // Output: string
    
    return 0
}
```

**Returns:** Field type as string, or `""` if field not found.

---

## Complete Example

```nano
struct Person {
    name: string,
    age: int,
    active: bool
}

/* Declare extern functions */
extern fn ___reflect_Person_field_count() -> int
extern fn ___reflect_Person_field_name(index: int) -> string
extern fn ___reflect_Person_field_type(index: int) -> string
extern fn ___reflect_Person_has_field(name: string) -> bool
extern fn ___reflect_Person_field_type_by_name(name: string) -> string

fn print_struct_info() -> int {
    (println "=== Person Struct Metadata ===")
    
    /* Print field count */
    let count: int = (___reflect_Person_field_count)
    (print "Fields: ")
    (println (int_to_string count))
    
    /* Iterate over all fields */
    let mut i: int = 0
    while (< i count) {
        let fname: string = (___reflect_Person_field_name i)
        let ftype: string = (___reflect_Person_field_type i)
        
        (print "  - ")
        (print fname)
        (print ": ")
        (println ftype)
        
        set i (+ i 1)
    }
    
    /* Check specific fields */
    let has_name: bool = (___reflect_Person_has_field "name")
    let has_salary: bool = (___reflect_Person_has_field "salary")
    
    (println (if has_name { "✓ Has 'name' field" } else { "✗ No 'name' field" }))
    (println (if has_salary { "✓ Has 'salary' field" } else { "✗ No 'salary' field" }))
    
    return 0
}

shadow print_struct_info {
    assert (== (print_struct_info) 0)
}

fn main() -> int {
    return (print_struct_info)
}

shadow main {
    assert (== (main) 0)
}
```

**Output:**
```
=== Person Struct Metadata ===
Fields: 3
  - name: string
  - age: int
  - active: bool
✓ Has 'name' field
✗ No 'salary' field
```

---

## Advanced Use Case: Generic Debug Printer

```nano
struct Config {
    host: string,
    port: int,
    enabled: bool
}

extern fn ___reflect_Config_field_count() -> int
extern fn ___reflect_Config_field_name(index: int) -> string
extern fn ___reflect_Config_field_type(index: int) -> string

fn debug_print_config(cfg: Config) -> int {
    (println "Config {")
    
    let count: int = (___reflect_Config_field_count)
    let mut i: int = 0
    while (< i count) {
        let fname: string = (___reflect_Config_field_name i)
        let ftype: string = (___reflect_Config_field_type i)
        
        (print "  ")
        (print fname)
        (print " (")
        (print ftype)
        (print "): ")
        
        /* NOTE: Actual field VALUE access requires manual handling
         * Reflection only provides metadata, not runtime value access */
        (println "<value>")
        
        set i (+ i 1)
    }
    
    (println "}")
    return 0
}
```

---

## Limitations

### 1. **No Runtime Field Access**
Reflection provides **metadata only** (field names and types). It does NOT provide runtime field value access:

```nano
/* ❌ NOT SUPPORTED */
fn get_field_value(obj: any, field_name: string) -> any { ... }

/* ✅ SUPPORTED */
fn get_field_type(struct_name: string, field_name: string) -> string { ... }
```

**Workaround:** Use code generation or macros for value access.

---

### 2. **Requires Explicit Extern Declarations**
You must declare each reflection function as `extern`:

```nano
extern fn ___reflect_Point_field_count() -> int
```

**Future Enhancement:** Macro system could auto-generate these declarations.

---

### 3. **Type Information is String-Based**
Field types are returned as strings (`"int"`, `"string"`), not type objects.

**Workaround:** Parse type strings or use pattern matching.

---

## Performance

- **Zero Runtime Overhead:** Reflection functions are `inline` in generated C
- **No Memory Allocation:** Returns static string literals
- **Compile-Time Only:** No reflection data structures in compiled binary
- **Call Cost:** O(1) for field count/type, O(n) for field name lookup

---

## Implementation Details

### Generated C Code

For `struct Point { x: int, y: int, label: string }`, the compiler generates:

```c
inline int64_t ___reflect_Point_field_count(void) {
    return 3;
}

inline const char* ___reflect_Point_field_name(int64_t index) {
    if (index == 0) { return "x"; }
    else if (index == 1) { return "y"; }
    else if (index == 2) { return "label"; }
    else { return ""; }
}

inline const char* ___reflect_Point_field_type(int64_t index) {
    if (index == 0) { return "int"; }
    else if (index == 1) { return "int"; }
    else if (index == 2) { return "string"; }
    else { return ""; }
}

inline bool ___reflect_Point_has_field(const char* name) {
    if (strcmp(name, "x") == 0) { return 1; }
    else if (strcmp(name, "y") == 0) { return 1; }
    else if (strcmp(name, "label") == 0) { return 1; }
    else { return 0; }
}

inline const char* ___reflect_Point_field_type_by_name(const char* name) {
    if (strcmp(name, "x") == 0) { return "int"; }
    else if (strcmp(name, "y") == 0) { return "int"; }
    else if (strcmp(name, "label") == 0) { return "string"; }
    else { return ""; }
}
```

---

## Future Enhancements

### 1. **Attribute-Based Reflection**
```nano
@derive(Reflect)
struct Point {
    x: int,
    y: int
}
```

### 2. **Generic Metadata Struct**
```nano
struct FieldInfo {
    name: string,
    type_name: string,
    offset: int
}

fn ___reflect_Point() -> array<FieldInfo> { ... }
```

### 3. **Runtime Field Access** (Advanced)
```nano
fn get_field(obj: any, field_name: string) -> any { ... }
fn set_field(obj: any, field_name: string, value: any) -> void { ... }
```

---

## FAQ

**Q: Can I iterate over all structs in my program?**  
A: No, only per-struct reflection is provided. Use code generation for cross-struct iteration.

**Q: Does this work with imported modules?**  
A: Yes! Reflection functions are generated for ALL structs, including imported ones.

**Q: What about private fields?**  
A: All fields are reflected, regardless of visibility. NanoLang doesn't have field-level privacy yet.

**Q: Can I modify reflection behavior?**  
A: Not directly. Reflection is automatic and uniform across all structs.

---

## Version History

- **v0.1.0** (2025-01-07): Initial implementation
  - Auto-generated reflection functions
  - 5 functions per struct
  - Zero runtime overhead

---

## See Also

- [Language Spec](../spec.json)
- [LLM Core Subset](./LLM_CORE_SUBSET.md)
- [Canonical Style Guide](./CANONICAL_STYLE.md)

---

**Status:** ✅ **Production Ready** (Reference Compiler)  
**Self-Hosted:** ⚠️ **90% Complete** (128 type errors remain in self-hosted compiler, but reflection system itself works)
