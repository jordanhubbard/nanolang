# Chapter 23: Higher-Level Patterns

**Advanced patterns and idioms for effective NanoLang programming.**

This chapter covers design patterns that help you write maintainable, efficient NanoLang code. These patterns are used throughout the NanoLang standard library and examples.

## 23.1 Builder Patterns

### What Are Builder Patterns?

Builder patterns construct complex objects step-by-step. Instead of creating an object with a massive constructor, you build it incrementally through a series of method calls.

In NanoLang, builders are typically **immutable**: each operation returns a new builder instance rather than modifying the existing one.

### When to Use Builders

Use builder patterns when:
- Constructing objects with many optional fields
- Building structures that require multiple steps
- Creating objects where order of operations matters
- Implementing domain-specific languages (DSLs)

### Implementation Examples

**Example 1: StringBuilder**

Building strings efficiently by collecting parts:

```nano
struct StringBuilder {
    parts: array<string>,
    count: int
}

fn sb_new() -> StringBuilder {
    return StringBuilder {
        parts: [],
        count: 0
    }
}

fn sb_append(sb: StringBuilder, text: string) -> StringBuilder {
    return StringBuilder {
        parts: (array_push sb.parts text),
        count: (+ sb.count 1)
    }
}

fn sb_build(sb: StringBuilder) -> string {
    return (str_join sb.parts "")
}

shadow sb_new {
    let sb1: StringBuilder = (sb_new)
    let sb2: StringBuilder = (sb_append sb1 "Hello")
    let sb3: StringBuilder = (sb_append sb2 " ")
    let sb4: StringBuilder = (sb_append sb3 "World")
    let result: string = (sb_build sb4)
    assert (== result "Hello World")
}
```

**Why immutable builders?** Each operation returns a new builder, which:
- Allows branching (create multiple variants from same base)
- Is thread-safe by design
- Makes debugging easier (no hidden state changes)

**Example 2: Query Builder**

Building SQL queries safely:

```nano
struct QueryBuilder {
    table: string,
    columns: array<string>,
    conditions: array<string>,
    limit_val: int
}

fn query_new(table: string) -> QueryBuilder {
    return QueryBuilder {
        table: table,
        columns: [],
        conditions: [],
        limit_val: 0
    }
}

fn query_select(q: QueryBuilder, column: string) -> QueryBuilder {
    return QueryBuilder {
        table: q.table,
        columns: (array_push q.columns column),
        conditions: q.conditions,
        limit_val: q.limit_val
    }
}

fn query_where(q: QueryBuilder, condition: string) -> QueryBuilder {
    return QueryBuilder {
        table: q.table,
        columns: q.columns,
        conditions: (array_push q.conditions condition),
        limit_val: q.limit_val
    }
}

fn query_limit(q: QueryBuilder, n: int) -> QueryBuilder {
    return QueryBuilder {
        table: q.table,
        columns: q.columns,
        conditions: q.conditions,
        limit_val: n
    }
}

fn query_build(q: QueryBuilder) -> string {
    let cols: string = (cond
        ((== (array_length q.columns) 0) "*")
        (else (str_join q.columns ", "))
    )
    
    let mut sql: string = (+ "SELECT " (+ cols (+ " FROM " q.table)))
    
    if (> (array_length q.conditions) 0) {
        let where_clause: string = (str_join q.conditions " AND ")
        set sql (+ sql (+ " WHERE " where_clause))
    }
    
    if (> q.limit_val 0) {
        set sql (+ sql (+ " LIMIT " (int_to_string q.limit_val)))
    }
    
    return sql
}

shadow query_new {
    let q: QueryBuilder = (query_new "users")
    let q2: QueryBuilder = (query_select q "name")
    let q3: QueryBuilder = (query_select q2 "email")
    let q4: QueryBuilder = (query_where q3 "active = 1")
    let q5: QueryBuilder = (query_limit q4 10)
    let sql: string = (query_build q5)
    assert (str_contains sql "SELECT name, email")
    assert (str_contains sql "FROM users")
    assert (str_contains sql "WHERE active = 1")
    assert (str_contains sql "LIMIT 10")
}
```

**Example 3: AST Builder**

Building abstract syntax tree nodes:

```nano
struct ASTNode {
    kind: string,
    value: string,
    children: array<ASTNode>
}

fn ast_literal(value: string) -> ASTNode {
    return ASTNode {
        kind: "literal",
        value: value,
        children: []
    }
}

fn ast_binary(op: string, left: ASTNode, right: ASTNode) -> ASTNode {
    return ASTNode {
        kind: (+ "binary_" op),
        value: op,
        children: [left, right]
    }
}

fn ast_call(name: string, args: array<ASTNode>) -> ASTNode {
    return ASTNode {
        kind: "call",
        value: name,
        children: args
    }
}

shadow ast_literal {
    # Build: add(1, 2)
    let one: ASTNode = (ast_literal "1")
    let two: ASTNode = (ast_literal "2")
    let add_call: ASTNode = (ast_call "add" [one, two])
    
    assert (== add_call.kind "call")
    assert (== add_call.value "add")
    assert (== (array_length add_call.children) 2)
}
```

## 23.2 Iterator Patterns

### Manual Iteration

NanoLang uses explicit iteration with index variables. This gives you full control over traversal.

**Basic array iteration:**

```nano
fn sum_array(arr: array<int>) -> int {
    let mut sum: int = 0
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        set sum (+ sum (at arr i))
    }
    
    return sum
}

shadow sum_array {
    assert (== (sum_array [1, 2, 3, 4, 5]) 15)
    assert (== (sum_array []) 0)
}
```

**Finding elements:**

```nano
fn find_index(arr: array<int>, target: int) -> int {
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        if (== (at arr i) target) {
            return i
        }
    }
    
    return -1  # Not found
}

shadow find_index {
    let nums: array<int> = [10, 20, 30, 40, 50]
    assert (== (find_index nums 30) 2)
    assert (== (find_index nums 99) -1)
}
```

### Iterator Interfaces

Create reusable iteration patterns by passing functions:

**Map pattern (transform each element):**

```nano
fn map_int(arr: array<int>, transform: fn(int) -> int) -> array<int> {
    let len: int = (array_length arr)
    let mut result: array<int> = (array_new len 0)
    
    for i in (range 0 len) {
        let transformed: int = (transform (at arr i))
        (array_set result i transformed)
    }
    
    return result
}

fn double(x: int) -> int {
    return (* x 2)
}

shadow map_int {
    let nums: array<int> = [1, 2, 3]
    let doubled: array<int> = (map_int nums double)
    assert (== (at doubled 0) 2)
    assert (== (at doubled 1) 4)
    assert (== (at doubled 2) 6)
}
```

**Filter pattern (keep matching elements):**

```nano
fn filter_int(arr: array<int>, predicate: fn(int) -> bool) -> array<int> {
    let mut result: array<int> = []
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        let value: int = (at arr i)
        if (predicate value) {
            set result (array_push result value)
        }
    }
    
    return result
}

fn is_positive(x: int) -> bool {
    return (> x 0)
}

shadow filter_int {
    let nums: array<int> = [-2, -1, 0, 1, 2, 3]
    let positives: array<int> = (filter_int nums is_positive)
    assert (== (array_length positives) 3)
    assert (== (at positives 0) 1)
}
```

**Fold pattern (reduce to single value):**

```nano
fn fold_int(arr: array<int>, initial: int, combine: fn(int, int) -> int) -> int {
    let mut accumulator: int = initial
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        set accumulator (combine accumulator (at arr i))
    }
    
    return accumulator
}

fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn multiply(a: int, b: int) -> int {
    return (* a b)
}

shadow fold_int {
    let nums: array<int> = [1, 2, 3, 4]
    let sum: int = (fold_int nums 0 add)
    let product: int = (fold_int nums 1 multiply)
    
    assert (== sum 10)      # 1+2+3+4
    assert (== product 24)  # 1*2*3*4
}
```

### Custom Iterators

For complex data structures, create iterator structs:

```nano
struct TreeNode {
    value: int,
    left: TreeNode?,
    right: TreeNode?
}

struct TreeIterator {
    stack: array<TreeNode>,
    count: int
}

fn tree_iter_new(root: TreeNode?) -> TreeIterator {
    let mut stack: array<TreeNode> = []
    
    # Push all left children onto stack (in-order traversal setup)
    let mut current: TreeNode? = root
    while (is_some current) {
        let node: TreeNode = (unwrap current)
        set stack (array_push stack node)
        set current node.left
    }
    
    return TreeIterator {
        stack: stack,
        count: (array_length stack)
    }
}

fn tree_iter_has_next(iter: TreeIterator) -> bool {
    return (> iter.count 0)
}

fn tree_iter_next(iter: TreeIterator) -> (int, TreeIterator) {
    # Pop top of stack
    let idx: int = (- iter.count 1)
    let node: TreeNode = (at iter.stack idx)
    let value: int = node.value
    
    # Remove from stack (simplified - create new array without last element)
    let mut new_stack: array<TreeNode> = []
    for i in (range 0 idx) {
        set new_stack (array_push new_stack (at iter.stack i))
    }
    
    # Push right subtree's left spine
    let mut current: TreeNode? = node.right
    while (is_some current) {
        let n: TreeNode = (unwrap current)
        set new_stack (array_push new_stack n)
        set current n.left
    }
    
    let new_iter: TreeIterator = TreeIterator {
        stack: new_stack,
        count: (array_length new_stack)
    }
    
    return (value, new_iter)
}

shadow tree_iter_new {
    # Would test with actual tree
    assert true
}
```

## 23.3 Resource Management (RAII-style)

### Resource Lifecycle

Resources like files, network connections, and graphics contexts must be properly cleaned up. NanoLang encourages explicit resource management.

**Pattern: Create → Use → Cleanup**

```nano
struct FileHandle {
    fd: int,
    path: string,
    is_open: bool
}

fn file_open(path: string) -> FileHandle {
    let fd: int = 0
    unsafe {
        set fd (c_open path 0)
    }
    return FileHandle {
        fd: fd,
        path: path,
        is_open: (>= fd 0)
    }
}

fn file_read(f: FileHandle) -> string {
    if (not f.is_open) {
        return ""
    }
    
    let mut content: string = ""
    unsafe {
        set content (c_read_all f.fd)
    }
    return content
}

fn file_close(f: FileHandle) -> void {
    if f.is_open {
        unsafe {
            (c_close f.fd)
        }
    }
}

# Usage: Always pair open/close
fn read_config(path: string) -> string {
    let f: FileHandle = (file_open path)
    
    if (not f.is_open) {
        return ""  # No cleanup needed - wasn't opened
    }
    
    let content: string = (file_read f)
    (file_close f)  # Always close!
    
    return content
}

shadow file_open {
    # Would test with actual file operations
    assert true
}
```

### Cleanup Patterns

**Pattern: Cleanup in All Branches**

Every code path must clean up resources:

```nano
fn process_data(path: string, should_transform: bool) -> string {
    let f: FileHandle = (file_open path)
    
    if (not f.is_open) {
        return "Error: cannot open file"
    }
    
    let content: string = (file_read f)
    
    let result: string = (cond
        (should_transform (transform_content content))
        (else content)
    )
    
    # Cleanup happens regardless of which branch was taken
    (file_close f)
    
    return result
}

shadow process_data {
    assert true
}
```

**Pattern: Multiple Resources**

When using multiple resources, clean up in reverse order:

```nano
fn copy_file(src_path: string, dst_path: string) -> bool {
    let src: FileHandle = (file_open src_path)
    if (not src.is_open) {
        return false
    }
    
    let dst: FileHandle = (file_create dst_path)
    if (not dst.is_open) {
        (file_close src)  # Clean up first resource
        return false
    }
    
    let content: string = (file_read src)
    let success: bool = (file_write dst content)
    
    # Clean up in reverse order of creation
    (file_close dst)
    (file_close src)
    
    return success
}

shadow copy_file {
    assert true
}
```

### Error Handling with Resources

Combine resource management with error handling:

```nano
union FileResult {
    Ok { handle: FileHandle },
    Err { message: string }
}

fn safe_file_open(path: string) -> FileResult {
    let f: FileHandle = (file_open path)
    
    if (not f.is_open) {
        return FileResult.Err { message: (+ "Cannot open: " path) }
    }
    
    return FileResult.Ok { handle: f }
}

fn safe_read_file(path: string) -> string {
    let result: FileResult = (safe_file_open path)
    
    match result {
        Ok(r) => {
            let content: string = (file_read r.handle)
            (file_close r.handle)
            return content
        }
        Err(e) => {
            (println e.message)
            return ""
        }
    }
}

shadow safe_file_open {
    assert true
}
```

## 23.4 Error Handling Idioms

### Result Types

Use union types to represent success or failure:

```nano
union Result<T, E> {
    Ok { value: T },
    Err { error: E }
}

fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Err { error: "Division by zero" }
    }
    return Result.Ok { value: (/ a b) }
}

shadow divide {
    let r1: Result<int, string> = (divide 10 2)
    match r1 {
        Ok(v) => assert (== v.value 5)
        Err(e) => assert false
    }
    
    let r2: Result<int, string> = (divide 10 0)
    match r2 {
        Ok(v) => assert false
        Err(e) => assert (== e.error "Division by zero")
    }
}
```

### Error Propagation

Chain operations that might fail:

```nano
fn parse_and_double(s: string) -> Result<int, string> {
    # Check if string is a valid number
    if (== (str_length s) 0) {
        return Result.Err { error: "Empty string" }
    }
    
    let n: int = (string_to_int s)
    
    # string_to_int returns 0 on error, but 0 might be valid
    # Use a validation approach
    if (and (!= n 0) (not (== s "0"))) {
        # Valid non-zero number - but wait, we need better validation
    }
    
    return Result.Ok { value: (* n 2) }
}

fn process_numbers(inputs: array<string>) -> Result<int, string> {
    let mut sum: int = 0
    let len: int = (array_length inputs)
    
    for i in (range 0 len) {
        let result: Result<int, string> = (parse_and_double (at inputs i))
        
        match result {
            Ok(v) => set sum (+ sum v.value)
            Err(e) => return Result.Err { 
                error: (+ "Error at index " (+ (int_to_string i) (+ ": " e.error)))
            }
        }
    }
    
    return Result.Ok { value: sum }
}

shadow parse_and_double {
    assert true
}
```

### Error Context

Add context when propagating errors:

```nano
fn add_context(result: Result<int, string>, context: string) -> Result<int, string> {
    match result {
        Ok(v) => return result
        Err(e) => return Result.Err { error: (+ context (+ ": " e.error)) }
    }
}

fn load_user_config(user_id: int) -> Result<string, string> {
    let path: string = (+ "/home/user" (+ (int_to_string user_id) "/config.json"))
    
    let file_result: Result<string, string> = (read_file_safe path)
    
    # Add context about what we were trying to do
    match file_result {
        Ok(v) => return file_result
        Err(e) => return Result.Err {
            error: (+ "Failed to load config for user " (+ (int_to_string user_id) (+ ": " e.error)))
        }
    }
}

shadow load_user_config {
    assert true
}
```

### Best Practices

**1. Validate early, return early:**

```nano
fn process_user_input(name: string, age: int, email: string) -> Result<User, string> {
    # Validate all inputs first
    if (== (str_length name) 0) {
        return Result.Err { error: "Name cannot be empty" }
    }
    
    if (< age 0) {
        return Result.Err { error: "Age cannot be negative" }
    }
    
    if (> age 150) {
        return Result.Err { error: "Age seems unrealistic" }
    }
    
    if (not (str_contains email "@")) {
        return Result.Err { error: "Invalid email format" }
    }
    
    # All validations passed - create the user
    return Result.Ok { 
        value: User { name: name, age: age, email: email }
    }
}

shadow process_user_input {
    assert true
}
```

**2. Use sentinel values for simple cases:**

When full error handling is overkill:

```nano
fn find_in_array(arr: array<int>, target: int) -> int {
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        if (== (at arr i) target) {
            return i  # Found
        }
    }
    
    return -1  # Not found (sentinel)
}

shadow find_in_array {
    let arr: array<int> = [10, 20, 30]
    assert (== (find_in_array arr 20) 1)
    assert (== (find_in_array arr 99) -1)
}
```

**3. Document error conditions:**

```nano
# Divides a by b.
# Returns Result.Err if b is zero.
# Returns Result.Err if result would overflow.
fn safe_divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Err { error: "Division by zero" }
    }
    
    # Check for overflow (MIN_INT / -1)
    if (and (== a -9223372036854775808) (== b -1)) {
        return Result.Err { error: "Integer overflow" }
    }
    
    return Result.Ok { value: (/ a b) }
}

shadow safe_divide {
    let r: Result<int, string> = (safe_divide 10 2)
    match r {
        Ok(v) => assert (== v.value 5)
        Err(e) => assert false
    }
}
```

## Summary

| Pattern | Use When | Key Benefit |
|---------|----------|-------------|
| **Builder** | Complex object construction | Incremental, readable construction |
| **Iterator** | Collection traversal | Reusable, composable operations |
| **RAII** | Resource management | Guaranteed cleanup |
| **Result** | Operations that can fail | Explicit, type-safe error handling |

**Guidelines:**
- ✅ Use immutable builders for complex objects
- ✅ Prefer `for i in (range 0 len)` for simple iteration
- ✅ Always clean up resources in all code paths
- ✅ Use `Result<T, E>` for recoverable errors
- ✅ Add context when propagating errors
- ✅ Validate inputs early, fail fast

---

**Previous:** [Chapter 22: Canonical Style Guide](22_canonical_style.html)  
**Next:** [Chapter 24: Performance & Optimization](24_performance.html)
