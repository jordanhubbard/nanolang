# Nanolang Compiler Optimizations Design

## Overview

Comprehensive optimization framework for the nanolang compiler to improve runtime performance, reduce binary size, and enable better code generation.

## Goals

1. **Performance**: Faster execution of generated code
2. **Size**: Smaller binary outputs
3. **Efficiency**: Better use of CPU caches and registers
4. **Maintainability**: Modular optimization passes
5. **Debuggability**: Preserve debug info where possible

## Architecture

### Optimization Pipeline

```
┌─────────────┐
│   Parser    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Type Checker│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     AST     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Optimization Passes       │
│                             │
│  1. Constant Folding        │
│  2. Dead Code Elimination   │
│  3. Common Subexpression    │
│  4. Tail Call Optimization  │
│  5. Loop Optimization       │
│  6. Inline Expansion        │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────┐
│  Optimized  │
│     AST     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Transpiler │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   C Code    │
└─────────────┘
```

## Optimization #1: Constant Folding

### Description

Evaluate compile-time constant expressions and replace them with their results.

### Examples

**Before:**
```nano
let x: int = (+ 2 3)
let y: int = (* 4 5)
let z: int = (+ x y)
```

**After:**
```nano
let x: int = 5
let y: int = 20
let z: int = 25
```

### Implementation

```c
typedef struct {
    bool is_constant;
    Value constant_value;
    ASTNode* optimized_node;
} ConstFoldResult;

ConstFoldResult constant_fold_expr(ASTNode* node, Environment* const_env) {
    switch (node->type) {
        case NODE_LITERAL:
            // Already constant
            return (ConstFoldResult){
                .is_constant = true,
                .constant_value = node->as.literal.value,
                .optimized_node = node
            };
            
        case NODE_BINARY_OP: {
            // Fold operands first
            ConstFoldResult left = constant_fold_expr(node->as.binop.left, const_env);
            ConstFoldResult right = constant_fold_expr(node->as.binop.right, const_env);
            
            // If both are constants, evaluate
            if (left.is_constant && right.is_constant) {
                Value result = evaluate_binop(
                    node->as.binop.op,
                    left.constant_value,
                    right.constant_value
                );
                
                return (ConstFoldResult){
                    .is_constant = true,
                    .constant_value = result,
                    .optimized_node = create_literal_node(result)
                };
            }
            
            // Return with optimized children
            node->as.binop.left = left.optimized_node;
            node->as.binop.right = right.optimized_node;
            return (ConstFoldResult){
                .is_constant = false,
                .optimized_node = node
            };
        }
        
        case NODE_UNARY_OP: {
            ConstFoldResult operand = constant_fold_expr(node->as.unop.operand, const_env);
            
            if (operand.is_constant) {
                Value result = evaluate_unop(node->as.unop.op, operand.constant_value);
                return (ConstFoldResult){
                    .is_constant = true,
                    .constant_value = result,
                    .optimized_node = create_literal_node(result)
                };
            }
            
            node->as.unop.operand = operand.optimized_node;
            return (ConstFoldResult){.is_constant = false, .optimized_node = node};
        }
        
        case NODE_VAR_REF: {
            // Check if variable is constant
            Value* val = env_lookup_const(const_env, node->as.var_ref.name);
            if (val) {
                return (ConstFoldResult){
                    .is_constant = true,
                    .constant_value = *val,
                    .optimized_node = create_literal_node(*val)
                };
            }
            return (ConstFoldResult){.is_constant = false, .optimized_node = node};
        }
        
        default:
            return (ConstFoldResult){.is_constant = false, .optimized_node = node};
    }
}
```

### Benefits

- **Reduced runtime computation**: Constants computed once at compile-time
- **Smaller code**: Fewer instructions in generated C
- **Better optimization**: C compiler can further optimize

### Challenges

- **Overflow handling**: Must match runtime overflow behavior
- **Floating-point precision**: Ensure consistency with runtime
- **Const propagation**: Track which variables are effectively constant

## Optimization #2: Dead Code Elimination

### Description

Remove code that is never executed or whose results are never used.

### Examples

#### Unreachable Code

**Before:**
```nano
fn example() -> int {
    return 42
    (println "This never runs")  // Dead code
}
```

**After:**
```nano
fn example() -> int {
    return 42
}
```

#### Unused Variables

**Before:**
```nano
fn calculate(x: int) -> int {
    let unused: int = (* x 2)  // Never used
    let result: int = (+ x 5)
    return result
}
```

**After:**
```nano
fn calculate(x: int) -> int {
    let result: int = (+ x 5)
    return result
}
```

#### Dead Branches

**Before:**
```nano
if true {
    (do_something)
} else {
    (never_runs)  // Dead branch
}
```

**After:**
```nano
(do_something)
```

### Implementation

```c
typedef struct {
    bool* var_used;        // Track used variables
    bool has_return;       // Track if path has return
    bool is_reachable;     // Track reachable code
} DCEContext;

ASTNode* eliminate_dead_code(ASTNode* node, DCEContext* ctx) {
    switch (node->type) {
        case NODE_BLOCK: {
            ASTNode** new_stmts = malloc(node->as.block.count * sizeof(ASTNode*));
            int new_count = 0;
            
            for (int i = 0; i < node->as.block.count; i++) {
                if (!ctx->is_reachable) {
                    // Skip unreachable code
                    continue;
                }
                
                ASTNode* stmt = eliminate_dead_code(node->as.block.stmts[i], ctx);
                if (stmt) {
                    new_stmts[new_count++] = stmt;
                }
                
                // Check if this statement makes rest unreachable
                if (stmt->type == NODE_RETURN || stmt->type == NODE_BREAK) {
                    ctx->is_reachable = false;
                }
            }
            
            node->as.block.stmts = new_stmts;
            node->as.block.count = new_count;
            return node;
        }
        
        case NODE_IF: {
            // Check if condition is constant
            if (is_constant_expr(node->as.if_stmt.condition)) {
                Value cond = eval_const_expr(node->as.if_stmt.condition);
                if (cond.as.boolean) {
                    // Only keep then branch
                    return eliminate_dead_code(node->as.if_stmt.then_block, ctx);
                } else if (node->as.if_stmt.else_block) {
                    // Only keep else branch
                    return eliminate_dead_code(node->as.if_stmt.else_block, ctx);
                } else {
                    // Entire if statement is dead
                    return NULL;
                }
            }
            
            // Process both branches
            node->as.if_stmt.then_block = eliminate_dead_code(
                node->as.if_stmt.then_block, ctx
            );
            if (node->as.if_stmt.else_block) {
                node->as.if_stmt.else_block = eliminate_dead_code(
                    node->as.if_stmt.else_block, ctx
                );
            }
            return node;
        }
        
        case NODE_VAR_DECL: {
            // Check if variable is ever used
            int var_id = get_var_id(node->as.var_decl.name);
            if (!ctx->var_used[var_id] && !has_side_effects(node->as.var_decl.value)) {
                // Variable unused and initializer has no side effects
                return NULL;
            }
            return node;
        }
        
        default:
            return node;
    }
}
```

### Benefits

- **Smaller binaries**: Less code to compile and link
- **Faster execution**: Fewer instructions to execute
- **Better cache usage**: More useful code in cache

### Challenges

- **Side effects**: Must preserve code with side effects
- **Debugging**: May make debugging harder if too aggressive
- **Inter-procedural**: Need whole-program analysis for best results

## Optimization #3: Tail Call Optimization

### Description

Convert tail-recursive calls into loops to avoid stack overflow and improve performance.

### Example

**Before:**
```nano
fn factorial(n: int, acc: int) -> int {
    if (<= n 1) {
        return acc
    }
    return (factorial (- n 1) (* n acc))  // Tail call
}
```

**After (conceptual C):**
```c
int64_t factorial(int64_t n, int64_t acc) {
tail_call:
    if (n <= 1) {
        return acc;
    }
    int64_t tmp_n = n - 1;
    int64_t tmp_acc = n * acc;
    n = tmp_n;
    acc = tmp_acc;
    goto tail_call;
}
```

### Implementation

```c
bool is_tail_call(ASTNode* func_body, const char* func_name) {
    // Check if last expression is a call to same function
    if (func_body->type != NODE_BLOCK) return false;
    
    ASTNode* last = func_body->as.block.stmts[func_body->as.block.count - 1];
    
    if (last->type == NODE_RETURN && 
        last->as.return_stmt.value->type == NODE_CALL &&
        strcmp(last->as.return_stmt.value->as.call.name, func_name) == 0) {
        return true;
    }
    
    return false;
}

ASTNode* optimize_tail_call(Function* func) {
    if (!is_tail_call(func->body, func->name)) {
        return func->body;
    }
    
    // Transform to loop
    // 1. Add label at function start
    // 2. Replace tail call with parameter updates + goto
    
    ASTNode* loop = create_loop_node();
    // ... transformation logic ...
    
    return loop;
}
```

### Benefits

- **No stack overflow**: Tail-recursive functions use constant stack
- **Better performance**: Loops are faster than function calls
- **Memory efficiency**: No frame allocations

### Challenges

- **Detection**: Must identify true tail calls
- **Mutual recursion**: Hard to optimize A→B→A cycles
- **Parameter handling**: Must update parameters correctly

## Optimization #4: Common Subexpression Elimination

### Description

Identify and eliminate redundant computations.

### Example

**Before:**
```nano
let a: int = (+ (* x y) z)
let b: int = (- (* x y) w)  // (* x y) computed twice
```

**After:**
```nano
let temp: int = (* x y)
let a: int = (+ temp z)
let b: int = (- temp w)
```

### Implementation

```c
typedef struct {
    ASTNode* expr;
    const char* temp_var;
} CSEEntry;

typedef struct {
    CSEEntry* entries;
    int count;
} CSETable;

ASTNode* cse_optimize(ASTNode* node, CSETable* table) {
    if (!is_pure_expr(node)) return node;
    
    // Check if expression already computed
    for (int i = 0; i < table->count; i++) {
        if (ast_equals(table->entries[i].expr, node)) {
            // Replace with temp variable reference
            return create_var_ref(table->entries[i].temp_var);
        }
    }
    
    // Add to table if expensive
    if (is_expensive_expr(node)) {
        const char* temp = generate_temp_var();
        add_cse_entry(table, node, temp);
    }
    
    return node;
}
```

## Optimization #5: Loop Optimizations

### Loop Invariant Code Motion

Move computations that don't change in loop outside the loop.

**Before:**
```nano
while (< i n) {
    let x: int = (* factor 2)  // Doesn't depend on i
    (process i x)
    set i (+ i 1)
}
```

**After:**
```nano
let x: int = (* factor 2)
while (< i n) {
    (process i x)
    set i (+ i 1)
}
```

### Loop Unrolling

Reduce loop overhead by duplicating loop body.

**Before:**
```nano
while (< i n) {
    (process i)
    set i (+ i 1)
}
```

**After:**
```nano
while (< i (- n 3)) {
    (process i)
    (process (+ i 1))
    (process (+ i 2))
    (process (+ i 3))
    set i (+ i 4)
}
// Handle remaining iterations
```

## Optimization Flags

```bash
nanoc --optimize=0       # No optimizations (debug)
nanoc --optimize=1       # Basic (constant folding)
nanoc --optimize=2       # Standard (+ DCE, CSE)
nanoc --optimize=3       # Aggressive (+ TCO, loop opts)
nanoc --optimize=size    # Minimize binary size
nanoc --optimize=speed   # Maximize runtime speed
```

## Testing Strategy

### Correctness Tests

```c
// For each optimization, verify:
// 1. Output is semantically equivalent
// 2. Shadow tests still pass
// 3. Edge cases handled
```

### Performance Tests

```bash
# Benchmark before/after optimization
./scripts/benchmark.sh --opt-level 0
./scripts/benchmark.sh --opt-level 3

# Compare runtime performance
diff benchmark_opt0.json benchmark_opt3.json
```

### Regression Tests

```bash
# Ensure optimizations don't break existing code
make test OPTIMIZE=3
```

## Implementation Roadmap

### Phase 1: Infrastructure (2 weeks)
- Optimization pass framework
- AST visitor pattern
- Optimization flags
- Testing infrastructure

### Phase 2: Constant Folding (1 week)
- Expression evaluation
- Const propagation
- Integration with type checker

### Phase 3: Dead Code Elimination (1 week)
- Reachability analysis
- Unused variable detection
- Branch elimination

### Phase 4: Tail Call Optimization (2 weeks)
- Tail call detection
- Loop transformation
- Mutual recursion handling

### Phase 5: Advanced Optimizations (4 weeks)
- Common subexpression elimination
- Loop invariant code motion
- Inline expansion
- Loop unrolling

## Future Optimizations

1. **Strength Reduction**: Replace expensive ops with cheaper ones
2. **Inlining**: Expand small functions at call sites
3. **Vectorization**: Use SIMD instructions
4. **Escape Analysis**: Stack-allocate non-escaping objects
5. **Profile-Guided**: Use runtime profiling data

## References

- [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- [LLVM Optimization Passes](https://llvm.org/docs/Passes.html)
- [Compilers: Principles, Techniques, and Tools (Dragon Book)](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools)
- [Engineering a Compiler](https://www.elsevier.com/books/engineering-a-compiler/cooper/978-0-12-088478-0)

## Related Issues

- `nanolang-dew`: Constant folding implementation
- `nanolang-d1w`: Tail call optimization implementation
- `nanolang-dlx`: Dead code elimination implementation
- Performance benchmarking (nanolang-qpo)
- Profiling infrastructure

