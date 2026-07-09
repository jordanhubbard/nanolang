# First-Class Functions Design Document

**Date:** November 15, 2025  
**Status:** Design Complete - Implementation Starting  
**Priority:** HIGH (Required before Phase 2 self-hosting)

---

## 🎯 Vision

**Enable functional programming patterns WITHOUT user-visible pointers.**

Users write clean, type-safe code with functions as values. The transpiler handles all pointer mechanics behind the scenes in the generated C code.

---

## 🔑 Key Innovation

**The user never sees `*` or pointer syntax, but gets full higher-order function capabilities!**

```nano
/* User writes: */
fn filter(numbers: array<int>, test: fn(int) -> bool) -> array<int> {
    /* test is just called like any function */
    if (test value) { /* ... */ } else {}
}

/* Transpiler generates: */
typedef bool (*IntPredicate)(int64_t);
int64_t* filter(int64_t* numbers, IntPredicate test) {
    /* test is a function pointer, but user never knew! */
    if (test(value)) { /* ... */ }
}
```

---

## 📋 Implementation Phases

### Phase B1: Functions as Parameters (CRITICAL)
**Time:** 10-12 hours  
**Benefit:** Enables map, filter, fold, callback patterns

### Phase B2: Functions as Return Values
**Time:** 5-8 hours  
**Benefit:** Function factories, strategy pattern

### Phase B3: Function Variables
**Time:** 5-10 hours  
**Benefit:** Dispatch tables, cleaner code organization

### Phase B4: Documentation
**Time:** 3-5 hours  
**Deliverables:** User guide, spec updates, examples

### Phase B5: Code Audit
**Time:** 5-8 hours  
**Goal:** Refactor src_nano/* to use new patterns

---

## 🎨 Syntax Design

### Function Type Syntax

```nano
/* General form */
fn(param_type1, param_type2, ...) -> return_type

/* Examples */
fn(int) -> bool                    /* Predicate */
fn(int, int) -> int                /* Binary operation */
fn(string) -> void                 /* Callback */
fn(Player, int) -> Player          /* User-defined types */
```

### As Parameter Types

```nano
fn filter(items: array<int>, test: fn(int) -> bool) -> array<int> {
    let mut result: array<int> = []
    let mut i: int = 0
    while (< i (array_length items)) {
        let item: int = (at items i)
        if (test item) {  /* Just call like normal function! */
            set result (array_push result item)
        } else {}
        set i (+ i 1)
    }
    return result
}

/* Usage */
fn is_positive(x: int) -> bool {
    return (> x 0)
}

let positives: array<int> = (filter numbers is_positive)
```

### As Return Types

```nano
fn get_comparator(ascending: bool) -> fn(int, int) -> bool {
    if ascending {
        return less_than
    } else {
        return greater_than
    }
}

fn less_than(a: int, b: int) -> bool { return (< a b) }
fn greater_than(a: int, b: int) -> bool { return (> a b) }

let compare: fn(int, int) -> bool = (get_comparator true)
let result: bool = (compare 5 10)  /* Calls less_than */
```

### As Variables

```nano
let my_test: fn(int) -> bool = is_positive
let my_op: fn(int, int) -> int = add

let result1: array<int> = (filter numbers my_test)
let result2: int = (my_op 5 3)
```

---

## 🏗️ Implementation Architecture

### 1. Lexer Changes

**Status:** Minimal changes needed

The lexer already handles:
- `fn` keyword (TOKEN_FN)
- `->` arrow (TOKEN_ARROW)
- `(` `)` for parameters (TOKEN_LPAREN, TOKEN_RPAREN)

**No new tokens required!** The existing tokens compose to form function types.

---

### 2. Parser Changes

**New Structures:**

```c
/* In src/nanolang.h */

/* Function type signature for parameters/returns */
typedef struct {
    Type *param_types;           /* Array of parameter types */
    int param_count;             /* Number of parameters */
    char **param_struct_names;   /* For struct/enum types */
    Type return_type;            /* Return type */
    char *return_struct_name;    /* For struct/enum return */
} FunctionSignature;

/* Extended Parameter struct */
typedef struct {
    char *name;
    Type type;
    char *struct_type_name;      /* For struct/enum/union */
    FunctionSignature *fn_sig;   /* For function types - NEW! */
} Parameter;

/* Extended Function struct */
typedef struct Function {
    char *name;
    Parameter *params;
    int param_count;
    Type return_type;
    char *return_struct_type_name;
    FunctionSignature *return_fn_sig;  /* For function return - NEW! */
    ASTNode *body;
    ASTNode *shadow_test;
    bool is_extern;
} Function;
```

**New Parsing Function:**

```c
/* Parse function type: fn(int, string) -> bool */
FunctionSignature* parse_function_type(Parser *p) {
    /* expect 'fn' */
    if (current_token(p)->type != TOKEN_FN) return NULL;
    advance(p);
    
    /* expect '(' */
    if (current_token(p)->type != TOKEN_LPAREN) {
        parse_error(p, "Expected '(' after 'fn'");
        return NULL;
    }
    advance(p);
    
    /* Parse parameter types */
    FunctionSignature *sig = malloc(sizeof(FunctionSignature));
    sig->param_types = NULL;
    sig->param_count = 0;
    sig->param_struct_names = NULL;
    
    if (current_token(p)->type != TOKEN_RPAREN) {
        /* Parse comma-separated types */
        while (1) {
            char *struct_name = NULL;
            Type param_type = parse_type_with_element(p, NULL, &struct_name);
            
            /* Grow arrays */
            sig->param_count++;
            sig->param_types = realloc(sig->param_types, 
                                      sizeof(Type) * sig->param_count);
            sig->param_struct_names = realloc(sig->param_struct_names,
                                             sizeof(char*) * sig->param_count);
            
            sig->param_types[sig->param_count - 1] = param_type;
            sig->param_struct_names[sig->param_count - 1] = struct_name;
            
            if (current_token(p)->type == TOKEN_COMMA) {
                advance(p);
            } else {
                break;
            }
        }
    }
    
    /* expect ')' */
    if (current_token(p)->type != TOKEN_RPAREN) {
        parse_error(p, "Expected ')' in function type");
        free_function_signature(sig);
        return NULL;
    }
    advance(p);
    
    /* expect '->' */
    if (current_token(p)->type != TOKEN_ARROW) {
        parse_error(p, "Expected '->' in function type");
        free_function_signature(sig);
        return NULL;
    }
    advance(p);
    
    /* Parse return type */
    char *return_struct_name = NULL;
    sig->return_type = parse_type_with_element(p, NULL, &return_struct_name);
    sig->return_struct_name = return_struct_name;
    
    return sig;
}
```

**Modified Type Parsing:**

```c
/* In parse_type_with_element - add check for function types */
static Type parse_type_with_element(Parser *p, Type *element_type_out, 
                                    char **type_param_name_out) {
    Token *tok = current_token(p);
    Type type = TYPE_UNKNOWN;
    
    /* Check for function type: fn(...) -> ... */
    if (tok->type == TOKEN_FN) {
        FunctionSignature *sig = parse_function_type(p);
        if (sig) {
            /* Store signature in type_param_name_out for now */
            /* Better: add FunctionSignature** to parse_type_with_element */
            return TYPE_FUNCTION;
        }
        return TYPE_UNKNOWN;
    }
    
    /* ... existing type parsing ... */
}
```

---

### 3. Type System Changes

**New Type:**

```c
/* In src/nanolang.h */
typedef enum {
    TYPE_UNKNOWN = 0,
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_STRING,
    TYPE_BOOL,
    TYPE_VOID,
    /* ... existing types ... */
    TYPE_FUNCTION,     /* NEW! */
} Type;
```

**Type Compatibility:**

```c
/* In src/typechecker.c */

bool function_signatures_compatible(FunctionSignature *sig1, 
                                   FunctionSignature *sig2) {
    /* Check parameter count */
    if (sig1->param_count != sig2->param_count) return false;
    
    /* Check each parameter type */
    for (int i = 0; i < sig1->param_count; i++) {
        if (sig1->param_types[i] != sig2->param_types[i]) return false;
        
        /* For struct/enum parameters, check names match */
        if (sig1->param_types[i] == TYPE_STRUCT || 
            sig1->param_types[i] == TYPE_ENUM) {
            if (strcmp(sig1->param_struct_names[i], 
                      sig2->param_struct_names[i]) != 0) {
                return false;
            }
        }
    }
    
    /* Check return type */
    if (sig1->return_type != sig2->return_type) return false;
    
    /* For struct/enum returns, check names match */
    if (sig1->return_type == TYPE_STRUCT || 
        sig1->return_type == TYPE_ENUM) {
        if (strcmp(sig1->return_struct_name, 
                  sig2->return_struct_name) != 0) {
            return false;
        }
    }
    
    return true;
}

/* When checking function call with function parameter */
Type check_call_with_function_param(ASTNode *call_expr, Environment *env) {
    Function *func = env_get_function(env, call_expr->as.call.name);
    
    for (int i = 0; i < func->param_count; i++) {
        if (func->params[i].type == TYPE_FUNCTION) {
            /* Get argument - should be function name */
            ASTNode *arg = call_expr->as.call.args[i];
            
            if (arg->type == AST_IDENTIFIER) {
                /* Look up function by name */
                Function *passed_func = env_get_function(env, arg->as.identifier);
                
                /* Check signature compatibility */
                if (!function_signatures_compatible(
                        func->params[i].fn_sig,
                        create_signature_from_function(passed_func))) {
                    fprintf(stderr, "Error: Function signature mismatch\n");
                    return TYPE_UNKNOWN;
                }
            } else {
                fprintf(stderr, "Error: Expected function name\n");
                return TYPE_UNKNOWN;
            }
        }
    }
    
    return func->return_type;
}
```

---

### 4. Transpiler Changes

**Function Pointer Typedefs:**

```c
/* In src/transpiler.c */

/* Generate typedef for function signature */
void generate_function_typedef(StringBuilder *sb, FunctionSignature *sig,
                               const char *typedef_name) {
    sb_append(sb, "typedef ");
    
    /* Return type */
    if (sig->return_type == TYPE_STRUCT) {
        sb_appendf(sb, "struct %s ", sig->return_struct_name);
    } else {
        sb_appendf(sb, "%s ", type_to_c(sig->return_type));
    }
    
    /* Function pointer syntax: (*typedef_name) */
    sb_appendf(sb, "(*%s)", typedef_name);
    
    /* Parameters */
    sb_append(sb, "(");
    for (int i = 0; i < sig->param_count; i++) {
        if (i > 0) sb_append(sb, ", ");
        
        if (sig->param_types[i] == TYPE_STRUCT) {
            sb_appendf(sb, "struct %s", sig->param_struct_names[i]);
        } else {
            sb_append(sb, type_to_c(sig->param_types[i]));
        }
    }
    sb_append(sb, ");\n");
}

/* Generate unique typedef name for signature */
char* get_function_typedef_name(FunctionSignature *sig) {
    static int typedef_counter = 0;
    char *name = malloc(64);
    
    /* Generate descriptive name: IntPredicate, IntBinaryOp, etc. */
    if (sig->param_count == 1 && sig->return_type == TYPE_BOOL) {
        /* Predicate: fn(T) -> bool */
        snprintf(name, 64, "Predicate_%d", typedef_counter++);
    } else if (sig->param_count == 2 && 
               sig->param_types[0] == sig->param_types[1] &&
               sig->return_type == sig->param_types[0]) {
        /* Binary op: fn(T, T) -> T */
        snprintf(name, 64, "BinaryOp_%d", typedef_counter++);
    } else {
        /* Generic: FnType_N */
        snprintf(name, 64, "FnType_%d", typedef_counter++);
    }
    
    return name;
}

/* Track and generate typedefs */
typedef struct {
    FunctionSignature **signatures;
    char **typedef_names;
    int count;
    int capacity;
} FunctionTypeRegistry;

FunctionTypeRegistry* create_fn_type_registry() {
    FunctionTypeRegistry *reg = malloc(sizeof(FunctionTypeRegistry));
    reg->signatures = malloc(sizeof(FunctionSignature*) * 16);
    reg->typedef_names = malloc(sizeof(char*) * 16);
    reg->count = 0;
    reg->capacity = 16;
    return reg;
}

/* Get or create typedef for signature */
const char* register_function_signature(FunctionTypeRegistry *reg,
                                       FunctionSignature *sig) {
    /* Check if already registered */
    for (int i = 0; i < reg->count; i++) {
        if (function_signatures_compatible(reg->signatures[i], sig)) {
            return reg->typedef_names[i];
        }
    }
    
    /* Register new signature */
    if (reg->count >= reg->capacity) {
        reg->capacity *= 2;
        reg->signatures = realloc(reg->signatures,
                                 sizeof(FunctionSignature*) * reg->capacity);
        reg->typedef_names = realloc(reg->typedef_names,
                                    sizeof(char*) * reg->capacity);
    }
    
    reg->signatures[reg->count] = sig;
    reg->typedef_names[reg->count] = get_function_typedef_name(sig);
    reg->count++;
    
    return reg->typedef_names[reg->count - 1];
}
```

**Generating Typedefs in Output:**

```c
/* In transpile_program - add before function forward declarations */

/* Generate function type typedefs */
FunctionTypeRegistry *fn_registry = create_fn_type_registry();

/* First pass: collect all function signatures */
for (int i = 0; i < program->as.program.count; i++) {
    ASTNode *item = program->as.program.items[i];
    
    if (item->type == AST_FUNCTION) {
        Function *func = &item->as.function;
        
        /* Check parameters for function types */
        for (int j = 0; j < func->param_count; j++) {
            if (func->params[j].type == TYPE_FUNCTION) {
                register_function_signature(fn_registry, 
                                          func->params[j].fn_sig);
            }
        }
        
        /* Check return type for function type */
        if (func->return_type == TYPE_FUNCTION) {
            register_function_signature(fn_registry, func->return_fn_sig);
        }
    }
}

/* Generate typedef declarations */
sb_append(&sb, "/* Function Type Typedefs */\n");
for (int i = 0; i < fn_registry->count; i++) {
    generate_function_typedef(&sb, fn_registry->signatures[i],
                            fn_registry->typedef_names[i]);
}
sb_append(&sb, "\n");
```

**Function Call Transpilation:**

```c
/* When transpiling function call with function arguments */
case AST_CALL: {
    Function *func = env_get_function(env, expr->as.call.name);
    
    sb_append(sb, expr->as.call.name);
    sb_append(sb, "(");
    
    for (int i = 0; i < expr->as.call.arg_count; i++) {
        if (i > 0) sb_append(sb, ", ");
        
        /* Check if this parameter expects a function */
        if (func && i < func->param_count && 
            func->params[i].type == TYPE_FUNCTION) {
            
            /* Argument should be identifier (function name) */
            if (expr->as.call.args[i]->type == AST_IDENTIFIER) {
                /* Just emit function name - C will take its address */
                sb_append(sb, expr->as.call.args[i]->as.identifier);
            } else {
                /* Should not happen - type checker catches this */
                transpile_expression(sb, expr->as.call.args[i], env);
            }
        } else {
            /* Regular argument */
            transpile_expression(sb, expr->as.call.args[i], env);
        }
    }
    
    sb_append(sb, ")");
    break;
}
```

---

### 5. Interpreter Changes

**Value Type:**

```c
/* In src/nanolang.h */
typedef enum {
    VAL_VOID,
    VAL_INT,
    VAL_FLOAT,
    VAL_STRING,
    VAL_BOOL,
    /* ... existing types ... */
    VAL_FUNCTION,  /* NEW! */
} ValueType;

/* Value union */
typedef struct Value {
    ValueType type;
    bool is_return;
    union {
        int64_t int_val;
        double float_val;
        char *string_val;
        bool bool_val;
        /* ... existing fields ... */
        struct {
            char *function_name;
            FunctionSignature *signature;
        } function_val;  /* NEW! */
    };
} Value;
```

**Function Call with Function Parameter:**

```c
/* In src/eval.c */

Value eval_call_with_function_param(ASTNode *call_expr, Environment *env) {
    Function *func = env_get_function(env, call_expr->as.call.name);
    
    /* Build argument values */
    Value *arg_values = malloc(sizeof(Value) * call_expr->as.call.arg_count);
    
    for (int i = 0; i < call_expr->as.call.arg_count; i++) {
        if (func->params[i].type == TYPE_FUNCTION) {
            /* Store function reference */
            ASTNode *arg = call_expr->as.call.args[i];
            if (arg->type == AST_IDENTIFIER) {
                arg_values[i].type = VAL_FUNCTION;
                arg_values[i].function_val.function_name = 
                    strdup(arg->as.identifier);
                arg_values[i].function_val.signature = func->params[i].fn_sig;
            }
        } else {
            /* Regular argument */
            arg_values[i] = eval_expression(call_expr->as.call.args[i], env);
        }
    }
    
    /* ... rest of function call evaluation ... */
}

/* When calling a function parameter inside the function body */
Value eval_function_parameter_call(const char *fn_param_name,
                                  Value *args, int arg_count,
                                  Environment *env) {
    /* Look up the actual function that was passed */
    Value fn_value = env_get_var(env, fn_param_name);
    
    if (fn_value.type != VAL_FUNCTION) {
        fprintf(stderr, "Error: '%s' is not a function\n", fn_param_name);
        exit(1);
    }
    
    /* Call the actual function */
    Function *actual_func = env_get_function(env, 
                                            fn_value.function_val.function_name);
    
    /* Create new environment and bind parameters */
    Environment *call_env = create_environment();
    call_env->parent = env;
    
    for (int i = 0; i < arg_count; i++) {
        env_define_var(call_env, actual_func->params[i].name, args[i]);
    }
    
    /* Execute function body */
    return eval_block(actual_func->body, call_env);
}
```

---

## 📝 Testing Strategy

### Unit Tests

```nano
/* tests/unit/functions/01_function_parameter.nano */

fn apply_twice(x: int, f: fn(int) -> int) -> int {
    let result1: int = (f x)
    let result2: int = (f result1)
    return result2
}

fn double(x: int) -> int {
    return (* x 2)
}

fn increment(x: int) -> int {
    return (+ x 1)
}

shadow apply_twice {
    assert (== (apply_twice 5 double) 20)     /* 5 * 2 * 2 */
    assert (== (apply_twice 5 increment) 7)   /* 5 + 1 + 1 */
}

fn main() -> int {
    let result: int = (apply_twice 3 double)
    if (!= result 12) {
        return 1
    } else {}
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### Integration Tests

```nano
/* examples/31_higher_order_functions.nano */

/* Filter: keep elements that pass test */
fn filter(numbers: array<int>, test: fn(int) -> bool) -> array<int> {
    let mut result: array<int> = []
    let mut i: int = 0
    while (< i (array_length numbers)) {
        let value: int = (at numbers i)
        if (test value) {
            set result (array_push result value)
        } else {}
        set i (+ i 1)
    }
    return result
}

/* Map: transform each element */
fn map(numbers: array<int>, transform: fn(int) -> int) -> array<int> {
    let mut result: array<int> = []
    let mut i: int = 0
    while (< i (array_length numbers)) {
        let value: int = (at numbers i)
        let transformed: int = (transform value)
        set result (array_push result transformed)
        set i (+ i 1)
    }
    return result
}

/* Fold: combine elements */
fn fold(numbers: array<int>, initial: int, combine: fn(int, int) -> int) -> int {
    let mut acc: int = initial
    let mut i: int = 0
    while (< i (array_length numbers)) {
        set acc (combine acc (at numbers i))
        set i (+ i 1)
    }
    return acc
}

/* Predicates */
fn is_positive(x: int) -> bool { return (> x 0) }
fn is_even(x: int) -> bool { return (== (% x 2) 0) }

/* Transforms */
fn double(x: int) -> int { return (* x 2) }
fn square(x: int) -> int { return (* x x) }

/* Combiners */
fn add(a: int, b: int) -> int { return (+ a b) }
fn multiply(a: int, b: int) -> int { return (* a b) }

fn main() -> int {
    let numbers: array<int> = [1, -2, 3, -4, 5]
    
    /* Filter */
    let positives: array<int> = (filter numbers is_positive)
    (println "Positives: [1, 3, 5]")
    
    /* Map */
    let doubled: array<int> = (map positives double)
    (println "Doubled: [2, 6, 10]")
    
    /* Fold */
    let sum: int = (fold positives 0 add)
    (print "Sum: ")
    (println sum)  /* Should be 9 */
    
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

---

## 🎯 Success Criteria

### Phase B1 Complete When:

✅ Parser recognizes `fn(type) -> type` syntax  
✅ Type checker validates function signatures  
✅ Transpiler generates correct C function pointers  
✅ Filter example compiles and runs  
✅ Map example compiles and runs  
✅ Fold example compiles and runs  
✅ All shadow tests pass  
✅ Generated C code is clean and idiomatic  

---

## 📊 Timeline

**Phase B1 (Functions as Parameters):** 10-12 hours
- Lexer/Parser: 2 hours
- Type System: 3 hours
- Type Checker: 2 hours
- Transpiler: 2 hours
- Interpreter: 1 hour
- Testing: 2 hours

**Phase B2 (Return Values):** 5-8 hours  
**Phase B3 (Variables):** 5-10 hours  
**Phase B4 (Documentation):** 3-5 hours  
**Phase B5 (Code Audit):** 5-8 hours

**Total:** 30-40 hours

---

## 🚀 Impact on Self-Hosted Compiler

After implementation, `src_nano/*` code can use:

```nano
/* Parser dispatch */
let expr_parsers: array<fn(Parser) -> ASTNode> = [
    parse_number,
    parse_string,
    parse_identifier
]

/* Token classification */
fn classify_tokens(tokens: List<Token>, classifier: fn(Token) -> bool) -> List<Token> {
    /* ... */
}

/* AST transformation */
fn transform_ast(ast: ASTNode, transformer: fn(ASTNode) -> ASTNode) -> ASTNode {
    /* ... */
}
```

**Much cleaner, more composable code!**

---

**Status:** Design complete, ready to implement  
**Next:** Start Phase B1 implementation  
**File:** `planning/FIRST_CLASS_FUNCTIONS_DESIGN.md`

