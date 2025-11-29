# Phase 2 PIVOT: Parser Generator & AST Modules

## New Direction: Following Python's Path

Similar to Python's `ast` module and parser ecosystem, nanolang will provide:
1. **LALR Parser Generator** - Build custom parsers
2. **AST Module** - Programmatic AST manipulation
3. **Meta-programming Foundation** - Enable DSLs and language tools

This positions nanolang as a **batteries-included systems language** for compiler construction.

---

## Module 1: AST Support Module (~800 lines)

### Purpose
Programmatic AST creation, traversal, and transformation - like Python's `ast` module.

### Core Types
```nanolang
/* AST Node Types */
enum ASTNodeType {
    Program,
    Function,
    Struct,
    Enum,
    Union,
    Let,
    If,
    While,
    Match,
    Call,
    Literal,
    BinaryOp,
    UnaryOp,
    FieldAccess,
    ArrayAccess
}

/* Generic AST Node */
struct ASTNode {
    type: ASTNodeType,
    line: int,
    column: int,
    children: array<ASTNode>,
    value: string,           /* For literals, identifiers */
    metadata: array<string>  /* Type info, etc */
}
```

### Key Functions
```nanolang
/* Creation */
fn ast_create(type: ASTNodeType, value: string) -> ASTNode
fn ast_add_child(node: ASTNode, child: ASTNode) -> ASTNode

/* Traversal */
fn ast_walk(node: ASTNode, visitor: fn(ASTNode) -> void) -> void
fn ast_find(node: ASTNode, predicate: fn(ASTNode) -> bool) -> array<ASTNode>

/* Transformation */
fn ast_replace(node: ASTNode, old: ASTNode, new: ASTNode) -> ASTNode
fn ast_transform(node: ASTNode, fn(ASTNode) -> ASTNode) -> ASTNode

/* Validation */
fn ast_validate(node: ASTNode) -> Result<void, string>

/* Pretty Printing */
fn ast_to_string(node: ASTNode, indent: int) -> string
fn ast_dump(node: ASTNode) -> void

/* S-Expression format (compatible with nanolang) */
fn ast_to_sexpr(node: ASTNode) -> string
fn sexpr_to_ast(s: string) -> Result<ASTNode, string>
```

### Use Cases
```nanolang
/* Example: Find all function calls */
let calls: array<ASTNode> = (ast_find root fn(n: ASTNode) -> bool {
    return (== n.type ASTNodeType.Call)
})

/* Example: Replace all occurrences */
let new_ast: ASTNode = (ast_transform root fn(n: ASTNode) -> ASTNode {
    if (== n.type ASTNodeType.Call) {
        if (== (str_equals n.value "old_func") true) {
            return (ast_create ASTNodeType.Call "new_func")
        } else {
            return n
        }
    } else {
        return n
    }
})

/* Example: Generate AST programmatically */
let main_func: ASTNode = (ast_create ASTNodeType.Function "main")
let print_call: ASTNode = (ast_create ASTNodeType.Call "println")
let str_lit: ASTNode = (ast_create ASTNodeType.Literal "\"Hello!\"")
set print_call (ast_add_child print_call str_lit)
set main_func (ast_add_child main_func print_call)
```

---

## Module 2: LALR Parser Generator (~2,000 lines)

### Purpose
Generate LR/LALR parsers from BNF grammars - like yacc/bison for nanolang.

### Phase 2.1: Grammar Definition (~200 lines)

```nanolang
/* Production rule: A -> B C D */
struct Production {
    id: int,
    lhs: string,                 /* Non-terminal */
    rhs: array<string>,          /* Symbols */
    action: string               /* Code to execute */
}

struct Grammar {
    productions: array<Production>,
    terminals: array<string>,
    nonterminals: array<string>,
    start_symbol: string
}

/* Parse grammar from BNF-like format */
fn parse_grammar(input: string) -> Result<Grammar, string>
```

Example grammar:
```
expr -> expr PLUS term    { return ast_create(BinaryOp, "+") }
expr -> term              { return $1 }
term -> term STAR factor  { return ast_create(BinaryOp, "*") }
term -> factor            { return $1 }
factor -> NUMBER          { return ast_create(Literal, $1) }
factor -> LPAREN expr RPAREN  { return $2 }
```

### Phase 2.2: First/Follow Sets (~200 lines)

```nanolang
/* Compute FIRST sets for all symbols */
fn compute_first(g: Grammar) -> array<array<string>>

/* Compute FOLLOW sets for all non-terminals */
fn compute_follow(g: Grammar, first: array<array<string>>) -> array<array<string>>
```

Classic algorithm:
- FIRST(X) = set of terminals that can start X
- FOLLOW(X) = set of terminals that can follow X
- Handle ε (empty) productions

### Phase 2.3: LR Item Sets (~400 lines)

```nanolang
/* LR(1) item: production with dot and lookahead */
struct LRItem {
    production_id: int,
    dot_position: int,       /* Position of • in production */
    lookahead: string        /* Lookahead terminal */
}

struct ItemSet {
    id: int,
    items: array<LRItem>
}

/* Closure operation */
fn closure(items: array<LRItem>, g: Grammar) -> array<LRItem>

/* Goto operation */
fn goto_set(items: array<LRItem>, symbol: string, g: Grammar) -> array<LRItem>

/* Build canonical collection of LR(1) item sets */
fn build_lr_collection(g: Grammar) -> array<ItemSet>
```

### Phase 2.4: Parse Table Generation (~400 lines)

```nanolang
union Action {
    Shift { state: int },
    Reduce { production_id: int },
    Accept {},
    Error { message: string }
}

struct ParseTable {
    action: array<array<Action>>,   /* action[state][terminal] */
    goto: array<array<int>>         /* goto[state][nonterminal] */
}

/* Build SLR table (simpler, good starting point) */
fn build_slr_table(g: Grammar, collection: array<ItemSet>) -> Result<ParseTable, string>

/* Build LALR table (more powerful, fewer conflicts) */
fn build_lalr_table(g: Grammar, collection: array<ItemSet>) -> Result<ParseTable, string>
```

### Phase 2.5: Conflict Detection & Resolution (~200 lines)

```nanolang
enum ConflictType {
    ShiftReduce,
    ReduceReduce
}

struct Conflict {
    type: ConflictType,
    state: int,
    symbol: string,
    actions: array<Action>
}

/* Detect all conflicts in table */
fn detect_conflicts(table: ParseTable) -> array<Conflict>

/* Resolve conflicts using precedence/associativity */
fn resolve_conflicts(conflicts: array<Conflict>, prec: array<Precedence>) -> ParseTable
```

### Phase 2.6: Parser Runtime (~300 lines)

```nanolang
/* Parse token stream using table */
fn parse(table: ParseTable, tokens: array<Token>, actions: array<string>) -> Result<ASTNode, string> {
    let mut stack: array<int> = (array_new 1 0)
    let mut ast_stack: array<ASTNode> = (array_new 0 (ast_create ASTNodeType.Program ""))
    
    /* Standard LR parsing algorithm */
    while true {
        let state: int = (at stack (- (array_length stack) 1))
        let token: Token = (get_current_token tokens)
        let action: Action = (get_action table state token.type)
        
        match action {
            Shift(next_state) => {
                /* Push state and token */
                (array_push stack next_state)
                (array_push ast_stack (token_to_ast token))
                (advance tokens)
            }
            Reduce(prod_id) => {
                /* Pop RHS, execute action, push result */
                let prod: Production = (get_production g prod_id)
                let children: array<ASTNode> = (pop_n ast_stack (array_length prod.rhs))
                let result: ASTNode = (execute_action prod.action children)
                (array_push ast_stack result)
                
                /* Pop states and goto */
                (pop_n stack (array_length prod.rhs))
                let prev_state: int = (at stack (- (array_length stack) 1))
                let next_state: int = (goto_state table prev_state prod.lhs)
                (array_push stack next_state)
            }
            Accept() => {
                return Ok((at ast_stack 0))
            }
            Error(msg) => {
                return Err(msg)
            }
        }
    }
}
```

### Phase 2.7: Code Generation (~300 lines)

```nanolang
/* Generate nanolang parser from grammar */
fn generate_parser(g: Grammar, table: ParseTable, output_file: string) -> Result<void, string> {
    /* Generate code like:
    fn parse_expr(tokens: array<Token>) -> Result<ASTNode, string> {
        // Table-driven parser code
        ...
    }
    */
}
```

---

## Module 3: Integration & Dogfooding

### Phase 2.8: Nanolang Grammar Definition (~200 lines)

Define nanolang's grammar in the new parser generator format:
```
program -> items
items -> items item
items -> item
item -> function_def
item -> struct_def
item -> enum_def
...
```

### Phase 2.9: Generate Nanolang Parser

**The Big Test:**
```bash
# Use LALR module to generate parser for nanolang itself
./bin/nanoc stdlib/lalr.nano stdlib/ast.nano nanolang.grammar -o gen_parser.nano

# Use generated parser to parse nanolang code
./bin/nanoc gen_parser.nano test.nano -o parsed_ast

# Compare with C parser output
diff c_parser_ast generated_parser_ast
```

---

## Comparison with Python

| Python | Nanolang |
|--------|----------|
| `ast` module | `stdlib/ast.nano` |
| Third-party parsers (PLY, ANTLR) | `stdlib/lalr.nano` (built-in!) |
| AST manipulation | First-class support |
| Parser generators | Native LALR generator |

---

## Timeline & Effort

| Phase | Lines | Hours | Priority |
|-------|-------|-------|----------|
| 2.1: AST Module | ~800 | 12-15 | **HIGH** |
| 2.2: Grammar Definition | ~200 | 3-4 | **HIGH** |
| 2.3: First/Follow | ~200 | 4-5 | **HIGH** |
| 2.4: LR Items | ~400 | 8-10 | **HIGH** |
| 2.5: Table Generation | ~400 | 10-12 | **HIGH** |
| 2.6: Conflict Resolution | ~200 | 5-6 | MEDIUM |
| 2.7: Parser Runtime | ~300 | 6-8 | **HIGH** |
| 2.8: Code Generation | ~300 | 6-8 | MEDIUM |
| 2.9: Nanolang Grammar | ~200 | 4-5 | **HIGH** |
| **TOTAL** | **~3,000** | **58-73 hours** | |

---

## Why This Is Awesome

1. **Practical Tool**: Real parser generator people can use
2. **Self-Hosting**: Generate nanolang's own parser
3. **Educational**: Complete LALR implementation
4. **Meta-Programming**: Foundation for DSLs, macros, code gen
5. **Competitive**: Python/Ruby level tooling
6. **Impressive**: Shows language maturity

---

## Success Metrics

✅ AST module can represent full nanolang syntax  
✅ LALR generator builds working parsers  
✅ Generated parser for arithmetic expressions works  
✅ Generated parser for nanolang itself works  
✅ All modules pass comprehensive test suites  
✅ Documentation & examples for both modules  
✅ **Dogfood moment**: nanolang parser generated by nanolang!

---

## Next Steps

1. Start with **AST Module** (foundation for everything)
2. Build **Grammar Definition** (simple data structures)
3. Implement **First/Follow** (classic algorithm)
4. Build **LR Item Construction** (core LALR logic)
5. Generate **Parse Tables**
6. Add **Conflict Resolution**
7. Implement **Parser Runtime**
8. Add **Code Generation**
9. **Dogfood**: Generate nanolang parser!

Ready to begin?
