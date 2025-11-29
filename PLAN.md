# True Self-Hosting Plan for nanolang

## Executive Summary

**Goal:** Achieve TRUE self-hosting where the entire nanolang compiler and interpreter are written in pure nanolang (no C FFI for core functionality), capable of Stage 3 compilation (Stage 2 compiling itself).

**Current Status:** ✅ FFI-based pseudo self-hosting (Stage 1 & 2)
**Target Status:** 🎯 Pure nanolang self-hosting with Stage 3 verification

**Key Insight:** nanolang is a high-level language - we should add features that make writing a compiler in nanolang EASIER than writing it in C.

---

## I. Current Reality

### What We Have (November 29, 2024)

```
✅ Stage 0: C compiler (bin/nanoc - 432 KB)
✅ Stage 1: nanolang wrapper (bin/nanoc_stage1 - 436 KB)
   - 237 lines of nanolang
   - Calls C FFI for actual compilation
   - Uses: nl_compiler_tokenize(), nl_compiler_parse(), etc.
✅ Stage 2: Stage 1 compiling itself (bin/nanoc_stage2 - 436 KB)
   - Same 237-line wrapper
   - Produces IDENTICAL output to Stage 1

✅ Verification: Stage 1 output ≡ Stage 2 output
```

**Achievement:** FFI-based self-hosting ✓
**Missing:** Pure nanolang implementation (13,361 lines C → ~8,600 lines nanolang)

### What Needs to be Written

```
Component          C Lines    Complexity    nanolang Estimate   Status
------------------------------------------------------------------------
lexer.c            327        Low           ~300 lines          Partial exists
parser.c           2,581      High          ~1,800 lines        Partial exists
typechecker.c      3,360      Very High     ~2,200 lines        Minimal exists
transpiler.c       3,063      High          ~1,800 lines        Minimal exists
eval.c             3,155      Very High     ~2,500 lines        Doesn't exist
env.c              875        Medium        ~700 lines          Doesn't exist
module.c           ~500       Medium        ~400 lines          Doesn't exist
------------------------------------------------------------------------
TOTAL              13,361                   ~9,700 lines        ~15% complete
```

---

## II. Strategic Approach: Enhance Language First

### The Insight

Writing a compiler in C requires:
- 450+ strcmp calls for string comparison
- 200+ strdup calls for manual memory
- 70+ realloc calls for dynamic arrays
- Manual StringBuilder implementation
- Verbose error handling
- Boilerplate everywhere

**nanolang is higher-level than C** - let's use that advantage!

### Proposed Language Enhancements

See [LANGUAGE_IMPROVEMENTS_FOR_SELFHOST.md](LANGUAGE_IMPROVEMENTS_FOR_SELFHOST.md) for full analysis.

#### Priority 1: Core Syntax Improvements

1. **String == operator** ✨
   ```nanolang
   /* Before */ if (str_equals keyword "fn") { ... }
   /* After  */ if (== keyword "fn") { ... }
   ```
   **Impact:** 450+ occurrences in compiler

2. **String interpolation** ✨
   ```nanolang
   /* Before */ (str_concat "Error at line " (int_to_string line))
   /* After  */ "Error at line ${line}, column ${col}"
   ```
   **Impact:** 200+ error messages, 40% code reduction

3. **Method syntax** ✨
   ```nanolang
   /* Before */ (str_length (str_substring source 0 10))
   /* After  */ source.substring(0, 10).length()
   ```
   **Impact:** Better readability, common throughout

4. **Character literals** ✨
   ```nanolang
   /* Before */ let newline: int = (char_at "\n" 0)
   /* After  */ let newline: int = '\n'
   ```
   **Impact:** Lexer uses heavily

#### Priority 2: Essential Modules

1. **StringBuilder module** ✨
   ```nanolang
   let sb: StringBuilder = StringBuilder.new()
   sb.append("code").append("\n")
   let result: string = sb.to_string()
   ```
   **Impact:** Transpiler has 3,000+ append operations

2. **Result<T, E> and Option<T> types** ✨
   ```nanolang
   fn parse_number(s: string) -> Result<int, string> {
       /* Type-safe error handling */
   }
   ```
   **Impact:** Clean error propagation throughout

3. **List methods (map, filter, find)** ✨
   ```nanolang
   tokens.filter(fn(t: Token) -> bool { 
       return (== t.type TOKEN_LPAREN) 
   })
   ```
   **Impact:** Replace 100+ manual loops

4. **StringUtils module** ✨
   ```nanolang
   split(path, "/")
   join(parts, ", ")
   trim(input)
   starts_with(line, "#")
   ```
   **Impact:** Common parsing operations

### Expected Benefits

**With enhancements:**
- 13,361 lines C → ~8,600 lines nanolang (35% reduction)
- No manual memory management
- Type-safe error handling
- More readable code
- Faster development

**Without enhancements:**
- 13,361 lines C → ~11,200 lines nanolang (15% reduction)
- Manual string operations
- Verbose error handling
- More bugs

**ROI: Language improvements save 40-70 hours AND benefit entire ecosystem!**

---

## III. Implementation Phases

### Phase 1: Language Enhancements (10-15 hours)

**Goal:** Add core features that make compiler implementation easier

**Tasks:**
1. ✨ String == operator (type checker + transpiler)
   - Modify type checker to allow == for strings
   - Transpiler emits strcmp() == 0
   - Test with examples

2. ✨ Character literals 'x' (lexer + parser)
   - Lexer recognizes 'c' and '\n' syntax
   - Parser creates integer literal
   - Handle escape sequences

3. ✨ Method syntax (parser + type checker)
   - Parse expr.method(args) as (method expr args)
   - Type checker resolves based on expr type
   - Syntactic sugar only

4. ✨ String interpolation (parser + transpiler)
   - Parse "${expr}" in strings
   - Desugar to str_concat chain at compile time
   - Support nested expressions

**Deliverables:**
- Modified lexer.c, parser.c, typechecker.c, transpiler.c
- Test suite for new features
- Updated SPECIFICATION.md
- Examples demonstrating new syntax

**Success Criteria:**
- All existing tests pass
- New syntax compiles correctly
- Generated C code is correct

---

### Phase 2: Standard Library Modules (15-20 hours)

**Goal:** Create reusable modules for compiler implementation

**Tasks:**
1. 📦 **stdlib/StringBuilder.nano** (~400 lines)
   ```nanolang
   struct StringBuilder { ... }
   fn StringBuilder.new() -> StringBuilder
   fn StringBuilder.append(self: mut StringBuilder, s: string) -> void
   fn StringBuilder.to_string(self: StringBuilder) -> string
   ```
   - Implement with mutable state
   - Port from C implementation
   - Optimize for common case (append)
   - Add tests with shadow tests

2. 📦 **stdlib/Result.nano** (~200 lines)
   ```nanolang
   union Result<T, E> {
       Ok { value: T },
       Err { error: E }
   }
   /* Methods for Result and Option */
   ```
   - Define Result and Option unions
   - Add helper methods
   - Examples of usage
   - Pattern matching integration

3. 📦 **stdlib/StringUtils.nano** (~600 lines)
   ```nanolang
   fn split(s: string, delimiter: string) -> List<string>
   fn join(parts: List<string>, sep: string) -> string
   fn trim(s: string) -> string
   /* + 10 more string utilities */
   ```
   - Common string operations
   - Efficient implementations
   - Comprehensive tests

4. 📦 **stdlib/ListUtils.nano** (~400 lines)
   ```nanolang
   fn map<T, U>(list: List<T>, f: fn(T) -> U) -> List<U>
   fn filter<T>(list: List<T>, pred: fn(T) -> bool) -> List<T>
   fn find<T>(list: List<T>, pred: fn(T) -> bool) -> Option<T>
   fn any<T>(list: List<T>, pred: fn(T) -> bool) -> bool
   ```
   - Higher-order list operations
   - Generic implementations
   - Performance benchmarks

5. 📦 **stdlib/HashMap.nano** (~800 lines) [Optional - for later]
   ```nanolang
   struct HashMap<K, V> { ... }
   /* Hash table for symbol tables */
   ```
   - Needed for environment/symbol tables
   - Can defer to Phase 3 if time-constrained

**Deliverables:**
- 4-5 new stdlib modules
- Comprehensive test suites
- Documentation in modules/
- Usage examples

**Success Criteria:**
- All modules compile and test successfully
- Can be used in self-hosted compiler
- Generic modules work with different types

---

### Phase 3: Pure nanolang Compiler (~50-70 hours)

**Goal:** Rewrite entire compiler in pure nanolang

#### 3.1 Lexer (~8-10 hours, ~300 lines)

**Tasks:**
- Read and analyze src/lexer.c (327 lines)
- Design Token and TokenType types
- Implement character-by-character processing
- Keyword recognition (use match!)
- String/number parsing
- Comment handling
- Position tracking (line/column)

**Output:** `compiler/lexer.nano`
```nanolang
enum TokenType { ... }
struct Token { type: TokenType, value: string, line: int, column: int }
fn tokenize(source: string) -> Result<List<Token>, string>
```

**Key simplifications vs C:**
- ✅ No manual memory management (malloc/free)
- ✅ List<Token> instead of dynamic array
- ✅ String interpolation for errors
- ✅ match for keyword lookup

---

#### 3.2 Parser (~15-20 hours, ~1,800 lines)

**Tasks:**
- Read and analyze src/parser.c (2,581 lines)
- Design AST node types (20+ types)
- Recursive descent parser
- Type annotation parsing
- Expression parsing (precedence)
- Statement parsing
- Error recovery
- Position tracking

**Output:** `compiler/parser.nano`
```nanolang
/* AST node types */
enum ASTNodeType { ... }
struct ASTNode { ... }  /* Tagged union of all node types */

/* Parser state */
struct Parser {
    tokens: List<Token>,
    pos: int,
    errors: List<string>
}

fn parse(tokens: List<Token>) -> Result<ASTNode, List<string>>
```

**Key simplifications vs C:**
- ✅ No manual AST node allocation
- ✅ Result type for error handling
- ✅ Pattern matching for token types
- ✅ Method syntax for parser state

---

#### 3.3 Environment / Symbol Table (~8-10 hours, ~700 lines)

**Tasks:**
- Read and analyze src/env.c (875 lines)
- Design Environment structure
- Scope management (stack of scopes)
- Symbol table (functions, variables, types)
- Type environment
- Lookup functions

**Output:** `compiler/env.nano`
```nanolang
struct Environment {
    scopes: List<Scope>,
    functions: HashMap<string, Function>,
    types: HashMap<string, TypeDef>,
    /* ... */
}

fn Environment.new() -> Environment
fn Environment.enter_scope(self: mut Environment) -> void
fn Environment.exit_scope(self: mut Environment) -> void
fn Environment.define_var(self: mut Environment, name: string, type: Type) -> void
fn Environment.lookup_var(self: Environment, name: string) -> Option<Type>
```

**Key simplifications vs C:**
- ✅ HashMap instead of manual hash table
- ✅ Option type for lookups
- ✅ Automatic scope cleanup

---

#### 3.4 Type Checker (~18-25 hours, ~2,200 lines)

**Tasks:**
- Read and analyze src/typechecker.c (3,360 lines)
- Type inference for expressions
- Type checking for statements
- Function signature validation
- Struct/enum/union validation
- Generic type resolution
- Error reporting
- Shadow test execution

**Output:** `compiler/typechecker.nano`
```nanolang
struct TypeChecker {
    env: Environment,
    errors: List<string>,
    /* ... */
}

fn type_check(ast: ASTNode, env: mut Environment) -> Result<void, List<string>>
fn check_expression(expr: ASTNode, env: Environment) -> Result<Type, string>
fn check_statement(stmt: ASTNode, env: mut Environment) -> Result<void, string>
```

**Key simplifications vs C:**
- ✅ Result type for propagating errors
- ✅ String interpolation for error messages
- ✅ Pattern matching on AST nodes
- ✅ No manual type comparison (use == for strings)

---

#### 3.5 Transpiler (~15-20 hours, ~1,800 lines)

**Tasks:**
- Read and analyze src/transpiler.c (3,063 lines)
- C code generation for all AST nodes
- Type to C type mapping
- Name mangling (nl_ prefix)
- Indentation management
- StringBuilder for output
- Runtime function calls
- Header generation

**Output:** `compiler/transpiler.nano`
```nanolang
struct Transpiler {
    output: StringBuilder,
    indent: int,
    temp_counter: int,
    env: Environment,
    /* ... */
}

fn transpile(ast: ASTNode, env: Environment) -> Result<string, string>
fn emit_expression(t: mut Transpiler, expr: ASTNode) -> void
fn emit_statement(t: mut Transpiler, stmt: ASTNode) -> void
```

**Key simplifications vs C:**
- ✅ StringBuilder module
- ✅ String interpolation for code gen
- ✅ Method chaining: sb.append().append()
- ✅ Pattern matching on node types

---

#### 3.6 Interpreter (~20-25 hours, ~2,500 lines)

**Tasks:**
- Read and analyze src/eval.c (3,155 lines)
- Value representation (union of types)
- Expression evaluation
- Statement execution
- Function calls
- Variable binding
- Control flow (if/while/for)
- Error handling

**Output:** `compiler/interpreter.nano`
```nanolang
union Value {
    Int { value: int },
    Float { value: float },
    Bool { value: bool },
    String { value: string },
    Array { elements: List<Value> },
    Struct { fields: HashMap<string, Value> },
    /* ... */
}

struct Interpreter {
    env: Environment,
    /* ... */
}

fn eval(ast: ASTNode, env: mut Environment) -> Result<Value, string>
fn eval_expression(expr: ASTNode, env: Environment) -> Result<Value, string>
fn exec_statement(stmt: ASTNode, env: mut Environment) -> Result<void, string>
```

**Key simplifications vs C:**
- ✅ Result type for errors
- ✅ Pattern matching for node evaluation
- ✅ Automatic memory management
- ✅ No manual value allocation

---

#### 3.7 Main Compiler Driver (~5-8 hours, ~400 lines)

**Tasks:**
- Integrate all phases
- Command-line argument parsing
- File I/O
- Error reporting
- Compilation pipeline
- Module system integration

**Output:** `compiler/main.nano`
```nanolang
fn compile_file(
    input_path: string,
    output_path: string,
    options: CompileOptions
) -> Result<void, List<string>> {
    /* Read source */
    let source: string = match read_file(input_path) {
        Ok(s) => s,
        Err(e) => return Err(["Failed to read ${input_path}: ${e}"])
    }
    
    /* Tokenize */
    let tokens: List<Token> = match tokenize(source) {
        Ok(t) => t,
        Err(e) => return Err([e])
    }
    
    /* Parse */
    let ast: ASTNode = match parse(tokens) {
        Ok(a) => a,
        Err(errors) => return Err(errors)
    }
    
    /* Type check */
    let mut env: Environment = Environment.new()
    match type_check(ast, env) {
        Ok(_) => {},
        Err(errors) => return Err(errors)
    }
    
    /* Transpile */
    let c_code: string = match transpile(ast, env) {
        Ok(code) => code,
        Err(e) => return Err([e])
    }
    
    /* Write C file and compile */
    /* ... */
    
    return Ok(())
}
```

**Key features:**
- Clean error propagation with Result
- Match expressions for pipeline
- Readable control flow

---

### Phase 4: Integration & Stage 2 Build (~8-12 hours)

**Goal:** Build pure nanolang compiler with Stage 1 (C-based)

**Tasks:**
1. Create build script `scripts/build_pure_stage2.sh`
   - Use bin/nanoc (Stage 0) to compile pure compiler
   - Link all modules
   - Test output binary

2. Test pure Stage 2 compiler
   - Compile hello.nano ✓
   - Compile factorial.nano ✓
   - Compile all examples ✓
   - Compare output to Stage 0/Stage 1

3. Benchmark performance
   - Compilation time (will be slower)
   - Memory usage
   - Output size

**Deliverables:**
- `bin/nanoc_stage2_pure` (pure nanolang implementation)
- Test results for all examples
- Performance comparison report

**Success Criteria:**
- Stage 2 (pure) can compile all examples
- Output programs run correctly
- No C FFI for core compilation

---

### Phase 5: Stage 3 Verification (~5-8 hours)

**Goal:** Prove true self-hosting with Stage 3

**Tasks:**
1. Use Stage 2 (pure) to compile itself
   ```bash
   ./bin/nanoc_stage2_pure compiler/main.nano -o bin/nanoc_stage3
   ```

2. Compare Stage 2 vs Stage 3 output
   - Compile same programs with both
   - Compare generated C code
   - Verify bit-identical or functionally identical

3. Stage 3 compilation chain
   - Stage 3 compiling test programs
   - Stage 3 compiling itself → Stage 4?
   - Verify stability

**Deliverables:**
- `bin/nanoc_stage3` (Stage 2 compiled by itself)
- Verification script `scripts/verify_stage3.sh`
- Comparison report: Stage 2 vs Stage 3

**Success Criteria:**
- ✅ Stage 2 successfully compiles itself → Stage 3
- ✅ Stage 3 produces same output as Stage 2 (or explainable differences)
- ✅ Stage 3 can compile test programs correctly
- ✅ TRUE SELF-HOSTING ACHIEVED

---

### Phase 6: Interpreter Self-Hosting (~15-20 hours) [Optional]

**Goal:** Pure nanolang interpreter

**Tasks:**
1. Rewrite eval.c in nanolang (~2,500 lines)
2. Integration with compiler modules
3. Test interpreter can run programs
4. Test interpreter can run compiler!

**Success Criteria:**
- ✅ Interpreter written in nanolang
- ✅ Can run all test programs
- ✅ Can run the compiler (meta!)

---

## IV. Timeline & Resource Estimates

### Optimistic (with experienced developer)
- Phase 1 (Language): 10 hours
- Phase 2 (Stdlib): 15 hours
- Phase 3 (Compiler): 50 hours
- Phase 4 (Integration): 8 hours
- Phase 5 (Stage 3): 5 hours
**Total: 88 hours (~11 days full-time)**

### Realistic (with testing & debugging)
- Phase 1 (Language): 15 hours
- Phase 2 (Stdlib): 20 hours
- Phase 3 (Compiler): 70 hours
- Phase 4 (Integration): 12 hours
- Phase 5 (Stage 3): 8 hours
**Total: 125 hours (~16 days full-time)**

### Conservative (with unknowns)
- Phase 1 (Language): 20 hours
- Phase 2 (Stdlib): 25 hours
- Phase 3 (Compiler): 90 hours
- Phase 4 (Integration): 15 hours
- Phase 5 (Stage 3): 12 hours
**Total: 162 hours (~20 days full-time)**

### Part-time Estimate
- 10 hours/week: 12-16 weeks (3-4 months)
- 20 hours/week: 6-8 weeks (1.5-2 months)

---

## V. Risk Mitigation

### Technical Risks

1. **Language limitations discovered during implementation**
   - *Mitigation:* Phase 1 adds features first
   - *Fallback:* Add more features as needed

2. **Performance too slow**
   - *Mitigation:* Profile and optimize hot paths
   - *Fallback:* Hybrid approach (keep C for slow parts)

3. **Memory usage too high**
   - *Mitigation:* Better GC, pool allocations
   - *Fallback:* Optimize data structures

4. **Stage 2 ≠ Stage 3 (non-deterministic output)**
   - *Mitigation:* Careful testing during Phase 3
   - *Fallback:* Functional equivalence instead of bitwise

### Scope Risks

1. **Underestimated complexity**
   - *Mitigation:* Conservative timeline (162 hours)
   - *Fallback:* Reduce scope (minimal subset)

2. **Feature creep**
   - *Mitigation:* Strict focus on self-hosting goal
   - *Fallback:* Defer nice-to-haves to Phase 6+

### Process Risks

1. **Losing momentum**
   - *Mitigation:* Break into small deliverables
   - *Fallback:* Document progress clearly

2. **Integration issues**
   - *Mitigation:* Test each component independently
   - *Fallback:* Modular design with clear interfaces

---

## VI. Success Metrics

### Quantitative
- ✅ Pure nanolang compiler: 0 C FFI calls for compilation
- ✅ Code size: < 10,000 lines nanolang
- ✅ Stage 3 verification: bit-identical or functionally equivalent output
- ✅ Test coverage: all examples compile and run
- ✅ Performance: < 5x slower than C implementation (acceptable for self-hosting)

### Qualitative
- ✅ Code readability: more readable than C version
- ✅ Maintainability: easier to understand and modify
- ✅ Correctness: all language features implemented
- ✅ Documentation: comprehensive design docs

### Milestone Checklist
- [ ] Phase 1 complete: Language features added
- [ ] Phase 2 complete: Stdlib modules ready
- [ ] Phase 3.1 complete: Lexer in nanolang ✓
- [ ] Phase 3.2 complete: Parser in nanolang ✓
- [ ] Phase 3.3 complete: Environment in nanolang ✓
- [ ] Phase 3.4 complete: Type checker in nanolang ✓
- [ ] Phase 3.5 complete: Transpiler in nanolang ✓
- [ ] Phase 3.6 complete: Interpreter in nanolang ✓
- [ ] Phase 3.7 complete: Main driver in nanolang ✓
- [ ] Phase 4 complete: Stage 2 (pure) builds ✓
- [ ] Phase 5 complete: Stage 3 verified ✓
- [ ] **TRUE SELF-HOSTING ACHIEVED** 🎉

---

## VII. Next Steps

### Immediate (This Week)
1. ✅ Review and approve this PLAN.md
2. ⏳ Review and approve LANGUAGE_IMPROVEMENTS_FOR_SELFHOST.md
3. ⏳ Decide on Phase 1 priority features
4. ⏳ Create detailed Phase 1 spec with examples

### Short Term (Next 2 Weeks)
1. ⏳ Implement Phase 1 language features
2. ⏳ Test new features thoroughly
3. ⏳ Update documentation
4. ⏳ Begin Phase 2 stdlib modules

### Medium Term (Next 1-2 Months)
1. ⏳ Implement Phase 2 modules
2. ⏳ Begin Phase 3 compiler components
3. ⏳ Regular progress updates
4. ⏳ Incremental testing

### Long Term (2-3 Months)
1. ⏳ Complete Phase 3 compiler
2. ⏳ Phase 4 integration
3. ⏳ Phase 5 Stage 3 verification
4. ⏳ Documentation and announcement

---

## VIII. Conclusion

This plan achieves **TRUE self-hosting** by:

1. ✨ **Enhancing the language first** - Make nanolang better than C for compiler implementation
2. 📦 **Building reusable modules** - Benefit the entire ecosystem
3. 🏗️ **Systematic implementation** - Phase-by-phase with clear milestones
4. ✅ **Rigorous verification** - Stage 3 proves self-hosting
5. 📚 **Comprehensive documentation** - Preserve knowledge and rationale

**Key Innovation:** Leverage nanolang's high-level features to make implementation easier and shorter than the C version (35% code reduction + better safety + readability).

**Expected Outcome:**
- Pure nanolang compiler (~8,600 lines)
- Stage 3 verification (Stage 2 ≡ Stage 3)
- Reusable stdlib modules
- Proof that nanolang is sufficiently expressive
- Foundation for future language evolution

**Timeline:** 88-162 hours (11-20 days full-time, or 3-4 months part-time)

**Next Review Point:** After approving language enhancements, create detailed Phase 1 specification.

---

*Last Updated: November 29, 2024*
*Status: Planning Phase*
*Current Focus: Review and approval of enhancement strategy*
