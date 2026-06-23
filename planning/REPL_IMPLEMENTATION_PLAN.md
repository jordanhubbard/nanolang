# Full-Featured REPL Implementation Plan

## Current Status

The current REPL (`examples/language/simple_repl.nano`) is a **calculator REPL** that only evaluates integer expressions by wrapping them in a temporary `main()` function, compiling, and executing.

### Current Limitations

1. ❌ Only evaluates integer expressions (no other types)
2. ❌ No persistent variable definitions
3. ❌ No function definitions
4. ❌ No multi-line input
5. ❌ No module imports
6. ❌ No stateful session context

## Architecture Options

### Option A: Compile-and-Execute (Current Approach - Extended)

**Pros:**
- Reuses existing compiler infrastructure
- Full language feature support automatically
- No separate interpreter needed

**Cons:**
- Slow (compile on every input)
- No true interactivity
- Difficult to maintain state between invocations

### Option B: True Interpreter (Recommended)

**Pros:**
- Fast evaluation
- Natural state management
- True REPL experience
- Can add debugging features

**Cons:**
- Requires new interpreter implementation
- Must keep in sync with language features
- More initial development effort

### Option C: Hybrid Approach

Compile functions and definitions once, interpret expressions:
- Persistent context with compiled artifacts
- Fast expression evaluation
- State management via symbol table

## Implementation Plan

### Phase 1: Core Infrastructure (8-12 hours)

#### 1.1. Stateful Session Manager (Task #7)
Create a session context that persists across REPL invocations:

```nano
struct REPLSession {
    variables: HashMap<string, Value>,
    functions: HashMap<string, CompiledFunction>,
    types: HashMap<string, TypeDef>,
    imports: List<string>,
    history: List<string>
}
```

**Tasks:**
- Design session state structure
- Implement variable storage (type + value)
- Implement function storage
- Add type definition storage
- Create session serialization (save/load)

**Estimated:** 4-6 hours

#### 1.2. Parser Modifications (4-6 hours)

Extend parser to handle REPL-specific constructs:
- Distinguish expressions from statements
- Handle partial/incomplete input
- Support multi-line mode indicators
- Parse variable definitions for storage
- Parse function definitions for storage

**Estimated:** 4-6 hours

### Phase 2: Feature Implementation (12-18 hours)

#### 2.1. Persistent Variables (Task #2) - 3-4 hours

**Requirements:**
- Store variable name, type, and value
- Type inference from expression
- Immutable and mutable variables
- Variable shadowing (new definition replaces old)

**Example:**
```nano
repl> let x: int = 42
Defined: x = 42

repl> let y: int = (+ x 10)
Defined: y = 52

repl> set x 100
Updated: x = 100

repl> y
52
```

**Implementation:**
1. Parse `let` statements
2. Evaluate expression and store result
3. Store in session.variables
4. Allow later expressions to reference stored variables

#### 2.2. Function Definitions (Task #3) - 4-6 hours

**Requirements:**
- Parse function definitions
- Compile to executable form
- Store in session context
- Allow calling from later expressions
- Support shadow tests for REPL-defined functions

**Example:**
```nano
repl> fn double(x: int) -> int { return (* x 2) }
Defined: double(int) -> int

repl> (double 21)
42

repl> fn add(a: int, b: int) -> int { return (+ a b) }
Defined: add(int, int) -> int

repl> (add (double 5) 3)
13
```

**Implementation:**
1. Parse function definition
2. Type check with existing context
3. Compile to C function or bytecode
4. Store in session.functions
5. Make available for subsequent calls

#### 2.3. All Type Support (Task #4) - 2-3 hours

**Requirements:**
- Support all NanoLang types, not just int
- String, float, bool, arrays, structs, unions
- Type inference from expressions

**Example:**
```nano
repl> let name: string = "Alice"
Defined: name = "Alice"

repl> let pi: float = 3.14159
Defined: pi = 3.14159

repl> let numbers: array<int> = [1, 2, 3]
Defined: numbers = [1, 2, 3]
```

**Implementation:**
1. Extend value storage to handle all types
2. Update display logic for each type
3. Ensure serialization works for all types

#### 2.4. Multi-line Input (Task #5) - 3-5 hours

**Requirements:**
- Detect incomplete input (unclosed braces, etc.)
- Continue prompting until complete
- Support multi-line function/struct definitions
- Visual indicator for continuation

**Example:**
```nano
repl> fn factorial(n: int) -> int {
....>     if (<= n 1) {
....>         return 1
....>     } else {
....>         return (* n (factorial (- n 1)))
....>     }
....> }
Defined: factorial(int) -> int

repl> (factorial 5)
120
```

**Implementation:**
1. Track brace/bracket/paren depth
2. Show continuation prompt when incomplete
3. Buffer lines until complete
4. Parse combined input

#### 2.5. Module Imports (Task #6) - 2-4 hours

**Requirements:**
- Import modules at REPL prompt
- Make module functions available
- Persistent imports across session

**Example:**
```nano
repl> from "std/math" import sqrt, pow
Imported: sqrt, pow from std/math

repl> (sqrt 16.0)
4.0

repl> (pow 2.0 8.0)
256.0
```

**Implementation:**
1. Parse import statements
2. Load module metadata
3. Register imported functions
4. Store in session.imports

### Phase 3: User Experience (4-6 hours)

#### 3.1. Enhanced Features

**Command History:**
- Up/down arrows for history
- Search history (Ctrl+R)
- Save history to file

**Tab Completion:**
- Complete variable names
- Complete function names
- Complete type names

**Help System:**
- `:help` command
- Show variable types with `:type varname`
- List definitions with `:vars`, `:funcs`
- Clear session with `:clear`

**Error Recovery:**
- Don't exit on errors
- Show helpful error messages
- Suggest corrections

#### 3.2. Special Commands

```
:help           Show help
:quit / :exit   Exit REPL
:clear          Clear session
:vars           List defined variables
:funcs          List defined functions
:type <expr>    Show type of expression
:save <file>    Save session to file
:load <file>    Load session from file
:history        Show command history
```

### Phase 4: Testing & Polish (2-4 hours)

- Comprehensive test suite
- Edge case handling
- Performance optimization
- Documentation

## Total Estimated Effort

- **Phase 1:** 8-12 hours
- **Phase 2:** 12-18 hours
- **Phase 3:** 4-6 hours
- **Phase 4:** 2-4 hours

**Total:** 26-40 hours

## Dependencies

### Required First:
- ✅ All basic language features
- ✅ Type checking infrastructure
- ✅ Module system
- ✅ Standard library

### Helpful But Optional:
- First-class functions (for advanced use)
- Generics (for rich data structures)
- Better error messages

## Implementation Strategy

### Recommended Approach:

1. **Week 1:** Core infrastructure (Session manager + Parser mods)
2. **Week 2:** Variables + Functions (Tasks #2, #3)
3. **Week 3:** Types + Multi-line + Imports (Tasks #4, #5, #6)
4. **Week 4:** UX polish + Testing

### Incremental Milestones:

**Milestone 1:** Persistent variables (like Python/Ruby REPL)
**Milestone 2:** Add functions (like IPython)
**Milestone 3:** Multi-line + imports (like Scala REPL)
**Milestone 4:** Full-featured (like Haskell GHCi)

## Alternative: Jupyter Kernel

Instead of a CLI REPL, consider implementing a Jupyter kernel:

**Pros:**
- Rich output (HTML, images, plots)
- Notebook format (save/share sessions)
- Integration with data science tools
- Web-based UI

**Cons:**
- More complex protocol
- Requires separate kernel process
- Less suitable for quick testing

## Current Simple REPL

Location: `examples/language/simple_repl.nano`

Architecture:
```
User Input → Wrap in main() → Compile to C → Execute → Show Result
```

This works for quick calculations but lacks:
- State persistence
- Function definitions
- Type flexibility
- Multi-line support

## Next Steps

1. **Decide architecture:** True interpreter vs Hybrid
2. **Build session manager** (Task #7)
3. **Implement persistent variables** (Task #2)
4. **Add function support** (Task #3)
5. **Extend type support** (Task #4)
6. **Add multi-line** (Task #5)
7. **Add imports** (Task #6)
8. **Polish UX**

## References

### Good REPL Examples:
- **Python:** Simple, persistent state, great UX
- **Scala:** Multi-line support, type inference
- **Haskell GHCi:** Powerful, :type/:info commands
- **Rust Evcxr:** Compile-based but with state
- **Node.js:** Fast, good multiline handling

### Implementation Guides:
- [Build Your Own REPL](https://www.cs.miami.edu/home/burt/learning/Csc521.032/notes/repl.html)
- [REPL Design Patterns](https://tratt.net/laurie/blog/2023/which_repl_evaluates_this_expression_correctly.html)

---

**Status:** Planning complete, ready for implementation
**Priority:** HIGH (great for learning and debugging)
**Complexity:** Moderate to High (26-40 hours estimated)
