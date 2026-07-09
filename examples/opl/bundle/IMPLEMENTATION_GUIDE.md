# IMPLEMENTATION_GUIDE.md — NanoLang Reference Implementation Plan

## Components

### 1) Lexer
- Follow TOKENS.md
- Track line/col for each token start

### 2) Parser
- Recursive descent from GRAMMAR.ebnf
- Emit AST matching AST_IR.schema.json
- Attach `loc` to every node

### 3) Validator
- Implement VALIDATION.md rules
- Build symbol tables per block
- Enforce `uses` gating for tool calls

### 4) Compiler
- Lower AST to plan per SEMANTICS.md and PLAN_IR.schema.json
- Encode identifiers as `{ "$var": "name" }`
- Encode compound expressions as `{ "$expr": <exprIR> }`
- Lower `when` to `if` steps

### 5) Tests
- Parse EXAMPLES.opl and compare to EXAMPLES.expected_ast.json
- Compile (at least agents in the golden file) and compare to EXAMPLES.expected_plan.json
- Run additional cases in TESTS.cases.json
