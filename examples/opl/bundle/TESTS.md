# TESTS.md — Test Strategy

## Goals
- Verify lexer + parser correctness
- Verify validation rules
- Verify compilation determinism and schema conformance

## Required categories
1) Parsing: valid blocks and statements
2) Validation: missing uses, unresolved id, duplicate key, type mismatch
3) Compilation: call ordering, when→if lowering, emit

## Golden files
- EXAMPLES.expected_ast.json
- EXAMPLES.expected_plan.json

## Machine-readable cases
Use TESTS.cases.json as the primary test runner input.
