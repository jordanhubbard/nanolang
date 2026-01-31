### OPL (Operator Prompt Language) example

This directory contains a **self-contained OPL DSL toolchain** implemented in NanoLang:

- **Lexer**: `opl_lexer.nano`
- **Parser**: `opl_parser.nano` → AST JSON (`bundle/AST_IR.schema.json`)
- **Validator**: `opl_validate.nano` → validation JSON (`bundle/ERRORS.md` codes)
- **Compiler/Lowering**: `opl_compile.nano` → PLAN_IR JSON (`bundle/PLAN_IR.schema.json`)
- **Codegen**: `opl_codegen.nano` → NanoLang program skeleton from PLAN_IR
- **CLI**: `opl_cli.nano` → `parse|validate|compile|test|codegen|build`

The original spec bundle is vendored in `bundle/` (including `EXAMPLES.opl`, expected golden outputs, and `TESTS.cases.json`).

### Build and run the CLI

Compile the CLI:

```bash
perl -e 'alarm 60; exec @ARGV' ./bin/nanoc examples/opl/opl_cli.nano -o /tmp/opl_cli
```

Commands:

```bash
# Parse → AST JSON
/tmp/opl_cli parse examples/opl/bundle/EXAMPLES.opl --out /tmp/ast.json

# Validate → { ok, errors[] }
/tmp/opl_cli validate examples/opl/bundle/EXAMPLES.opl --out /tmp/validate.json

# Compile/lower → PLAN_IR JSON (minimal subset: agent blocks, call/let/assert/emit/when)
/tmp/opl_cli compile examples/opl/bundle/EXAMPLES.opl --out /tmp/plan.json

# Run bundled cases (golden_ast, golden_plan, validate_error)
/tmp/opl_cli test

# PLAN_IR → NanoLang skeleton program
/tmp/opl_cli codegen examples/opl/bundle/EXAMPLES.expected_plan.json --out /tmp/generated.nano

# End-to-end: codegen + compile to an executable (timeout enforced)
# NOTE: build currently requires --out to be exactly /tmp/opl_codegen_tmp_bin.
/tmp/opl_cli build examples/opl/bundle/EXAMPLES.expected_plan.json --out /tmp/opl_codegen_tmp_bin
```

### Notes / constraints

- **Determinism**:
  - Parser/AST ordering follows source order.
  - Validator error ordering is deterministic (see `opl_validate.nano`).
  - Lowering emits steps in source order.
- **Minimal subset**:
  - The compiler focuses on the subset described in `bundle/PLAN.md` (“Minimal implementation subset”).
  - `when` lowers to PLAN_IR `if` with `then` steps and empty `else`.
- **Codegen is a skeleton**:
  - Generated programs are intentionally **auditable**: they print intended calls and emit values.
  - Tool calls are stubbed as `call_tool(tool, args)` returning `null`.
  - Nested `if.then` steps are currently not executed (left as a TODO in the generated program).

