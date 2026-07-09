# PLAN.md — OPL (Operator Prompt Language) DSL Bundle

Date: 2026-01-08

This directory is a **self-contained specification pack** for a tiny human-typeable DSL called **OPL** (Operator Prompt Language).
The goal is to enable:
- **Concise, unambiguous** prompts and tool calls
- **Executable specifications** for systems and agents
- A **deterministic compilation pipeline**:
  `OPL text → AST (JSON) → validated → Execution Plan (JSON) → tool calls`

You said you will implement this in **NanoLang** by handing these files to another advanced coding LLM.
This bundle is intentionally complete: **no dangling references**; all files are in this **single flat directory**.

---

## 0) Directory Contents (flat)

Entry point:
- **PLAN.md** (this file)

Specifications:
- **SPEC.md** — goals, non-goals, design constraints, semantics overview
- **GRAMMAR.ebnf** — full EBNF grammar for OPL
- **TOKENS.md** — lexical rules and tokenization details
- **AST_IR.schema.json** — JSON Schema for the OPL AST/IR
- **PLAN_IR.schema.json** — JSON Schema for the compiled Execution Plan
- **VALIDATION.md** — static validation rules (deterministic checks)
- **SEMANTICS.md** — evaluation model, scoping, execution semantics, determinism rules
- **BUILTINS.md** — optional built-in pure functions for expressions (whitelist)
- **ERRORS.md** — canonical error codes + messages shape
- **STYLE.md** — conventions for formatting + determinism

Examples & tests:
- **EXAMPLES.opl** — curated language examples
- **EXAMPLES.expected_ast.json** — expected AST for EXAMPLES.opl (golden output; partial but representative)
- **EXAMPLES.expected_plan.json** — expected plan for a subset (golden output; partial but representative)
- **TESTS.md** — test strategy and required test cases
- **TESTS.cases.json** — machine-readable test cases (inputs → expected outputs/errors)

Implementation guidance:
- **IMPLEMENTATION_GUIDE.md** — how to build parser/validator/compiler in NanoLang

---

## 1) What the generated NanoLang example should implement

Implement a reference toolchain with these stages:

### Stage A — Lexer/Tokenizer
Input: OPL text  
Output: token stream (with line/col positions)  
Rules: TOKENS.md

### Stage B — Parser
Input: tokens  
Output: AST that conforms to AST_IR.schema.json  
Grammar: GRAMMAR.ebnf

### Stage C — Validator
Input: AST  
Output: either:
- `{ "ok": true }` or
- `{ "ok": false, "errors": [...] }` using ERRORS.md codes  
Rules: VALIDATION.md + SEMANTICS.md

### Stage D — Compiler (Lowering)
Input: validated AST  
Output: Execution Plan JSON conforming to PLAN_IR.schema.json  
Rules: SEMANTICS.md

### Stage E — Golden tests
Use TESTS.cases.json + golden files for deterministic regression.

---

## 2) Expected CLI interface (example)

The generated NanoLang example should provide a CLI resembling:

- `opl parse <file.opl> --out <ast.json>`
- `opl validate <file.opl> --out <validation.json>`
- `opl compile <file.opl> --out <plan.json>`
- `opl test` (runs all cases from TESTS.cases.json)

The CLI is only an example; the key is that outputs match schemas and golden tests.

---

## 3) Determinism requirements (must-haves)

1. Parser output is **stable** (ordering preserved as written).
2. Validator error ordering is stable:
   - sort by location (line, col), then by error code, then by path
3. Compiler output is stable:
   - plan steps appear in source order
4. No implicit defaults except those explicitly stated in SPEC/SEMANTICS.

---

## 4) Capability gating (safety + auditability)

A call is only permitted if the surrounding block declares `uses <tool.ref>`.

Example:
```opl
agent a {
  uses web.search
  call web.search { query: "x" }
}
```

This is mandatory unless the implementation opts into a relaxed mode.

---

## 5) Minimal implementation subset (if you want to ship sooner)

- Support: `agent` blocks, `uses`, `input`, `let`, `call ... as`, `assert`, `emit`, `when ... -> ...`
- Skip: `on` rules, `schema` blocks, `callExpr` builtins beyond `len()`

The tests include both core and optional features; see TESTS.md.

---

## 6) Hand-off prompt (copy/paste to another coding LLM)

> You are implementing the OPL DSL toolchain described by the files in this directory.
> The implementation must be self-contained and use only these files as the source of truth.
> Implement lexer, parser, validator, compiler, and tests matching the schemas and golden outputs.
> Do not invent new syntax or semantics; if something is unclear, follow SPEC.md and SEMANTICS.md.
> Output must be deterministic and conform to AST_IR.schema.json and PLAN_IR.schema.json.
> Provide a CLI as described in PLAN.md and implement golden tests from TESTS.cases.json.

---

## 7) Start here

Read in this order:
1) SPEC.md  
2) TOKENS.md  
3) GRAMMAR.ebnf  
4) AST_IR.schema.json  
5) VALIDATION.md  
6) SEMANTICS.md  
7) TESTS.md + TESTS.cases.json  
8) EXAMPLES.opl + expected outputs  
