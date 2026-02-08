#!/usr/bin/env python3
"""
Extract a human-readable formal specification from the NanoCore Coq proofs.

Reads formal/Syntax.v, formal/Semantics.v, and formal/Typing.v to generate
a Markdown document describing the NanoCore language specification with:
- Type system overview
- Operator semantics table
- Evaluation rules in plain English
- Type rules summary
- Edge case catalog
"""

import re
import sys
import os
from pathlib import Path

FORMAL_DIR = Path(__file__).parent.parent / "formal"

def read_file(name):
    path = FORMAL_DIR / name
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return path.read_text()


# ── Syntax extraction ────────────────────────────────────────────────────────

def extract_types(syntax):
    """Extract type definitions from Syntax.v"""
    types = []
    m = re.search(r'Inductive ty : Type :=\n(.*?)\.', syntax, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            line = line.strip()
            match = re.match(r'\|\s*(\w+)\s*(?::.*)?(?:\(\*\s*(.*?)\s*\*\))?', line)
            if match:
                name = match.group(1)
                comment = match.group(2) or ""
                types.append((name, comment))
    return types

def extract_binops(syntax):
    """Extract binary operators from Syntax.v"""
    ops = []
    m = re.search(r'Inductive binop : Type :=\n(.*?)\.', syntax, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            line = line.strip()
            match = re.match(r'\|\s*(\w+)\s*:\s*binop\s*(?:\(\*\s*(.*?)\s*\*\))?', line)
            if match:
                name = match.group(1)
                comment = match.group(2) or ""
                ops.append((name, comment))
    return ops

def extract_unops(syntax):
    """Extract unary operators from Syntax.v"""
    ops = []
    m = re.search(r'Inductive unop : Type :=\n(.*?)\.', syntax, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            line = line.strip()
            match = re.match(r'\|\s*(\w+)\s*:\s*unop\s*(?:\(\*\s*(.*?)\s*\*\))?', line)
            if match:
                name = match.group(1)
                comment = match.group(2) or ""
                ops.append((name, comment))
    return ops

def extract_exprs(syntax):
    """Extract expression forms from Syntax.v"""
    exprs = []
    m = re.search(r'Inductive expr : Type :=\n(.*?)\.', syntax, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            line = line.strip()
            match = re.match(r'\|\s*(\w+)\s*(?::.*)?(?:\(\*\s*(.*?)\s*\*\))?', line)
            if match:
                name = match.group(1)
                comment = match.group(2) or ""
                exprs.append((name, comment))
    return exprs


# ── Semantics extraction ─────────────────────────────────────────────────────

def extract_eval_rules(semantics):
    """Extract evaluation rules from Semantics.v"""
    rules = []
    # Match each rule constructor
    pattern = r'\(\*\*\s*(.*?)\s*\*\)\s*\|\s*(\w+)\s*:\s*forall[^,]+'
    for m in re.finditer(pattern, semantics):
        comment = m.group(1).strip()
        name = m.group(2).strip()
        rules.append((name, comment))
    return rules

def extract_arith_semantics(semantics):
    """Extract arithmetic operator definitions"""
    ops = []
    m = re.search(r'Definition eval_arith_binop.*?end\.', semantics, re.DOTALL)
    if m:
        block = m.group(0)
        for line in block.split('\n'):
            match = re.match(r'\s*\|\s*(\w+)\s*=>\s*(.*)', line.strip())
            if match and 'None' not in match.group(2) and 'match' not in match.group(2):
                op = match.group(1)
                result = match.group(2).rstrip()
                ops.append((op, result))
        # Division special case
        ops.append(("OpDiv", "n1 / n2 (returns None if n2 = 0)"))
        ops.append(("OpMod", "n1 mod n2 (returns None if n2 = 0)"))
    return ops

def extract_cmp_semantics(semantics):
    """Extract comparison operator definitions"""
    ops = []
    m = re.search(r'Definition eval_cmp_binop.*?end\.', semantics, re.DOTALL)
    if m:
        block = m.group(0)
        for line in block.split('\n'):
            match = re.match(r'\s*\|\s*(\w+)\s*=>\s*Some\s*\(VBool\s*\((.*?)\)\)', line.strip())
            if match:
                op = match.group(1)
                result = match.group(2)
                ops.append((op, result))
    return ops


# ── Typing extraction ────────────────────────────────────────────────────────

def extract_type_rules(typing):
    """Extract typing rules from Typing.v"""
    rules = []
    pattern = r'\(\*\*\s*(.*?)\s*\*\)\s*\|\s*(\w+)\s*:\s*forall'
    for m in re.finditer(pattern, typing):
        comment = m.group(1).strip()
        name = m.group(2).strip()
        rules.append((name, comment))
    return rules


# ── Edge cases ───────────────────────────────────────────────────────────────

EDGE_CASES = [
    ("Division by zero", "`a / 0` and `a % 0`", "Returns `None` (evaluation gets stuck). Programs must guard against zero divisors."),
    ("Array out of bounds", "`arr[i]` where `i >= length(arr)` or `i < 0`", "Returns `None` (evaluation gets stuck). The type system does not prevent this."),
    ("Short-circuit AND", "`false and e2`", "Returns `false` without evaluating `e2`. Side effects in `e2` are not executed."),
    ("Short-circuit OR", "`true or e2`", "Returns `true` without evaluating `e2`. Side effects in `e2` are not executed."),
    ("String indexing", "`s[i]` where `i >= length(s)` or `i < 0`", "Returns `\"\"` (empty string). String indexing is total — it never gets stuck."),
    ("Empty array literal", "`[]`", "Has type `array<T>` for any `T`. Type is determined by context."),
    ("Let binding scoping", "`let x = e1 in e2`", "After `e2` evaluates, the `x` binding is popped. Mutations to outer variables are preserved."),
    ("Closure capture", "`fun (x : T) => body`", "Captures the current environment at closure creation time (lexical scoping)."),
    ("While loop value", "`while cond do body`", "Always returns `unit`. Loop value is discarded each iteration."),
    ("Record field update", "`set x.f = v`", "Updates the field in-place. Requires `x` in scope and `f` is a valid field with a prior value."),
    ("Pattern matching", "`match e { Tag x => body, ... }`", "Binding `x` is scoped to the match arm body. After the body evaluates, the binding is popped."),
]


# ── Document generation ──────────────────────────────────────────────────────

def binop_symbol(name):
    mapping = {
        "OpAdd": "+", "OpSub": "-", "OpMul": "*", "OpDiv": "/", "OpMod": "%",
        "OpEq": "==", "OpNe": "!=", "OpLt": "<", "OpLe": "<=",
        "OpGt": ">", "OpGe": ">=", "OpAnd": "and", "OpOr": "or",
        "OpStrCat": "+ (strings)"
    }
    return mapping.get(name, name)

def unop_symbol(name):
    mapping = {
        "OpNeg": "- (unary)", "OpNot": "not",
        "OpStrLen": "str_length", "OpArrayLen": "array_length"
    }
    return mapping.get(name, name)

def type_name(name):
    mapping = {
        "TInt": "int", "TBool": "bool", "TString": "string", "TUnit": "unit",
        "TArrow": "T1 -> T2", "TArray": "array<T>",
        "TRecord": "{f1: T1, ..., fn: Tn}", "TVariant": "Tag1(T1) | ... | Tagn(Tn)"
    }
    return mapping.get(name, name)

def expr_name(name):
    mapping = {
        "EInt": "integer literal", "EBool": "boolean literal",
        "EString": "string literal", "EUnit": "unit literal",
        "EVar": "variable reference", "EBinOp": "binary operation",
        "EUnOp": "unary operation", "EIf": "if-then-else",
        "ELet": "let binding", "ESet": "mutable assignment",
        "ESeq": "sequence (e1; e2)", "EWhile": "while loop",
        "ELam": "lambda abstraction", "EApp": "function application",
        "EFix": "recursive function (fix)", "EArray": "array literal",
        "EIndex": "array indexing", "EArraySet": "array update",
        "EArrayPush": "array push", "ERecord": "record literal",
        "EField": "field access", "ESetField": "field update",
        "EConstruct": "variant constructor", "EMatch": "pattern matching",
        "EStrIndex": "string indexing"
    }
    return mapping.get(name, name)


def generate_spec():
    syntax = read_file("Syntax.v")
    semantics = read_file("Semantics.v")
    typing = read_file("Typing.v")

    types = extract_types(syntax)
    binops = extract_binops(syntax)
    unops = extract_unops(syntax)
    exprs = extract_exprs(syntax)
    eval_rules = extract_eval_rules(semantics)
    type_rules = extract_type_rules(typing)

    doc = []
    doc.append("# NanoCore Formal Specification")
    doc.append("")
    doc.append("*Auto-generated from the Coq formal proofs in `formal/`.*")
    doc.append("*This document describes the formally verified subset of NanoLang.*")
    doc.append("")
    doc.append("## Verification Status")
    doc.append("")
    doc.append("All properties below are proven **axiom-free** in Coq (~5,635 LOC):")
    doc.append("")
    doc.append("| Property | File | Status |")
    doc.append("|----------|------|--------|")
    doc.append("| Type Soundness | `Soundness.v` | Proven |")
    doc.append("| Progress | `Progress.v` | Proven |")
    doc.append("| Determinism | `Determinism.v` | Proven |")
    doc.append("| Big-step/Small-step Equivalence | `Equivalence.v` | Proven |")
    doc.append("| Computable Evaluator Soundness | `EvalFn.v` | Partial (10/25 cases) |")
    doc.append("")

    # Types
    doc.append("## Types")
    doc.append("")
    doc.append("NanoCore has 8 types:")
    doc.append("")
    doc.append("| Coq Name | NanoLang Syntax | Description |")
    doc.append("|----------|----------------|-------------|")
    for name, comment in types:
        doc.append(f"| `{name}` | `{type_name(name)}` | {comment} |")
    doc.append("")

    # Expression forms
    doc.append("## Expression Forms")
    doc.append("")
    doc.append(f"NanoCore has {len(exprs)} expression forms:")
    doc.append("")
    doc.append("| # | Coq Constructor | Description |")
    doc.append("|---|----------------|-------------|")
    for i, (name, comment) in enumerate(exprs, 1):
        doc.append(f"| {i} | `{name}` | {expr_name(name)}{' — ' + comment if comment else ''} |")
    doc.append("")

    # Operator semantics
    doc.append("## Operator Semantics")
    doc.append("")
    doc.append("### Binary Operators")
    doc.append("")
    doc.append("| Operator | Symbol | Operand Types | Result Type | Notes |")
    doc.append("|----------|--------|--------------|-------------|-------|")
    for name, comment in binops:
        sym = binop_symbol(name)
        if name in ["OpAdd", "OpSub", "OpMul"]:
            doc.append(f"| `{name}` | `{sym}` | int, int | int | |")
        elif name == "OpDiv":
            doc.append(f"| `{name}` | `/` | int, int | int | Returns None if divisor = 0 |")
        elif name == "OpMod":
            doc.append(f"| `{name}` | `%` | int, int | int | Returns None if divisor = 0 |")
        elif name in ["OpEq", "OpNe"]:
            doc.append(f"| `{name}` | `{sym}` | int/bool/string | bool | Polymorphic equality |")
        elif name in ["OpLt", "OpLe", "OpGt", "OpGe"]:
            doc.append(f"| `{name}` | `{sym}` | int, int | bool | |")
        elif name in ["OpAnd", "OpOr"]:
            doc.append(f"| `{name}` | `{sym}` | bool, bool | bool | Short-circuit evaluation |")
        elif name == "OpStrCat":
            doc.append(f"| `{name}` | `+` | string, string | string | String concatenation |")
    doc.append("")

    doc.append("### Unary Operators")
    doc.append("")
    doc.append("| Operator | Symbol | Operand Type | Result Type |")
    doc.append("|----------|--------|-------------|-------------|")
    for name, comment in unops:
        sym = unop_symbol(name)
        if name == "OpNeg":
            doc.append(f"| `{name}` | `{sym}` | int | int |")
        elif name == "OpNot":
            doc.append(f"| `{name}` | `{sym}` | bool | bool |")
        elif name == "OpStrLen":
            doc.append(f"| `{name}` | `{sym}` | string | int |")
        elif name == "OpArrayLen":
            doc.append(f"| `{name}` | `{sym}` | array<T> | int |")
    doc.append("")

    # Evaluation rules
    doc.append("## Evaluation Rules")
    doc.append("")
    doc.append("The big-step semantics (`eval renv e renv' v`) evaluates expression `e`")
    doc.append("in environment `renv`, producing value `v` and updated environment `renv'`.")
    doc.append("")
    doc.append("Key properties:")
    doc.append("- **Eager evaluation** (call-by-value)")
    doc.append("- **Deterministic** (each expression has at most one value)")
    doc.append("- **Lexical scoping** via closures")
    doc.append("- **Store-passing** style for mutable variables")
    doc.append("")
    doc.append("| Rule | Description |")
    doc.append("|------|-------------|")
    for name, comment in eval_rules:
        doc.append(f"| `{name}` | {comment} |")
    doc.append("")

    # Type rules
    doc.append("## Type Rules")
    doc.append("")
    doc.append("The typing judgment (`has_type ctx e t`) assigns type `t` to expression `e`")
    doc.append("in typing context `ctx`.")
    doc.append("")
    doc.append("| Rule | Description |")
    doc.append("|------|-------------|")
    for name, comment in type_rules:
        doc.append(f"| `{name}` | {comment} |")
    doc.append("")

    # Edge cases
    doc.append("## Edge Cases")
    doc.append("")
    doc.append("These behaviors are specified by the formal semantics and proven correct:")
    doc.append("")
    doc.append("| Case | Expression | Behavior |")
    doc.append("|------|-----------|----------|")
    for case, expr, behavior in EDGE_CASES:
        doc.append(f"| {case} | {expr} | {behavior} |")
    doc.append("")

    # Proven properties
    doc.append("## Proven Properties")
    doc.append("")
    doc.append("### Type Soundness (Soundness.v)")
    doc.append("")
    doc.append("If expression `e` has type `t` in context `ctx`, and `e` evaluates to value `v`,")
    doc.append("then `v` has type `t`. Types are preserved through evaluation.")
    doc.append("")
    doc.append("### Progress (Progress.v)")
    doc.append("")
    doc.append("If expression `e` has type `t`, then either `e` is a value, or `e` can take")
    doc.append("a step (small-step semantics). Well-typed programs don't get stuck.")
    doc.append("")
    doc.append("### Determinism (Determinism.v)")
    doc.append("")
    doc.append("If `eval renv e renv1 v1` and `eval renv e renv2 v2`, then `v1 = v2` and")
    doc.append("`renv1 = renv2`. Evaluation is a partial function.")
    doc.append("")
    doc.append("### Big-step/Small-step Equivalence (Equivalence.v)")
    doc.append("")
    doc.append("The big-step semantics (natural semantics) and the small-step semantics")
    doc.append("(structural operational semantics) agree on all programs. This provides")
    doc.append("confidence that the specification is consistent.")
    doc.append("")
    doc.append("### Computable Evaluator (EvalFn.v)")
    doc.append("")
    doc.append("A fuel-based computable evaluator `eval_fn` is provided with partial soundness")
    doc.append("proofs. If `eval_fn fuel renv e = Some (renv', v)`, then `eval renv e renv' v`.")
    doc.append("The evaluator can be extracted to OCaml for use as a reference interpreter.")
    doc.append("")

    # NanoCore vs full NanoLang
    doc.append("## NanoCore vs Full NanoLang")
    doc.append("")
    doc.append("NanoCore is a subset of NanoLang. The following features are in NanoLang")
    doc.append("but **not** in the formally verified NanoCore subset:")
    doc.append("")
    doc.append("| Feature | NanoLang | NanoCore |")
    doc.append("|---------|---------|----------|")
    doc.append("| Integers | Yes | Yes |")
    doc.append("| Booleans | Yes | Yes |")
    doc.append("| Strings | Yes | Yes |")
    doc.append("| Floats | Yes | **No** |")
    doc.append("| Arrays | Yes | Yes |")
    doc.append("| Structs/Records | Yes | Yes |")
    doc.append("| Unions/Variants | Yes | Yes |")
    doc.append("| Pattern Matching | Yes | Yes |")
    doc.append("| Tuples | Yes | **No** |")
    doc.append("| Hashmaps | Yes | **No** |")
    doc.append("| Enums | Yes | **No** |")
    doc.append("| Generics | Yes | **No** |")
    doc.append("| Modules/Imports | Yes | **No** |")
    doc.append("| FFI/Extern | Yes | **No** |")
    doc.append("| Opaque Types | Yes | **No** |")
    doc.append("| Unsafe Blocks | Yes | **No** |")
    doc.append("| For Loops | Yes | **No** (equivalent to while+let) |")
    doc.append("| Print/Assert | Yes | **No** |")
    doc.append("")
    doc.append("Use `nanoc --trust-report <file.nano>` to see which functions in your")
    doc.append("program fall within the verified NanoCore subset.")
    doc.append("")

    return "\n".join(doc)


if __name__ == "__main__":
    output_path = FORMAL_DIR.parent / "docs" / "FORMAL_SPECIFICATION.md"
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])

    spec = generate_spec()

    os.makedirs(output_path.parent, exist_ok=True)
    output_path.write_text(spec)
    print(f"Generated {output_path} ({len(spec)} bytes, {spec.count(chr(10))} lines)")
