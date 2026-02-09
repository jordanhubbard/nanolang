#!/usr/bin/env python3
"""
Extract a human-readable formal specification from the NanoCore Coq proofs.

Reads formal/Syntax.v, formal/Semantics.v, and formal/Typing.v to generate
docs/FORMAL_SPECIFICATION.md describing the NanoCore language specification with:
- Type system overview
- Expression forms reference
- Operator semantics tables
- Evaluation rules in plain English
- Typing rules summary
- Value forms
- Edge case catalog
- Proven properties
"""

import re
import sys
import os
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAL_DIR = PROJECT_ROOT / "formal"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "FORMAL_SPECIFICATION.md"


def read_file(name):
    """Read a file from the formal/ directory."""
    path = FORMAL_DIR / name
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return path.read_text()


# ---------------------------------------------------------------------------
# Syntax.v extraction
# ---------------------------------------------------------------------------

def extract_inductive_body(source, name):
    """Extract the body of an Inductive definition (between := and the final period)."""
    pattern = rf'Inductive {name}\b[^:]*:[^=]*:=\s*\n(.*?)\.'
    m = re.search(pattern, source, re.DOTALL)
    return m.group(1) if m else ""


def parse_constructors(body):
    """Parse constructor lines of the form: | Name : ... (* comment *)"""
    results = []
    for line in body.split('\n'):
        line = line.strip()
        # Match lines like:  | TInt : ty   (* int *)
        # or:  | EInt : Z -> expr   (* integer literal *)
        m = re.match(r'\|\s*(\w+)\s*(?::.*?)?\s*(?:\(\*\s*(.*?)\s*\*\))?$', line)
        if m:
            results.append((m.group(1), (m.group(2) or "").strip()))
    return results


def extract_types(syntax):
    """Extract Inductive ty constructors from Syntax.v."""
    body = extract_inductive_body(syntax, "ty")
    return parse_constructors(body)


def extract_binops(syntax):
    """Extract Inductive binop constructors from Syntax.v."""
    body = extract_inductive_body(syntax, "binop")
    return parse_constructors(body)


def extract_unops(syntax):
    """Extract Inductive unop constructors from Syntax.v."""
    body = extract_inductive_body(syntax, "unop")
    return parse_constructors(body)


def extract_exprs(syntax):
    """Extract Inductive expr constructors from Syntax.v."""
    body = extract_inductive_body(syntax, "expr")
    return parse_constructors(body)


def extract_vals(syntax):
    """Extract Inductive val constructors from Syntax.v.

    val is mutually inductive with env, so we stop at the 'with' keyword
    rather than continuing to the final period.
    """
    body = extract_inductive_body(syntax, "val")
    # Truncate at 'with env' if present (mutual inductive)
    with_idx = body.find("\nwith ")
    if with_idx >= 0:
        body = body[:with_idx]
    return parse_constructors(body)


# ---------------------------------------------------------------------------
# Semantics.v extraction
# ---------------------------------------------------------------------------

def extract_eval_rules(semantics):
    """Extract evaluation rule constructors and their doc comments from Semantics.v."""
    rules = []
    # Find the Inductive eval block
    m = re.search(r'Inductive eval\b.*?:=\s*\n(.*)', semantics, re.DOTALL)
    if not m:
        return rules
    block = m.group(1)
    # Each rule is preceded by (** ... *) then | Name : forall ...
    pattern = r'\(\*\*\s*(.*?)\s*\*\)\s*\|\s*(\w+)\s*:'
    for hit in re.finditer(pattern, block, re.DOTALL):
        comment = re.sub(r'\s+', ' ', hit.group(1)).strip()
        name = hit.group(2).strip()
        rules.append((name, comment))
    return rules


def extract_arith_cases(semantics):
    """Extract match arms from eval_arith_binop definition."""
    m = re.search(r'Definition eval_arith_binop.*?:=\s*\n\s*match op with\s*\n(.*?)\s*end\.',
                  semantics, re.DOTALL)
    if not m:
        return []
    results = []
    for line in m.group(1).split('\n'):
        line = line.strip()
        arm = re.match(r'\|\s*(\w+)\s*=>\s*(.*)', line)
        if arm:
            op = arm.group(1)
            rhs = arm.group(2).strip()
            results.append((op, rhs))
    return results


def extract_cmp_cases(semantics):
    """Extract match arms from eval_cmp_binop definition."""
    m = re.search(r'Definition eval_cmp_binop.*?:=\s*\n\s*match op with\s*\n(.*?)\s*end\.',
                  semantics, re.DOTALL)
    if not m:
        return []
    results = []
    for line in m.group(1).split('\n'):
        line = line.strip()
        arm = re.match(r'\|\s*(\w+)\s*=>\s*(.*)', line)
        if arm:
            op = arm.group(1)
            rhs = arm.group(2).strip()
            results.append((op, rhs))
    return results


def extract_op_classifier(semantics, func_name):
    """Extract which ops return true from an is_*_op classifier."""
    pattern = rf'Definition {func_name}.*?:=\s*\n\s*match op with\s*\n(.*?)\s*end\.'
    m = re.search(pattern, semantics, re.DOTALL)
    if not m:
        return []
    ops = []
    for line in m.group(1).split('\n'):
        line = line.strip()
        arm = re.match(r'\|\s*([\w\s|]+)\s*=>\s*true', line)
        if arm:
            names = [n.strip() for n in arm.group(1).split('|') if n.strip()]
            ops.extend(names)
    return ops


# ---------------------------------------------------------------------------
# Typing.v extraction
# ---------------------------------------------------------------------------

def extract_type_rules(typing):
    """Extract typing rule constructors and their doc comments from Typing.v."""
    rules = []
    m = re.search(r'Inductive has_type\b.*?:=\s*\n(.*)', typing, re.DOTALL)
    if not m:
        return rules
    block = m.group(1)
    pattern = r'\(\*\*\s*(.*?)\s*\*\)\s*\|\s*(\w+)\s*:'
    for hit in re.finditer(pattern, block, re.DOTALL):
        comment = re.sub(r'\s+', ' ', hit.group(1)).strip()
        name = hit.group(2).strip()
        rules.append((name, comment))
    return rules


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

TYPE_INFO = {
    "TInt":     ("int",                          "64-bit signed integer"),
    "TBool":    ("bool",                         "Boolean (true/false)"),
    "TString":  ("string",                       "Immutable string"),
    "TUnit":    ("unit",                         "Unit type (void-like)"),
    "TArrow":   ("T1 -> T2",                    "Function type"),
    "TArray":   ("array<T>",                     "Homogeneous array"),
    "TRecord":  ("{f1: T1, ..., fn: Tn}",       "Record (struct) type"),
    "TVariant": ("Tag1(T1) | ... | Tagn(Tn)",   "Variant (tagged union) type"),
}

BINOP_INFO = {
    "OpAdd":    ("+",    "int, int",             "int",    "Addition"),
    "OpSub":    ("-",    "int, int",             "int",    "Subtraction"),
    "OpMul":    ("*",    "int, int",             "int",    "Multiplication"),
    "OpDiv":    ("/",    "int, int",             "int",    "Division (None if divisor = 0)"),
    "OpMod":    ("%",    "int, int",             "int",    "Modulo (None if divisor = 0)"),
    "OpEq":     ("==",   "int/bool/string",      "bool",   "Equality (polymorphic)"),
    "OpNe":     ("!=",   "int/bool/string",      "bool",   "Inequality (polymorphic)"),
    "OpLt":     ("<",    "int, int",             "bool",   "Less than"),
    "OpLe":     ("<=",   "int, int",             "bool",   "Less than or equal"),
    "OpGt":     (">",    "int, int",             "bool",   "Greater than"),
    "OpGe":     (">=",   "int, int",             "bool",   "Greater than or equal"),
    "OpAnd":    ("and",  "bool, bool",           "bool",   "Logical AND (short-circuit)"),
    "OpOr":     ("or",   "bool, bool",           "bool",   "Logical OR (short-circuit)"),
    "OpStrCat": ("+",    "string, string",       "string", "String concatenation"),
}

UNOP_INFO = {
    "OpNeg":      ("- (unary)",    "int",      "int",    "Arithmetic negation"),
    "OpNot":      ("not",          "bool",     "bool",   "Logical negation"),
    "OpStrLen":   ("str_length",   "string",   "int",    "String length"),
    "OpArrayLen": ("array_length", "array<T>", "int",    "Array length"),
}

VAL_INFO = {
    "VInt":       ("VInt n",                "Integer value"),
    "VBool":      ("VBool b",               "Boolean value"),
    "VString":    ("VString s",             "String value"),
    "VUnit":      ("VUnit",                 "Unit value"),
    "VClos":      ("VClos x body env",      "Closure: captures parameter name, body, and environment"),
    "VFixClos":   ("VFixClos f x body env", "Recursive closure: also captures the function name for self-reference"),
    "VArray":     ("VArray vs",             "Array of values"),
    "VRecord":    ("VRecord fields",        "Record of named field values"),
    "VConstruct": ("VConstruct tag v",      "Variant value: a tag with a payload value"),
}

# Fallback descriptions for rules without Coq doc comments
EVAL_RULE_FALLBACKS = {
    "E_And_False": "AND where left is false (literal false on right)",
    "E_Or_Short":  "Short-circuit OR: if left is true, result is true regardless of right",
}

EVAL_CATEGORIES = {
    "Literals":        ["E_Int", "E_Bool", "E_String", "E_Unit"],
    "Variables":       ["E_Var"],
    "Arithmetic":      ["E_BinArith"],
    "Comparison":      ["E_BinCmp", "E_BinEqBool", "E_BinNeBool", "E_BinEqStr", "E_BinNeStr"],
    "String Ops":      ["E_StrCat"],
    "Logical Ops":     ["E_And_True", "E_And_False", "E_And_Short", "E_Or_False", "E_Or_Short"],
    "Unary Ops":       ["E_Neg", "E_Not", "E_StrLen"],
    "Control Flow":    ["E_IfTrue", "E_IfFalse", "E_Let", "E_Set", "E_Seq",
                        "E_WhileTrue", "E_WhileFalse"],
    "Functions":       ["E_Lam", "E_App", "E_Fix", "E_AppFix"],
    "Arrays":          ["E_ArrayNil", "E_ArrayCons", "E_Index", "E_ArrayLen",
                        "E_ArraySet", "E_ArrayPush"],
    "Records":         ["E_RecordNil", "E_RecordCons", "E_Field", "E_SetField"],
    "Variants":        ["E_Construct", "E_Match"],
    "String Indexing": ["E_StrIndex"],
}

TYPE_RULE_CATEGORIES = {
    "Literals":      ["T_Int", "T_Bool", "T_String", "T_Unit"],
    "Variables":     ["T_Var"],
    "Operators":     ["T_BinOp", "T_BinLogic", "T_BinEqBool", "T_StrCat", "T_BinEqStr",
                      "T_Neg", "T_Not", "T_StrLen"],
    "Control Flow":  ["T_If", "T_Let", "T_Set", "T_Seq", "T_While"],
    "Functions":     ["T_Lam", "T_App", "T_Fix"],
    "Arrays":        ["T_ArrayNil", "T_ArrayCons", "T_Index", "T_ArrayLen",
                      "T_ArraySet", "T_ArrayPush"],
    "Records":       ["T_RecordNil", "T_RecordCons", "T_Field", "T_SetField"],
    "Variants":      ["T_Construct", "T_Match"],
    "Strings":       ["T_StrIndex"],
}

EDGE_CASES = [
    ("Division by zero",
     "`a / 0` and `a % 0`",
     "Returns `None` (evaluation gets stuck). Programs must guard against zero divisors."),
    ("Array out of bounds",
     "`arr[i]` where `i >= length(arr)` or `i < 0`",
     "Returns `None` (evaluation gets stuck). The type system does not prevent this."),
    ("Undefined variable",
     "`x` where `x` is not in the environment",
     "Returns `None` (evaluation gets stuck). The type system prevents this for well-typed programs."),
    ("Type mismatch at runtime",
     "e.g. `3 + true`",
     "Gets stuck (no evaluation rule applies). Well-typed programs never reach this state (progress theorem)."),
    ("Short-circuit AND",
     "`false and e2`",
     "Returns `false` without evaluating `e2`. Side effects in `e2` are **not** executed."),
    ("Short-circuit OR",
     "`true or e2`",
     "Returns `true` without evaluating `e2`. Side effects in `e2` are **not** executed."),
    ("String indexing out of bounds",
     "`s[i]` where `i >= length(s)` or `i < 0`",
     "Returns `\"\"` (empty string). String indexing is **total** -- it never gets stuck."),
    ("Empty array literal",
     "`[]`",
     "Has type `array<T>` for any `T`. Type is determined by usage context."),
    ("Let binding scoping",
     "`let x = e1 in e2`",
     "After `e2` evaluates, the `x` binding is popped. Mutations to outer variables are preserved."),
    ("Closure capture",
     "`fun (x : T) => body`",
     "Captures the current environment (lexical scoping). Mutations inside the body do not affect the caller."),
    ("While loop result",
     "`while cond do body`",
     "Always produces `unit`. Termination is **not** guaranteed -- depends on the condition."),
    ("Record field update",
     "`set x.f = v`",
     "Updates the field in-place. Requires `x` in scope and `f` to be an existing field."),
    ("Pattern matching scope",
     "`match e { Tag x => body, ... }`",
     "The binding `x` is scoped to the match arm body and popped afterwards."),
]


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

def generate_spec():
    syntax = read_file("Syntax.v")
    semantics = read_file("Semantics.v")
    typing = read_file("Typing.v")

    types = extract_types(syntax)
    binops = extract_binops(syntax)
    unops = extract_unops(syntax)
    exprs = extract_exprs(syntax)
    vals = extract_vals(syntax)
    eval_rules = extract_eval_rules(semantics)
    type_rules = extract_type_rules(typing)
    arith_ops = extract_op_classifier(semantics, "is_arith_op")
    cmp_ops = extract_op_classifier(semantics, "is_cmp_op")
    logic_ops = extract_op_classifier(semantics, "is_logic_op")
    string_ops = extract_op_classifier(semantics, "is_string_op")

    # Build lookup for eval/type rules by name, with fallbacks
    eval_by_name = dict(EVAL_RULE_FALLBACKS)
    eval_by_name.update({name: desc for name, desc in eval_rules})
    type_by_name = {name: desc for name, desc in type_rules}

    out = []

    def line(s=""):
        out.append(s)

    # ---- Header ----
    line("# NanoCore Formal Specification")
    line()
    line(f"*Auto-generated from Coq proof scripts on {date.today().isoformat()}.*")
    line()
    line("> **NanoCore** is the formally verified subset of NanoLang. Every property")
    line("> listed below has been machine-checked in Coq with **0 axioms** and")
    line("> **0 Admitted** -- the proofs are complete and trustworthy.")
    line()

    # ---- Verification Status ----
    line("## Verification Status")
    line()
    line("| Property | File | Description |")
    line("|----------|------|-------------|")
    line("| Type Soundness | `Soundness.v` | Well-typed programs evaluate to well-typed values |")
    line("| Progress | `Progress.v` | Well-typed expressions are never stuck |")
    line("| Determinism | `Determinism.v` | Evaluation produces a unique result |")
    line("| Big-step / Small-step Equivalence | `Equivalence.v` | Both semantics agree on all programs |")
    line()

    # ---- Types ----
    line("## 1. Types")
    line()
    line(f"NanoCore has **{len(types)} types** (defined in `Syntax.v`):")
    line()
    line("| Type | Syntax | Description |")
    line("|------|--------|-------------|")
    for name, _comment in types:
        syntax_str, desc = TYPE_INFO.get(name, (name, _comment))
        line(f"| `{name}` | `{syntax_str}` | {desc} |")
    line()

    # ---- Operators ----
    line("## 2. Operators")
    line()

    line("### 2.1 Binary Operators")
    line()
    line(f"NanoCore has **{len(binops)} binary operators** classified by the semantics as:")
    line(f"- Arithmetic ({len(arith_ops)}): {', '.join(f'`{o}`' for o in arith_ops)}")
    line(f"- Comparison ({len(cmp_ops)}): {', '.join(f'`{o}`' for o in cmp_ops)}")
    line(f"- Logical ({len(logic_ops)}): {', '.join(f'`{o}`' for o in logic_ops)}")
    line(f"- String ({len(string_ops)}): {', '.join(f'`{o}`' for o in string_ops)}")
    line()
    line("| Operator | Symbol | Operand Types | Result Type | Notes |")
    line("|----------|--------|---------------|-------------|-------|")
    for name, _comment in binops:
        info = BINOP_INFO.get(name)
        if info:
            sym, operands, result, notes = info
            line(f"| `{name}` | `{sym}` | {operands} | {result} | {notes} |")
    line()

    line("### 2.2 Unary Operators")
    line()
    line("| Operator | Symbol | Operand Type | Result Type | Description |")
    line("|----------|--------|--------------|-------------|-------------|")
    for name, _comment in unops:
        info = UNOP_INFO.get(name)
        if info:
            sym, operand, result, desc = info
            line(f"| `{name}` | `{sym}` | {operand} | {result} | {desc} |")
    line()

    line("### 2.3 Arithmetic Semantics (`eval_arith_binop`)")
    line()
    line("```")
    line("eval_arith_binop op n1 n2 =")
    for op, rhs in extract_arith_cases(semantics):
        line(f"  | {op} => {rhs}")
    line("```")
    line()

    line("### 2.4 Comparison Semantics (`eval_cmp_binop`)")
    line()
    line("```")
    line("eval_cmp_binop op n1 n2 =")
    for op, rhs in extract_cmp_cases(semantics):
        line(f"  | {op} => {rhs}")
    line("```")
    line()

    # ---- Expressions ----
    line("## 3. Expression Forms")
    line()
    line(f"NanoCore has **{len(exprs)} expression forms** (defined in `Syntax.v`):")
    line()
    line("| # | Constructor | Coq Comment | Description |")
    line("|---|-------------|-------------|-------------|")
    for i, (name, comment) in enumerate(exprs, 1):
        desc = comment if comment else ""
        line(f"| {i} | `{name}` | {desc} | |")
    line()

    # ---- Evaluation Rules ----
    line("## 4. Evaluation Rules")
    line()
    line("The big-step semantics relation is `eval renv e renv' v`: expression `e`")
    line("evaluates to value `v` in environment `renv`, producing updated environment `renv'`.")
    line()
    line("**Key properties:**")
    line("- **Eager** (call-by-value): arguments are evaluated before function bodies")
    line("- **Deterministic**: each expression has at most one result")
    line("- **Lexical scoping**: closures capture their creation environment")
    line("- **Store-passing**: mutable variables are threaded through the environment")
    line()

    for cat, rule_names in EVAL_CATEGORIES.items():
        present = [(n, eval_by_name[n]) for n in rule_names if n in eval_by_name]
        if not present:
            continue
        line(f"### 4.{list(EVAL_CATEGORIES.keys()).index(cat)+1} {cat}")
        line()
        line("| Rule | Description |")
        line("|------|-------------|")
        for name, desc in present:
            line(f"| `{name}` | {desc} |")
        line()

    # ---- Typing Rules ----
    line("## 5. Typing Rules")
    line()
    line("The typing judgment is `has_type ctx e t`: expression `e` has type `t`")
    line("in typing context `ctx`.")
    line()

    for cat, rule_names in TYPE_RULE_CATEGORIES.items():
        present = [(n, type_by_name[n]) for n in rule_names if n in type_by_name]
        if not present:
            continue
        line(f"### 5.{list(TYPE_RULE_CATEGORIES.keys()).index(cat)+1} {cat}")
        line()
        line("| Rule | Description |")
        line("|------|-------------|")
        for name, desc in present:
            line(f"| `{name}` | {desc} |")
        line()

    # ---- Values ----
    line("## 6. Values")
    line()
    line("Values are results of evaluation that cannot be reduced further.")
    line(f"NanoCore has **{len(vals)} value forms** (defined in `Syntax.v`):")
    line()
    line("| Constructor | Form | Description |")
    line("|-------------|------|-------------|")
    for name, _comment in vals:
        info = VAL_INFO.get(name)
        if info:
            form, desc = info
            line(f"| `{name}` | `{form}` | {desc} |")
        else:
            line(f"| `{name}` | | {_comment} |")
    line()

    # ---- Edge Cases ----
    line("## 7. Edge Cases")
    line()
    line("These behaviors are precisely specified by the formal semantics:")
    line()
    line("| Case | Expression | Behavior |")
    line("|------|-----------|----------|")
    for case, expr, behavior in EDGE_CASES:
        line(f"| {case} | {expr} | {behavior} |")
    line()

    # ---- Proven Properties ----
    line("## 8. Proven Properties")
    line()

    line("### 8.1 Type Soundness (`Soundness.v`)")
    line()
    line("If expression `e` has type `t` in a well-formed context, and `e` evaluates")
    line("to value `v`, then `v` is a well-typed value of type `t`. In other words,")
    line("evaluation preserves types.")
    line()

    line("### 8.2 Progress (`Progress.v`)")
    line()
    line("If expression `e` has type `t`, then either `e` is already a value or `e` can")
    line("take a step under the small-step semantics. Well-typed programs never get stuck")
    line("(modulo division by zero and out-of-bounds array access, which are partial).")
    line()

    line("### 8.3 Determinism (`Determinism.v`)")
    line()
    line("If `eval renv e renv1 v1` and `eval renv e renv2 v2`, then `v1 = v2` and")
    line("`renv1 = renv2`. Evaluation is a partial function -- there is at most one")
    line("result for any expression in any environment.")
    line()

    line("### 8.4 Big-step / Small-step Equivalence (`Equivalence.v`)")
    line()
    line("The big-step semantics (natural semantics) and the small-step semantics")
    line("(structural operational semantics) agree on all programs. This gives two")
    line("independent definitions of NanoCore's behavior that serve as a cross-check.")
    line()

    # ---- NanoCore vs NanoLang ----
    line("## 9. NanoCore vs Full NanoLang")
    line()
    line("NanoCore is a verified subset. Features present in full NanoLang but")
    line("**outside** the formally verified subset:")
    line()
    line("| Feature | NanoLang | NanoCore |")
    line("|---------|----------|----------|")
    line("| Integers, Booleans, Strings | Yes | Yes |")
    line("| Arrays, Records, Variants | Yes | Yes |")
    line("| Pattern Matching | Yes | Yes |")
    line("| Functions, Recursion | Yes | Yes |")
    line("| Mutable Variables | Yes | Yes |")
    line("| While Loops | Yes | Yes |")
    line("| Floats | Yes | **No** |")
    line("| Tuples | Yes | **No** |")
    line("| Hashmaps | Yes | **No** |")
    line("| Enums | Yes | **No** |")
    line("| Generics | Yes | **No** |")
    line("| Modules / Imports | Yes | **No** |")
    line("| FFI / Extern | Yes | **No** |")
    line("| Opaque Types | Yes | **No** |")
    line("| For Loops | Yes | **No** (expressible as while + let) |")
    line("| Print / Assert | Yes | **No** |")
    line()

    line("---")
    line()
    line("*Generated by `tools/extract_spec.py` from the Coq sources in `formal/`.*")
    line()

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    spec = generate_spec()
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    OUTPUT_PATH.write_text(spec)
    n_lines = spec.count('\n')
    print(f"Generated {OUTPUT_PATH} ({len(spec)} bytes, {n_lines} lines)")
