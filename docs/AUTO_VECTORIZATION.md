# Auto-Vectorization in Nanolang: Opportunities and Implementation Plan

## Executive Summary

Nanolang has several strong vectorization opportunities hiding in plain sight. The language's design — typed arrays with homogeneous elements, explicit `map`/`filter`/`fold` builtins, and immutability as the default — maps cleanly onto SIMD hardware. This document identifies the concrete sites where auto-vectorization can be applied, from quick wins to deeper compiler passes.

---

## 1. Where the Opportunities Are

### 1.1 Array Arithmetic (Highest Impact, Easiest Win)

**File:** `src/eval.c` — `eval_dyn_array_binop()` and `eval_dyn_array_scalar_right/left()`

The interpreter already loops over homogeneous typed arrays to apply element-wise arithmetic. The current code:

```c
for (int64_t i = 0; i < len; i++) {
    double x = dyn_array_get_float(a, i);
    double y = dyn_array_get_float(b, i);
    r = x + y;
    dyn_array_push_float(out, r);
}
```

This is exactly what SIMD is for. The array data is contiguous in memory (the `DynArray.data` field is a raw `void*` backed by a flat allocation). The element type is known statically (`ELEM_INT`, `ELEM_FLOAT`). The operation is known at the call site.

**Fix:** Replace the scalar loop with a vectorized one using compiler intrinsics or `__attribute__((optimize("O3")))` with `restrict` pointers. For GCC/Clang, auto-vectorization will fire automatically if the loop is written cleanly:

```c
// Add restrict qualifiers to DynArray data pointers
// Then the compiler can prove no aliasing and vectorize automatically
double *restrict src_a = (double*)a->data;
double *restrict src_b = (double*)b->data;
double *restrict dst   = (double*)out->data;
for (int64_t i = 0; i < len; i++) {
    dst[i] = src_a[i] + src_b[i];  // GCC/Clang will SIMD this at -O2
}
```

This works for `ELEM_INT` (256-bit AVX2 can do 4×int64 or 8×int32 per cycle) and `ELEM_FLOAT` (4×double per cycle with AVX2).

**Cost:** Minimal — no AST or language changes needed. Just rewrite the inner loops in `eval.c` and add `restrict` to `DynArray` allocation. Compiler does the rest.

---

### 1.2 `map()` Over Homogeneous Arrays

**File:** `src/eval.c` — `builtin_map()`

Currently, `map(array, fn)` dispatches element-by-element via `call_function()`. When the transform function is a **pure arithmetic expression** (e.g., `map(arr, fn(x) { x * 2 })`), this is massively over-abstracted — each element goes through the full interpreter dispatch pipeline.

**Opportunity:** Detect **lambda-eligible** map operations at compile/typecheck time:
- Input array is homogeneous typed (all `int` or all `float`)
- Transform function is a single arithmetic expression over its argument
- No mutable captures, no side effects

When these conditions hold, the transpiler can emit a tight vectorizable C loop instead of the general `call_function` dispatch:

```c
// Generated for: map(arr, fn(x: float) { x * 2.0 })
double *restrict data = (double*)arr->data;
double *restrict out  = (double*)result->data;
for (int64_t i = 0; i < len; i++) {
    out[i] = data[i] * 2.0;   // compiler vectorizes this
}
```

**Detection heuristic in the type checker / transpiler:**
- Function body is a single `return` statement
- The return expression contains only arithmetic binary ops and the function's parameter
- No calls, no conditionals, no mutable variable access

This is a "simple arithmetic lambda" pattern. Common in numeric code; high payoff.

---

### 1.3 `for x in array { ... }` — Loop Body Analysis

**File:** `src/transpiler.c` — `AST_FOR` case

The transpiler already generates C `for` loops for `for x in array`. Currently these are scalar. When the loop body touches only the loop variable and immutable bindings, it's a candidate for vectorization.

**Static analysis pass (new):** Walk the loop body AST. If:
1. All reads are from the loop variable or immutable `let` bindings
2. The only write is to a locally-scoped accumulator or an output array
3. No function calls with unknown effects

Then emit the loop with `#pragma clang loop vectorize(enable)` and `#pragma GCC ivdep` (or equivalent SIMD intrinsics). The C compiler then vectorizes automatically.

For the **transpiler path**, this is a compiler pass over the AST before C code emission. For the **interpreter path**, this would be a JIT-style specialization — recognize the pattern at runtime and switch to a vectorized implementation.

---

### 1.4 `filter()` with Predicate Inlining

**File:** `src/eval.c` — `builtin_filter()`

Same story as `map()`. When the predicate is a simple comparison (e.g., `filter(arr, fn(x) { x > 0 })`), it can be lowered to a SIMD-friendly masked gather/conditional store pattern.

Modern AVX-512 has direct support for masked vector operations — compare a vector of values against a threshold and extract matching elements in one pass. Without AVX-512, the standard approach is a two-pass: vectorized comparison to build a mask bitmask, then scalar gather of matching elements.

---

### 1.5 Struct-of-Arrays Transformation (AoS → SoA)

**This is the deep one.**

Nanolang arrays of structs are stored as **Array of Structures (AoS)** — each struct is laid out sequentially in memory. For SIMD, **Structure of Arrays (SoA)** is almost always better: all `x` fields packed together, all `y` fields packed together, etc.

Example: `[Point]` where `Point = { x: float, y: float }` 

- **AoS** (current): `x0 y0 x1 y1 x2 y2 ...` — accessing all `x` values requires strided loads (SIMD unfriendly)
- **SoA** (optimal): `x0 x1 x2 ... y0 y1 y2 ...` — accessing all `x` values is a single contiguous SIMD load

**When to apply it:** When an array of structs is used in a loop that accesses the same field on every element. The type checker can detect this pattern.

**Implementation cost:** High — requires a new internal representation for typed struct arrays, and the transpiler needs to emit different C code. Worth a dedicated planning doc. But the payoff for numerical/graphics code (vectors, particles, boids) is 4–8× on AVX2 hardware.

---

### 1.6 Reduction Operations (`fold`/`reduce`)

**File:** `src/eval.c` — `builtin_reduce()`

Sum, min, max, and product reductions over numeric arrays are classic SIMD targets. SIMD can compute these with a parallel reduction tree — O(log n) sequential depth vs. O(n) for scalar.

GCC/Clang auto-vectorize these automatically when:
- The reduction variable is a scalar (not an array or struct)
- The combine function is one of: `+`, `*`, `min`, `max`, `and`, `or`, `xor`
- The input array is homogeneous and contiguous

The interpreter's `builtin_reduce` currently calls back into the user function per element. For pure arithmetic reductions, detect the pattern and use a direct loop.

---

## 2. Implementation Priority

| Opportunity | Impact | Effort | Notes |
|---|---|---|---|
| Array arithmetic (`eval_dyn_array_binop`) | High | Low | Just rewrite inner loops with `restrict` |
| `map()` simple lambda specialization | High | Medium | Needs lambda pattern detection |
| `for x in array` loop vectorization pragma | Medium | Low | Add `#pragma ivdep` + analysis pass |
| `filter()` predicate inlining | Medium | Medium | Similar to map() |
| `fold`/`reduce` arithmetic detection | Medium | Low | Detect builtin ops, use direct loop |
| Struct-of-Arrays transformation | Very High | High | Needs new representation + planning |

---

## 3. Transpiler Changes Needed

### 3.1 `restrict` on DynArray Allocations

In `src/runtime/dyn_array.h`, the `data` field is `void*`. The transpiler generates code that accesses this as a cast pointer. Adding `__restrict__` to the cast in generated loops tells the compiler the buffers don't alias:

```c
// Current transpiler output
for (int64_t i = 0; i < len; i++) {
    ((double*)out->data)[i] = ((double*)a->data)[i] + ((double*)b->data)[i];
}

// Vectorization-friendly output
double *__restrict__ pa = (double*)a->data;
double *__restrict__ pb = (double*)b->data;
double *__restrict__ po = (double*)out->data;
for (int64_t i = 0; i < len; i++) {
    po[i] = pa[i] + pb[i];
}
```

The `__restrict__` tells the compiler: `pa`, `pb`, `po` don't alias each other. It can then generate vectorized code without alias analysis uncertainty.

### 3.2 Pure Function Annotation

Add a purity flag to `FunctionDef` in the type checker. A function is **pure** if:
- All inputs are immutable (no `mut` parameters)
- No mutable global bindings accessed
- No I/O builtins called
- All callees are also pure (transitively)

Pure functions passed to `map`/`filter`/`fold` are candidates for inlining into the generated loop, eliminating the function call overhead entirely.

### 3.3 Loop Vectorization Pass

Add a pre-emission AST pass that walks `AST_FOR` bodies and annotates them with one of:
- `LOOP_VECTORIZABLE` — emit with `#pragma` hints
- `LOOP_PARALLEL` — safe to parallelize (no shared mutable state, per the paper)
- `LOOP_SERIAL` — cannot be parallelized (has side effects or mutable captures)

This annotation drives code generation decisions downstream.

---

## 4. Interpreter-Path Specializations

For the interpreter (not the transpiler), vectorization can be applied through **runtime specialization**:

When `builtin_map(array, fn)` is called:
1. Check if `array` is `ELEM_INT` or `ELEM_FLOAT` (homogeneous numeric)
2. Check if `fn` resolves to a function with a single arithmetic-expression body
3. If both: JIT-compile a specialized loop using computed element type and operation
4. Otherwise: fall through to the general dispatch

This is a lightweight form of polymorphic inline caching — cache the specialized loop keyed by `(element_type, operation)` and reuse it on subsequent calls with the same types.

---

## 5. Relationship to the Mutability-as-Concurrency Model

The vectorization opportunities identified here are exactly the cases where the mutability-as-concurrency model (see `docs/MUTABILITY_AS_CONCURRENCY.md`) provides the strongest guarantees:

- **Immutable array + pure function** → trivially data-parallel, SIMD and multi-core simultaneously
- **Mutable cell** → synchronization required, SIMD within the critical section only
- **No mutable captures** → loop body is an independent computation over each element

The effect system that proves concurrency safety (the mutation effect set) also proves SIMD legality. They are the same analysis. A loop body that touches no mutable cells is both:
1. Safe to vectorize across elements (SIMD)
2. Safe to parallelize across iterations (multi-core)

This means a single static analysis pass — the mutation effect set computation — gates both the vectorization pass and the parallelization pass. The compiler needs to do this work once.

---

## 6. Quick Start: Immediate Changes

To get vectorization today with minimal code changes:

1. **`src/runtime/dyn_array.h`**: Align `data` allocation to 32 bytes (AVX2 requirement):
   ```c
   void *data = aligned_alloc(32, capacity * elem_size);
   ```

2. **`src/eval.c`, `eval_dyn_array_binop()`**: Rewrite as separate typed functions with `restrict` pointers and direct casts. Let GCC/Clang auto-vectorize.

3. **`Makefile`**: Add `-O3 -march=native -ftree-vectorize -fopt-info-vec` to CFLAGS. The `-fopt-info-vec` flag reports which loops were vectorized — use it to verify the changes are working.

4. **`src/transpiler.c`**: For `AST_FOR` loops over arrays of numeric type, add:
   ```c
   sb_append(sb, "#pragma clang loop vectorize(enable)\n");
   sb_append(sb, "#pragma GCC ivdep\n");
   ```
   before the emitted `for` loop.

These four changes can be made in an afternoon and will produce measurable speedups on any code that operates on numeric arrays. Everything else in this document is the deeper follow-through.
