# Passive Parallelism in Nanolang: Language Design Extensions

**Status:** Design Proposal  
**Author:** Rocky (do-host1)  
**Date:** 2026-03-26  

---

## Philosophy

The goal is not to give programmers *more ways to ask for parallelism*. Fork/join, async/await, `spawn`, `go` — these are all mechanisms that require the programmer to decompose their problem and manage concurrency explicitly. The cognitive overhead is high, and the failure modes (races, deadlocks, missed synchronization) are severe.

The goal is to make parallelism **the path of least resistance** — to design language constructs such that the natural way to write code is also the automatically-parallel way to execute it. The programmer expresses *what* they want; the runtime figures out *how* to compute it concurrently.

Nanolang is already close. This document identifies the specific language-level changes that would get it the rest of the way.

---

## Foundation: What We Already Have

Nanolang's existing design provides several building blocks:

1. **Immutable-by-default** (`let` vs `mut`) — the type system already separates mutable from immutable state
2. **Affine types for resources** — resources can only be owned by one computation at a time (in progress)
3. **Effect system** (planned) — `@pure` annotations distinguish computations with no side effects
4. **Typed homogeneous arrays** — contiguous memory, known element types (enables SIMD, vectorization)
5. **First-class functions** — `map`, `filter`, `fold` are already built-in
6. **Shadow tests** — mandatory correctness tests enable the compiler to be more aggressive

What's missing is the **connective tissue** — the language constructs that let the compiler observe data dependencies, eliminate false sharing, and schedule parallel execution automatically.

---

## Extension 1: Pure Expression Blocks (`par { ... }`)

### Problem

Currently there is no way to express that multiple expressions can be evaluated in any order. The programmer writes them sequentially and the compiler evaluates them sequentially.

### Design

Introduce `par { ... }` — a block of expressions where **the programmer asserts no ordering dependency** between the statements. The compiler may evaluate them in parallel, in any order, or even speculatively.

```nano
# Current: sequential, arbitrary order imposed
let a: float = expensive_a()
let b: float = expensive_b()
let c: float = expensive_c()
let result: float = a + b + c

# With par: explicitly unordered, compiler parallelizes
par {
    let a: float = expensive_a()
    let b: float = expensive_b()
    let c: float = expensive_c()
}
let result: float = a + b + c
```

**Key property:** A `par` block is only valid when:
- All expressions inside are `@pure` (or the effect system can verify no shared mutable state)
- No expression inside depends on the result of another expression inside the same block
- The type checker verifies these statically; a `par` block with a dependency inside is a compile error

This is NOT fork/join. `par` is not about spawning threads — it's a declaration of independence. The compiler may use SIMD, thread pools, out-of-order execution, or nothing at all (if the hardware doesn't benefit). The programmer just says "these don't depend on each other."

### Why This Is Different From Fork/Join

Fork/join: programmer creates tasks, waits for them  
`par { }`: programmer declares independence, runtime decides execution strategy

The programmer expresses *data independence*, not *execution management*.

---

## Extension 2: Parallel Collection Operations With Effect Typing

### Problem

`map`, `filter`, and `fold` are already built-in, but there's no way to express that they can run in parallel without the programmer annotating anything. The compiler can't safely parallelize them today because it can't prove the transform function has no side effects.

### Design

Integrate the effect system (already planned) with the collection builtins. When the transform function is `@pure`, the compiler *automatically* parallelizes the operation. No annotation from the programmer:

```nano
# The @pure annotation on the function is the only thing needed
fn double(x: float) -> float @pure {
    return x * 2.0
}

# This map is automatically parallel — no programmer action required
# The compiler sees @pure and knows it's safe to chunk the array and parallelize
let result: array<float> = map(data, double)
```

For inline lambdas, purity is inferred:

```nano
# Lambda has no external references, no I/O — compiler infers @pure
# Automatically eligible for parallel execution
let result: array<float> = map(data, fn(x: float) -> float { x * 2.0 })
```

For `fold`, parallelism requires the combine operation to be **associative**. The compiler can verify this for the built-in arithmetic ops (`+`, `*`, `min`, `max`) but not for arbitrary functions. Add an `@associative` annotation for user-defined combine functions:

```nano
fn combine(a: float, b: float) -> float @pure @associative {
    return a + b  # order doesn't matter
}

# Parallel reduction — O(log n) depth with tree reduction
let total: float = fold(data, 0.0, combine)
```

The `@associative` annotation is verified at compile time using shadow tests (already mandatory in nanolang). The shadow tests must include a commutativity/associativity test:

```nano
shadow combine {
    # Compiler enforces: must test associativity
    assert combine(combine(1.0, 2.0), 3.0) == combine(1.0, combine(2.0, 3.0))
}
```

This turns mandatory shadow tests into **parallel safety proofs** — a uniquely nanolang design.

---

## Extension 3: Dataflow Bindings (`flow let`)

### Problem

The biggest source of false sequential dependencies is assignment order. When you write:

```nano
let a: float = f(x)
let b: float = g(x)
let c: float = h(a, b)
```

The compiler evaluates `f` and `g` sequentially, even though they're independent. The dependency of `c` on `a` and `b` is explicit, but the independence of `a` and `b` from each other is implicit and the compiler may not exploit it.

### Design

Introduce `flow let` — bindings that declare themselves as part of a dataflow graph. The compiler builds the dependency graph from the uses and evaluates nodes as soon as their inputs are available:

```nano
flow {
    let a: float = f(x)          # depends on: x
    let b: float = g(x)          # depends on: x  (independent of a!)
    let c: float = h(a, b)       # depends on: a, b
    let d: float = k(c)          # depends on: c
}
# a and b run concurrently
# c runs when both a and b are done
# d runs when c is done
# Critical path: max(f,g) + h + k, not f + g + h + k
```

The dataflow graph is **derived from data dependencies**, not from programmer-specified tasks. The programmer just writes the natural computation; the compiler observes the dependency structure.

**Relationship to `mut`:** In a `flow` block, `mut` variables are synchronization points — reads and writes are coordinated by the runtime scheduler. This is the mutability-as-concurrency model (see `MUTABILITY_AS_CONCURRENCY.md`) applied at fine granularity inside a single function.

---

## Extension 4: Struct-of-Arrays Types (`soa struct`)

### Problem

Arrays of structs (`array<Point>`) store data in AoS (Array-of-Structures) layout: `x0 y0 x1 y1 x2 y2 ...`. This is SIMD-unfriendly — accessing all `x` values requires strided loads.

For any loop that processes one field across all elements (`for p in points { p.x * scale }`), SoA (Structure-of-Arrays) layout is 4–8× faster on AVX2 hardware.

### Design

Introduce `soa struct` — a struct declaration that instructs the compiler to store arrays of this type in SoA layout:

```nano
# Standard struct (AoS when used in arrays)
struct Point {
    x: float,
    y: float
}

# SoA struct (when used as array<SoaPoint>, stored as [x0,x1,...][y0,y1,...])
soa struct SoaPoint {
    x: float,
    y: float
}

let points: array<SoaPoint> = [...]

# This loop accesses contiguous x values — SIMD-friendly
for p in points {
    p.x = p.x * scale  # accesses contiguous float memory
}
```

The programmer chooses `soa struct` when they expect to process fields column-wise (all `x` values, then all `y` values). The compiler handles the layout transformation transparently.

**Auto-detection opportunity:** The compiler can *suggest* `soa struct` when it detects the dominant access pattern in loops is field-wise. Shadow tests provide the usage data for this analysis.

---

## Extension 5: Async I/O Without Callbacks (`await` in effect system)

### Problem

Nanolang currently has no async I/O model. I/O operations block, and there's no way to overlap I/O with computation.

### Design

Extend the effect system with an `@async` effect. Functions with `@async` return immediately with a handle; the result is available when the computation completes. The `await` keyword extracts the result, blocking only if necessary:

```nano
fn read_file(path: string) -> string @async @io {
    # Non-blocking: returns handle immediately
}

fn process_all(paths: array<string>) -> array<string> {
    # Fire all reads concurrently — no blocking
    let handles: array<Future<string>> = map(paths, read_file)
    
    # Block only when results are needed
    let contents: array<string> = map(handles, await)
    return map(contents, process)
}
```

The key design choice: `await` is a **regular function**, not special syntax. This means it composes naturally with `map`, `filter`, and `fold`. The pattern `map(handles, await)` is idiomatic nanolang — it maps a pure transformation over a collection — but it also efficiently awaits all futures concurrently.

This is structurally identical to JavaScript's `Promise.all` but with no callback pyramid and no special async/await syntax beyond the effect annotation.

---

## Extension 6: Immutable Shared Memory via `frozen`

### Problem

Today, if you want to share data across concurrent computations, you need mutable cells (with coordination cost). But read-only data needs no coordination at all — any number of readers can access it simultaneously without synchronization.

There's no language-level way to express "this large data structure is frozen — share it freely."

### Design

Add a `frozen` qualifier. A `frozen` value:
- Is deeply immutable (no field can be mutated, even transitively)
- Can be freely shared across concurrent computations
- Requires zero coordination on access
- Can be passed to `@pure` functions by reference without copying

```nano
frozen let lookup_table: array<float> = build_lookup_table()

# All workers can read lookup_table concurrently — zero synchronization cost
let results: array<float> = map(inputs, fn(x: float) -> float @pure {
    return lookup_table[hash(x)]  # safe: lookup_table is frozen
})
```

The type checker enforces that `frozen` values are never passed to functions expecting `mut` parameters. The compiler can generate uncoordinated reads for `frozen` access paths.

This is particularly important for LLM-generated code: large constant tables (embeddings, lookup tables, model weights) are naturally `frozen`, and expressing that enables the runtime to distribute them across workers without defensive copying.

---

## Extension 7: Pipeline Operator With Automatic Fusion

### Problem

The existing `|>` pipe operator (in planning) chains transformations sequentially. In parallel, this creates unnecessary intermediate arrays:

```nano
let result: array<float> = data
    |> map(fn(x) { x * 2.0 })     # creates intermediate array
    |> filter(fn(x) { x > 0.0 })  # creates another intermediate
    |> fold(0.0, add)              # reduces to scalar
```

Each step materializes a full intermediate collection.

### Design

The pipe operator should support **automatic fusion** when all stages are `@pure`:

```nano
# Compiler fuses all three into a single pass over the data
# No intermediate arrays allocated
# Single loop: for each element, multiply, check, accumulate
let result: float = data
    |> map @pure (fn(x) { x * 2.0 })
    |> filter @pure (fn(x) { x > 0.0 })
    |> fold(0.0, add)
```

The compiler recognizes a chain of pure map/filter/fold operations and fuses them into a single loop:
- No intermediate allocation
- Single pass through the data (cache-friendly)
- The fused loop is SIMD-eligible (it's a single pure computation per element)

This is **loop fusion** — a classical compiler optimization — but expressed at the language level through the effect system. The `@pure` annotations on each stage are the enabling condition.

---

## Implementation Priority

| Extension | Parallelism Type | Effort | Leverage Existing |
|---|---|---|---|
| `par { }` blocks | Task-level | Medium | Effect system (planned) |
| `@pure` auto-parallel map/filter | Data-parallel | Low | Effect system + builtins |
| `@associative` fold | Reduction | Low | Shadow tests |
| `flow let` dataflow bindings | Dataflow | High | `mut` model |
| `soa struct` | SIMD / memory layout | Medium | DynArray |
| `@async` effect + `await` | I/O overlap | High | Effect system |
| `frozen` qualifier | Shared read-only | Low | Type checker |
| Pipeline fusion `|>` | Loop fusion | Medium | Effect system |

**Recommended order:**
1. `frozen` — pure type system addition, no runtime changes, immediately useful
2. `@pure` auto-parallel map/filter — leverages planned effect system, high payoff
3. `@associative` for fold — shadow tests do the proof work, low cost
4. `par { }` blocks — explicit independence annotation, clean semantics
5. Pipeline fusion via `|>` — builds on `@pure`, eliminates allocation waste
6. `soa struct` — biggest SIMD payoff for numeric code
7. `flow let` — most general, highest implementation complexity
8. `@async` effect — needs runtime scheduler changes

---

## Relationship to Existing Plans

- **Effect system** (`docs/features/EFFECT_SYSTEM_PLAN.md`): Extensions 1–3, 5, 7 all depend on `@pure`. The effect system is the load-bearing wall. **Prioritize Phase 1 (annotations) immediately.**
- **Affine types** (`docs/AFFINE_TYPES_DESIGN.md`): Resource affinity is the single-owner model; `frozen` is the zero-owner/all-readers model. They're complementary.
- **NanoISA / NanoVM**: The `par { }` and `flow let` semantics require runtime scheduler support. The VM is the natural place to implement the parallel scheduler — bytecode instructions for "fire these concurrently" can be emitted by the compiler.
- **Shadow tests**: The `@associative` extension repurposes mandatory shadow tests as parallel safety proofs. This is a uniquely nanolang design that no other language has.

---

## The Unifying Principle

All of these extensions share a single design philosophy:

> **The programmer expresses what is true about their data. The runtime exploits it.**

- `@pure` — "this computation has no side effects" → runtime can parallelize it
- `frozen` — "this data never changes" → runtime can share it without coordination  
- `@associative` — "this combination is order-independent" → runtime can do tree reduction
- `par { }` — "these expressions don't depend on each other" → runtime can interleave them
- `soa struct` — "I'll access this field-by-field" → compiler can lay it out for SIMD
- `flow let` — "here are my data dependencies" → runtime can schedule as a dataflow graph

None of these require the programmer to think about threads, locks, workers, or scheduling. The programmer thinks about **properties of their data and computations**. The system handles the rest.

This is passive parallelism — not parallelism you ask for, but parallelism the system finds automatically because the code tells the truth about itself.
