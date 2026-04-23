# Async Primitives

I support async at the language level with `async fn` and `await`.

## Model

I use cooperative concurrency. I do not preempt running code.

- `async fn` marks a function body as suspension-capable.
- `await expr` marks a suspension boundary inside an async function.
- My CPS pass validates `await` placement and preserves async boundaries.
- My coroutine scheduler is cooperative and deterministic for a fixed schedule.

## VM Boundary

My NanoISA VM already separates pure execution (`vm_core_execute`) from side effects through traps.

- Pure instructions run in `vm_core_execute`.
- External boundaries (I/O, assertions, FFI, halt) trap to the runtime harness.
- This trap split is the suspension/resumption hook for VM async scheduling work.

Today, my tested async execution path is in my compiler/runtime CPS + coroutine flow. VM async trap scheduling is designed around the existing trap contract and kept explicitly at that boundary.

## C Backend Lowering

I lower async programs into state-machine/coroutine-friendly C paths:

- async syntax is lowered during CPS/coroutine lowering
- generated C links against my coroutine runtime helpers
- module-level concurrency helpers can use libdispatch when available

## Errors Across Async Boundaries

I propagate runtime errors through normal return/error flow. I do not silently swallow async failures.

## Shadow Tests

I enforce shadow tests for async functions exactly like synchronous functions. See `tests/test_async.nano` for executable examples.

## Formal Status

I keep my proof boundary explicit:

- proved: my existing NanoCore subset in `formal/`
- tested: async syntax/type/runtime behavior in compiler and runtime tests
- planned proof work: async typing and operational soundness extensions
