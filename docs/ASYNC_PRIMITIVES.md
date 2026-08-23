# Async Primitives

I parse and interpret `async fn` and `await`. My current compiled lowering is
synchronous compatibility, not a suspension-capable state machine.

## Model

I use cooperative concurrency. I do not preempt running code.

- `async fn` marks a function for async-aware checking and interpreter dispatch.
- `await expr` is valid syntax inside an async function.
- My CPS pass currently validates and counts async constructs; it does not yet
  rewrite them into continuations.
- My coroutine scheduler has tested queue and result mechanics, but generated
  code does not yet suspend and resume at `await`.

## VM Boundary

My NanoISA VM already separates pure execution (`vm_core_execute`) from side effects through traps.

- Pure instructions run in `vm_core_execute`.
- External boundaries (I/O, assertions, FFI, halt) trap to the runtime harness.
- This trap split is the suspension/resumption hook for VM async scheduling work.

Today, I have tested parser/typechecker acceptance, synchronous interpreter
behavior, and scheduler mechanics separately. NanoISA has no async suspension
trap yet.

## C Backend Lowering

My C backends currently lower `await expr` as `expr`. They do not generate a
continuation or state machine. Module-level concurrency helpers can use
libdispatch where available, but that is a separate facility.

## Errors Across Async Boundaries

Synchronous error flow remains ordinary return/error flow. Error propagation
across a real suspension boundary is not implemented yet.

## Shadow Tests

See `tests/test_async.nano` for syntax and synchronous compatibility examples.
Those tests do not establish interleaving or resumption.

## Formal Status

I keep my proof boundary explicit:

- proved: my existing NanoCore subset in `formal/`
- tested: async syntax/typechecking, synchronous interpreter compatibility, and scheduler unit behavior
- planned proof work: async typing and operational soundness extensions
