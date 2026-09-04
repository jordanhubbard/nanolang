# ANS Forth on NanoISA

## Status

This is the implementation contract for replacing my current Forth-like
example. I will target the maintained Forth standard, including the Core word
set and every optional word set.

I pinned the standard, test-suite revisions, Gforth 0.7.3, licensing facts,
and the environmental contract in [Forth 2012 Pins](../../FORTH_2012.md).
I am not a Forth 2012 Standard System until those pinned suites pass on the
NanoISA compiler.

Passing a test suite is evidence. It is not, by itself, a conformance claim.

The session runtime in `src/forth/` exists: one mutable `NvmModule`, one
persistent `VmState`, Forth stacks, virtual addresses, file handles,
dictionary headers with early-bound execution tokens, nested input
sources, and colon compilation to verified NanoISA functions.
`make test-forth-session` is the evidence. Typed imports and `SEE` are
still ahead.

## Architecture

I will compile each Forth colon definition to a verified, zero-argument
NanoISA function in one mutable `NvmModule`. One `VmState` will live for the
whole interpreter session. Definitions will call earlier definitions with
`OP_CALL`, so redefining a word will not change already compiled callers.

NanoVM's operand stack belongs to its function calling convention and keeps at
most one return value. It therefore cannot also be the Forth data stack. The
Forth data stack, return stack, floating-point stack, and address space are
persistent session-owned state (`ForthSession`). NanoISA functions will use
private runtime helpers to access that state and use the operand stack only
for temporary values.

The outer interpreter and compiler will maintain:

- input-source state and nested source restoration;
- interpretation and compilation state;
- dictionary headers, execution tokens, name tokens, immediacy, and word lists;
- a compile-control stack for forward and backward branch patching;
- exception frames that restore stacks, locals, and input sources;
- implementation-defined environmental limits and query results.

New definitions are built privately, appended between VM invocations, checked
with `nvm_verify_function`, and published to the dictionary only after they
verify. A host call uses `vm_invoke`, which restores NanoVM's temporary operand
stack and frames while preserving globals and heap state.

## Module Interoperation

A typed Forth import declaration will create an `NvmImportEntry`. Compiled use
of that word will emit `OP_CALL_EXTERN`, using the same `vm_ffi_call` and
co-process isolation path as NanoLang compiled to NanoISA.

An import declaration must name:

- the NanoLang module;
- the exported C symbol;
- each NanoVM parameter tag;
- the NanoVM return tag.

I will reject signatures that the current FFI ABI cannot call correctly. The
initial boundary supports integer, boolean, string, opaque, and supported array
values, at most ten parameters, plus all-float signatures. Mixed float/integer
signatures require a general ABI layer and will not be guessed.

Adding an import invalidates a running FFI co-process because that child has a
copy of the old import table. I will stop it and let the next external call
restart it lazily.

## Standard Word Sets

Implementation order follows semantic dependencies rather than chapter order:

1. Core and Core Extensions
2. Exception
3. Double Number, String, and Search Order
4. File Access, Memory Allocation, and Locals
5. Facility and Programming Tools
6. Floating Point and Extended Character
7. Block

The completed system will document support for:

- Core and Core Extensions
- Block and Block Extensions
- Double Number and Double Number Extensions
- Exception and Exception Extensions
- Facility and Facility Extensions
- File Access and File Access Extensions
- Floating Point and Floating Point Extensions
- Locals and Locals Extensions
- Memory Allocation
- Programming Tools and Programming Tools Extensions
- Search Order and Search Order Extensions
- String and String Extensions
- Extended Character and Extended Character Extensions

## Environmental Model

The initial contract is:

- 64-bit two's-complement cells;
- 8-bit address units and UTF-8 text;
- logical 128-bit double cells represented by two 64-bit cells;
- floored division for `/`, `/MOD`, and `FM/MOD`;
- a byte-addressable virtual Forth address space, not host pointers;
- IEEE binary64 values on a separate floating-point stack;
- nested terminal, evaluated-string, included-file, and block input sources;
- file and allocation handles validated by runtime-owned tables;
- block tests confined to an explicitly disposable backing image.

Every implementation-defined behavior required by the standard will be listed
in the eventual conformance document.

## Verification

I will pin and run:

- the committee tests from `Forth-Standard/forth200x`;
- the Gerry Jackson Forth-2012 test suite where its licensing permits vendoring;
- differential runs against a pinned Gforth release;
- repository tests for malformed definitions, early binding, exceptions,
  overflow, source nesting, UTF-8 boundaries, FFI, and persistent VM recovery.

The existing 280 tests remain regression tests while the implementation is
replaced. They will be rewritten to exercise standard behavior rather than
special built-in test words.

## SDL IDE

The SDL IDE remains a PTY client. Its file panel will load the updated standard
Forth examples. The interpreter executable behind the PTY changes; the IDE does
not get a private language implementation.

`SEE` will disassemble the actual NanoISA function for a colon definition.
Imported words will show their module, symbol, and typed signature.

## Delivery Rule

I will not label the result an ANS or Forth-2012 Standard System until all
selected word sets pass their pinned automated tests and the required manual
and implementation-defined behaviors are documented. Until then, banners and
documentation will name the completed milestone explicitly.
