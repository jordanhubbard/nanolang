# My shared verifier profile checkpoint

I tested source `5bfea584`, then restacked it as `47c9c931` onto main
`92680b8a`. That integration changed only documentation; my compiler and tests
are identical. MAC `task_037b12aecc894b86ba335828fa1eb1a2`.

I moved my existing LLVM module/signature/opcode eligibility predicate into
`nvm_verify_profile`. General admission returns ordinary verification unchanged.
Closed-scalar admission verifies first and applies the same existing restrictions.
LLVM consumes that API; Wasm consumes LLVM. I retain existing diagnostic wording
that names the scalar LLVM profile: the shared boundary applies to both targets.
There is no wire-format change, extra source-list dependency or execution widening.

On Linux ARM64 I passed:

- `make test-verifier-profiles`: 12 module cases compare ordinary/general
  decisions and diagnostics, closed-scalar/LLVM decisions, unknown-selector
  refusal, and zero emitted LLVM bytes on refusal. The cases include integer,
  float/bool/U8/void, implicit return, advisory metadata, strings, globals,
  imports, nominal metadata, non-scalar parameters, initializers and missing or
  unsuitable executable entries. LLVM/Wasm refusal preserves prior artifacts.
  Missing-module verification remains refused by both profiles.
- `make test-nvm2llvm`: 16 execution methods.
- `make test-nvm2wasm`: 39 Wasm and adjacent scalar methods, including U8,
  floats, truthiness, implicit returns and generic comparison semantics.
- `make test-verifier`: 96 existing assertions and the verifier allocation
  cleanup gate.

My unit profile gate needs no external LLVM tool for refusal checks; successful
Wasm execution remains in its explicit existing target gate. Independent source
review found no scoped blocker. No `.nano` source changed and I did not repeat
an unrelated compiler bootstrap or claim full release acceptance.

I have not defined a GPU kernel contract, completed rich frontend-fact transport,
or completed full applicable-language LLVM/Wasm coverage. Those parent roadmap
obligations remain open.
