# My formal tooling decision

Status: 2026-09-11 assessment; Sail adoption remains an experiment.

## Decision

I retain Rocq for NanoCore and evaluate Sail for executable NanoISA semantics.
I do not port the language proofs to Lean, Isabelle/HOL, or HOL4 for 5.0.
This is an engineering decision based on the evidence below, not a claim that
one logic or prover is universally better.

My trust gap was not a missing prover. Tuple changes had broken the proof
build, my advertised evaluator theorem did not exist, and the reference
evaluator disagreed with the relational semantics on short-circuiting. Those
defects are now repaired. I have a checked general evaluator theorem, independent
library checking, exact theorem-type contracts, assumption rejection, and a
passing GitHub proof job. Rewriting this development would reopen obligations
without establishing correspondence with my production implementation.
See [my checked scope](../formal/README.md).

## Alternatives and their costs

| Tool | Relevant capability | My adoption decision |
| --- | --- | --- |
| Rocq | My existing preservation, progress, determinism, pure simulation and evaluator proofs now build and check independently. | Keep it as the language-model authority; spend effort on correspondence and useful runtime invariants. |
| Lean | A kernel checks elaborated proof terms; axiom dependencies are auditable. Native computation can introduce a compiler-trust dependency. | A credible future contributor/tooling choice, but I have no measured migration benefit. Do not duplicate NanoCore now. |
| Isabelle/HOL | Its code generator produces SML, OCaml, Haskell and Scala; its ecosystem includes Sledgehammer and Nitpick. | Consider it for an ISA proof when an existing Sail/Isabelle development provides a concrete advantage, not as a second language specification. |
| HOL4 | Its documented system includes the kernel and theorem libraries; the CakeML work demonstrates a substantial verified implementation ecosystem. | Consider it when a specific CakeML/HOL4 integration is required. That integration is not present in my C/VM implementation today. |
| Sail | It targets ISA definitions, bitvector checking, executable C/OCaml models and prover definitions. Backend support varies by model. | Trial it on real encoding and execution behavior. Do not equate generated prover definitions with a proved ISA or verified VM. |

Primary sources: [Lean kernel and compilation](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/),
[Lean axiom tracking](https://lean-lang.org/doc/reference/latest/Axioms/),
[Isabelle code generation](https://isabelle.in.tum.de/doc/codegen.pdf),
[Isabelle tools](https://isabelle.in.tum.de/documentation.html),
[HOL4 documentation](https://hol-theorem-prover.org/docs/trindemossen-2/),
[CakeML verified proof-checker work](https://cakeml.org/jlamp20.pdf),
[Sail capabilities and limitations](https://github.com/rems-project/sail).
I inspected documentation for these alternatives; I have not benchmarked their
proof ergonomics or ported NanoCore to them. The Sail 0.20.2 binary also exposes
a Lean backend in its help output; I have not validated that export.

## Authority and correspondence

I keep `spec/nanoisa.yaml` authoritative for opcode identities and layouts.
`scripts/gen_nanoisa_schema.py` already generates production metadata from it.
A Sail experiment must check its opcode declarations against that schema;
copying constants indefinitely would create another drift-prone specification.

I write Sail behavior independently, then compare its generated executable with
`isa_decode` and the VM. Generating both expected and actual behavior from the
same C implementation would hide errors. Differential tests are evidence of
agreement on tested inputs, not a refinement proof.

My first slice covers little-endian integer constants and stack operations.
It must grow to cover truncation, extension prefixes, stack/type errors,
integer boundaries, branches and trap outcomes. Heap ownership, scheduling,
capabilities and service recovery need explicit state models; Sail does not
provide those policies merely by generating an emulator. No trial model becomes
normative until coverage, compatibility and maintenance ownership are reviewed.

## Product boundary

My users should write NanoLang, not discharge Rocq obligations to use an array
or call a service. I concentrate proofs on reusable semantics, verifier rules,
ownership transitions and runtime boundaries. I keep shadow tests, structured
diagnostics and ordinary executable examples as the everyday authoring path.
More proof tools do not make that path enjoyable by themselves.

My [bounded Sail trial](../formal/sail/README.md) now typechecks, generates C,
compiles and runs seven smoke assertions with Sail 0.20.2. Schema checks and
1,524 differential byte sequences now establish tested agreement with the
production decoder for that slice. Another 1,140 comparisons establish tested
agreement with actual NanoVM integer stack execution, including 34 underflows
and frame-local boundaries. This exposed and repaired silent underflow in
three unverified VM handlers. This is not an ISA refinement proof.
The next acceptance evidence is broader integer arithmetic, control flow,
trap semantics, and a validated prover export.
An actual Rocq export attempt now exposes a Sail 0.20.2 internal rewriting
failure on my decoder's literal-byte list pattern. The export-only command
and failed annotation experiment are recorded in the trial README. I retain
the C differential evidence, but do not claim that this backend can yet
produce a checked Rocq model.
The broader language contract, native service runtime, scoped FFI and release
controls remain in [my roadmap](ROADMAP.md).
