# Exclusive multiple-borrow File calls

I continue #989 from `205ff8f1b` with the [counted reference-map contract](../../NANOISA_FILE_MULTI_BORROW.md).
I preserve `CALL_REF` and add `FILE_CALL_REFS` for arbitrary parameter maps within
my existing 256-slot bound. Both lowerers retain source evaluation order and
emit their own maps. The query owns copied maps; runtime entry validates every
reference and distinct owner before staging values and binding formals. Native
agreement covers the copied map as well as decoded operands and obligations.

My source fixtures use three reordered and forwarded loans mixed with a scalar,
repeat calls in a loop, then read back exact bytes. The second fixture moves and
returns an owned File between two borrowed arguments. All five generated shadows
stay unchanged; helper shadows make the selected suites eight and nine entries.
Assertion controls retain multiple active loans and verify failure cleanup and
unchanged publication output.

My lowerer harness tests malformed lengths, invalid slots, missing reference
sentinels, overlapping owners and invalid constant indices. It verifies copied
facts remain unchanged after source-map mutation. The same modules run through
real granted VM/native consumers with descriptor-balance and cleanup assertions.
The opcode matrix adds raw encoding/roundtrip/truncation and complete generic
consumer refusal for `0x97`; it retains all original six opcode cases.

## Qualification

- `final-c-cli.log`: seven C driver methods pass in 123.976 seconds.
  `owned-c-cli.log` and `failure-c-cli.log` cover the two added methods in
  56.248 and 0.428 seconds, completing the nine-method C matrix.
- `final-nano-cli.log`: all nine Nano driver methods pass in VM/native form
  in 358.641 seconds, including both multi-borrow fixtures and shadow refusal.
- `final-lowering.log`: all eleven source methods pass in 238.538 seconds,
  with the selected flow/runtime/lowerer instrumentation and native generated
  code controls. The two mapped calls retain malformed-map and copied-fact checks.
- `final-neighbors.log`: linked/instrumented opcode, cyclic, cyclic-hosted and
  indirect-hosted queries pass, including their allocation-failure sweeps.
- `dispatch.log`: matched linked/instrumented VM/native cyclic dispatch passes
  in 237.084 seconds.
- `final-driver-build.log`, `translate.log`, `native.log`: the C-produced Nano
  compiler builds with its selected shadows and translates to native form.
  Its current artifact dependency hashes match the changed query/runtime sources.
- `inputs.json` pins source files and development compiler artifacts. The
  generated schema check also passes.

## First qualification failures

`neighbors.log` stops at missing `opt` while rebuilding the managed runtime
package. I put the installed LLVM tools on PATH. `neighbors2.log` and
`neighbors3.log` retain the File opcode harness selecting AppleClang despite the
Make CC selection; its leak detector refuses before fixture execution. I pass
CC and platform CFLAGS into that runner and retain the corrected final results.
These failures do not establish a File runtime defect.

## Scope

These Darwin development gates do not replace Linux installed qualification,
exact release bootstrap or the full 5.1 scope. Shared-reference and indirect
source calls remain open. Runtime/flow providers are instrumented in the source
harness; its common compiler objects and archive neighbors retain their stated
ordinary builds. The compiler executables themselves are not fully instrumented.
