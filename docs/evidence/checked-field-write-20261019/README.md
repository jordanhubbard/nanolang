# Checked exclusive field-write component receipt

I qualify the bounded #998 field-write continuation from field-read component
`d67295b3d`, following reviewed design `a155a81a0`. Existing exclusive assignment
checks already establish a destination place. I retain that event only after the
single existing RHS traversal succeeds, then publish independent advisory
`nano.field-write.v1` only for the actual successful REF_SET lowering. No new
executable profile, checker permission or runtime authority is introduced.

## Local qualification

- My direct ledger exercises exact assignment/RHS kind and pointer, target/field,
  coordinates, completed epoch, cloned-site refusal and recheck invalidation.
  All three call/read/write budget-trigger directions omit all optional facts.
  All 24 reached write growth-allocation failure prefixes and independent recovery
  pass, alongside the previous 22 read prefixes. Actual Environment teardown with
  all three vectors and expired AST storage passes all seven initial denial
  positions. I do not claim whole-program leak qualification.
- My codec passes 1504 checks, including exact mode 1, framing, malformed/duplicate
  refusal, copy, canonical transport and native-byte invariance. I corrected the
  new unknown-version test's literal length from 18 to 19 to include its entire
  version suffix, then reran it successfully (`codec/final-test.log`).
- All 14 strict GCC 16/Clang provider syntax checks pass. Five comparator controls
  pass, preserving changed-authority and surviving/shared string references.
- The isolated changed C provider passes two source methods in 0.855s. Existing
  owned/shared/exclusive/nested read counts remain exact; the exclusive helper has
  two distinct writes and three reads. The admitted RHS sequence uses existing
  reads/ADD/MUL, with exactly two REF_SETs and verified execution order/result.
  Later invalid RHS, shared destination and runtime-shadow failures preserve prior
  output. This requires no unsupported effectful helper/foreign profile.
- The C-hosted canonical Nano compiler builds successfully with new helper
  shadows. The paired corpus passes 13 methods in 85.640s: actual checked-context
  RHS/target mutation refusal, stale/recheck/reset and descriptor lifetime,
  source facts, canonical transport, actual VM/sanitized native execution,
  raw/checked comparisons and negative output-preservation controls.
- The final standalone raw emitter builds successfully. Its four-fixture
  raw/C/checked neighbor method passes in 3.452s. The earlier full suite used the
  already qualified field-read raw provider; this final gate qualifies the new
  standalone raw product without rerunning unchanged tests unnecessarily.
- A no-write fixture is whole-module byte-identical to the previous field-read
  C provider, including all existing read/call records (`no-write/`).

## Original comparator boundary and correction

I drafted the narrow new-key normalizer from the approved suffix contract before
executing the old normalizer on a new artifact. I then retained an actual
`d67295b3d` normalizer invocation on emitted write bytecode: it refuses the suffix
with `I require an exclusively advisory trailing pool suffix`. The same artifact
passes the reviewed write-only prefix. `normalizer-before/` retains exact original
normalizer/source/module/canonical hashes, commands and failure. This chronology
is explicit; I do not claim a failed pre-edit suite that did not occur.

The new prefix validates fixed-header/site/UTF8/mode semantics, then proves its
trailing strings are exclusively owned before removal. Existing read/call/affine
normalization follows unchanged. It never ignores arbitrary metadata, CODE or
ownership/layout changes. All comparator controls and actual paired products pass.

## Exact evidence boundaries

`compiler-inputs.json` still matches final production sources and seed.
`final-source-sha256.json` pins final source/tests; `artifacts-sha256.json` pins
actual products. Focused C rebuild commands and retained compatible dependency
hashes remain under `c/`; this is not a claim of an independently rebuilt complete
aggregate. C-hosted compiler products are not fresh raw stages.

Fresh aggregate Stage1/Stage2 generation, fixed point and exact Linux/Darwin
qualification remain required for new write/context/result shapes. Full #998
remains open. Existing imported/helper, generic/synthetic and raw absence
boundaries remain unchanged. Existing checked-borrow ledger/Environment/metadata
Make targets cover the additions; the coordinator owns aggregate workflow and
fresh-product integration. The metadata selector remains colon-separated
`NANOLANG_BORROW_FACT_COMPILERS`, with `NANOLANG_BORROW_RAW_EMITTER` selecting the
actual standalone raw compiler.
