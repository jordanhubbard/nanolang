# Bounded nested STRING authority qualification

I extend the existing retained ARRAY_FIELDS profile to exact depth-two
`array<array<string>>` fields. I keep a distinct normalized STRING obligation,
original descriptor identity, independently checked transport, and all-writes
constraints. I do not infer a leaf from the outer ARRAY tag. Existing flat and
nested INT byte ordering and resource-budget controls remain in the corpus.

## Implementation and review

The coordinator reviewed the contract, proof/runtime/native changes and paired
C/Nano serializers before qualification. The serializers emit deterministic
STRING leaf and ARRAY-reference rows; mixed INT/STRING rows have exact wire
assertions. Optional publication still requires the complete selected proof and
final verifier. Foreign/large modules and unsupported descriptors retain their
omission controls. No retained proof token authorizes runtime execution.

I use separately instrumented actual public/private VM providers and canonical
native output, not a VM wrapper presented as native execution. The native corpus
has seven programs at O0/O2 in each provider build (28 executions), with
ASan/UBSan, forced collection, alias/mutation/call/tail and string-lifetime checks.
Copied plans retain independent embedded-NUL constant bytes after source storage
is destroyed. Typed refusal, exact failure cleanup, retry, and the maximum
private release stack are checked separately.

## Reproduced failures

* `consumer-first.log`: the canonical native forward-return path selected scalar
  storage before an unknown receiver resolved to a STRING-array child. I extend
  only the existing canonical nested-array deferral to SARR; CALL/TAIL fixtures
  subsequently execute as native code at both optimization levels.
* `intern-before.log`: actual public STR_CONCAT matched a different saturated
  interned result and wrapped its count to zero. I make the existing
  vm_string_new NULL-on-UINT32_MAX guard unconditional. Final controls verify
  public/private memory refusal before pooled-count mutation, unchanged operand
  counts, balanced teardown and independent valid retry. This is not a claim
  that every other retain path has been qualified.
* `source-first.log`: a new fixture omitted whitespace between `>` and `=`,
  forming GE. The correction is test syntax only.
* `source-untyped-empty.log`: the new fixture's untyped empty replacement was
  refused by the Nano checker and failed a C shadow after refill. The supported
  case explicitly declares `empty:array<string>`. I retain contextual empty
  inference as a separate compiler boundary; I do not widen types or weaken
  shadow assertions to make publication pass.

## Evidence boundaries

These are macOS focused component results from base4cfad3737 plus this change.
Fresh raw-generation bootstrap and Linux/Darwin aggregate gates are owned by the
coordinator and remain required. This receipt does not close issue1001 or the
release umbrella. ARR_REMOVE, deeper arrays, other nested leaf types, resources,
foreign callbacks, recursive/cyclic authority and other unsupported profiles
remain outside this bounded slice.

I retain filtered logs here (only compiler command lines above1500 characters
removed); original log hashes are in original-log-sha256.txt. Temporary artifact
paths remain recorded in logs rather than being claimed portable CI artifacts.

## Completed local checks

| Check | Result |
| --- | --- |
| Final public/private nested STRING consumers | 2 methods, 35.113s; 28 native O0/O2 executions |
| Expanded INT/STRING origin and resource controls | 1 method, 4.249s |
| C plus freshly C-hosted Nano source emitter | 19 methods, 22.735s with typed empty replacement |
| Publication wrapper failures plus timeout controls | 7 methods, 125.749s |
| Ordinary declaration provenance | 9 methods, 4.484s |
| Earlier unchanged nested INT consumer regression | 2 methods, 31.759s |
| Earlier unchanged tail consumer regression | 1 method, 26.672s |
| Earlier unchanged private VM regression | 2 methods, 18.018s |
| Earlier unchanged public lifecycle regression | 1 method, 229.383s |

`final.log` contains the green final consumer/origin checks followed by the
untyped-source fixture failure; it is not a green aggregate run. The subsequent
`source-typed.log` records the corrected source fixture pass.

The consumer checks enumerate 55 proof, 289 public VM, 86 private preparation,
and 13 private execution failure positions, each transient and persistent with
independent recovery. The wrapper checks enumerate publisher/query positions:
flat264/259, nested INT268/268, nested STRING280/282, both failure modes. They
also force assembler buffer-growth failure, preserve sentinel output, and retry.
The earlier regression log predates the narrow unconditional saturation guard;
the final STRING suite exercises that guard in both provider configurations.

The final paired source run passes all19 methods in20.429s, including canonical
C-hosted Nano imported-source binding, mixed leaf ordinals, exact flat/INT/STRING
wire bytes including reserved padding, real VM/native execution, independent
assembly verification and prior-output refusal controls. `source-tools-sha256.txt`
pins the C producer, C seed, fresh source emitter and canonical compiler.
The canonical build is `bin/nanoc_c src_nano/nanoc_v06.nano -o
/private/tmp/nanolang-nested-string-canonical` (exit0); this is a C-hosted
compiler build, not a fresh raw self-host bootstrap.

The final source command sets NANO_FLAT_ARRAY_COMPILERS to the pinned nano_virt
and emitter paths, NANO_FLAT_ARRAY_BOUND_COMPILERS to nano_virt and the pinned
canonical path, and runs `python3 -m unittest -v
tests.test_flat_array_source_publication`. Consumer qualification uses
`make -j1 test-record-array-nested-string test-record-array-origins`; wrapper and
provenance qualification uses `make -j1 test-publication-allocation
test-ordinary-declaration-provenance`. Source hashes in component-source-sha256.txt
pin the isolated production/test inputs. All commands run in this worktree.
