# I qualify private STRING scopes and retain ordinary-mode failures

I qualify source `c4e2aa0500e918fb0304a0734d4cedb30e1c0af6` on Linux ARM64 with GCC13.3/as2.42. My private STRING reference scopes pass the selected lifetime controls. **The ordinary native-byte criterion still fails, and ordinary sibling binding mutation still produces use-after-free.** #1213, #1217, #1211, #1116, #1140 and full release acceptance remain open.

I track explicitly fresh evaluator STRING references, retain separate references for managed reads and owning bindings, and release consumed references at scope boundaries. Returning a value preserves one reference, including duplicate read references to the same payload. The original pointer and original owner's reference remain intact. Fresh helper results are distinguished from borrowed identity conversions. My CLI and ordinary environment creation do not enable this private session. Value/container/FFI layouts and checker admission are unchanged.

My final52-command cohort uses fresh complete normal and ASan/UBSan providers. The original ten-shadow carrier fixture, complete six-shadow record/union fixture, seven-shadow scalar fixture/static refusal and two-shadow typed-list source witness retain their original assertions. The new ten-shadow STRING fixture covers aliases, owning record rebinding, projections, helpers, loops, sibling mutation and argument evaluation. Both64-call batches retain93bytes/two GC objects accounted for by the existing global STRING slots; context teardown leaves zero GC bytes/objects.20950 registered references balance20949 releases and one caller export. The C caller's input pointer/content/refcount survive. A separate loop-return source control balances11/10/1/0 through duplicate references and leaves zero GC objects. These are reference counts, not allocation counts.

All134 original evaluator cases pass in ordinary and privately tracked modes on both profiles; original checker88 also passes. Those broad sanitizer suites explicitly disable LSan. Focused source, metadata, engine67667/fault, storage24 injected failures, binding991/991, borrowed-value, duplicate-reference and leaf BSTRING/record-field/string-bounds controls enable it. The original callback-record batch balances403 registrations as402 destructions/one transfer and holds one live carrier after each64-call batch. Explicit host-list roots preserve original push/get pointer identity through replacement and collection.

I retain and repair these scoped prerequisites:

| Scope | Retained failure | Qualified behavior |
| --- | --- | --- |
| #1213 library scratch storage | Direct C/API owners retire, then LSan reports13bytes/two buffers | I free only the copied concat/substring scratch buffers; caller stack storage survives two allocation failures and recovery. The existing substring empty fallback remains. |
| #1214 checker loop metadata | The completed STRING fixture reports352bytes/two inferred TypeInfo nodes after runtime references balance | Only newly copied trees gain recursive checker ownership. Eight checker-only teardowns and32 private runtime shadows preserve borrowed AST data after environment teardown. |
| #1215 nested-array iteration | Admitted nested-loop shadows receive VOID | I read the existing Value slot; all original plus nested break/continue controls pass. |
| #1216 foreign argument refusal | Private root observation reads a deliberately NULL argument array before the original guard | I observe only readable widths, keep dispatcher arguments/count unchanged and pass every original evaluator case. |
| #1211 array accounting | The private path misses the original >=5 retirement-counter assertion | I count actual array frees in either ownership path; I do not count registration or intended cleanup. |

I retain initial fixture setup failures: a missing C header, unsupported interior STRING-field mutation, an incorrect VOID expectation for the existing substring fallback, and a one-global assertion. My admitted positive uses owning-record rebinding; the separate negative still requires the exclusive borrowed-owner refusal. I observed both existing global slots, derived the final baseline from their distinct live roots, and retained no-growth and zero-after-teardown assertions.

## I keep the failing boundaries visible

The unchanged native observer passes its2MiB allocation/release control and reports zero ASan allocated-byte growth over calls65..128. The final ordinary `mallinfo2` observation adds1776bytes and exits94 against the unchanged1024-byte criterion. An earlier repaired-provider run adds1584. I retain both and assign no allocator or product cause; I do not raise the allowance or treat the ASan result as ordinary acceptance.

The unchanged sibling-mutation source fails under ASan in both initial ordinary/private modes. The repaired private route completes all three shadows; ordinary mode still fails. Default scalar/record source retains1857bytes/78allocations and3600/153; both original default typed-list UAFs remain. Default nested-loop runtime replay completes all32 shadows then reports5080bytes/176 unbound-array allocations. A separately labeled semantic-only replay disables LSan and completes; private runtime replay keeps LSan enabled. None closes default aggregate ownership.

Trusted helper/foreign construction, asynchronous storage, external-list/GC-shutdown lifetimes, complete typed roots and full original compiler/source/platform gates still precede default activation. I claim no paired bootstrap, Darwin, current CLI or69-requirement release acceptance and no historical timeout/performance attribution.

## I record the archive boundary

I verify2053 records,1413 exact Git inputs and38 ELF images (36 executables/two test shared objects). All146 development product and1207 provider object/archive identities are recorded; other bytes remain in their retained private roots. The nine-library loader closure is an after-qualification identity snapshot, with no before/after stability claim. Hash-bound objects from the preceding archive are reused. The original fixtures and all other source inputs outside the18 changed paths remain exact; the reader checks the unchanged native threshold and original assertion bodies.

I overwrote the first combined Make stdout/stderr for the incorrect substring-fault expectation during iteration. Its recorded exit2, exact source snapshot and failed executable remain intact. `curation-gap.json` records that limitation; `fault-retained-replay.*` is a separately labeled bounded replay of that retained executable, exiting1 at the same assertion. I do not present the replay as the original combined output.

All native jobs are terminal, with no Linux/Darwin reservation. Puck-owned source/merge/capture/CI inputs, the five retained tool images, frozen roots, audits, stashes and NAS/TCC/migration pauses remain unchanged.

```sh
python3 docs/evidence/carrier-string-scopes-linux-20261123/read_archive.py
```
