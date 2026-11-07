# I qualify declaration-only MAP_FIELDS

I implement reviewed `030f1afa9` on base `878c2bb83`, after the retained map baseline/design ancestry. Issue ownership: https://github.com/jordanhubbard/nanolang/issues/1001#issuecomment-6312992270. This checkpoint provides copied, immutable exact map declarations; it does not admit map execution or publish source authority. Public verifier/native/serialization still refuse selected map metadata, including map-only metadata without ARRAY. No successful executable canonical roundtrip is claimed.

## I retain actual qualification

- Final `python3 -m unittest -v tests.test_map_declarations`: PASS 1 method, 2.297s; linked 131825 and instrumented 135346 checks. Exact 65536 bindings, sorted/duplicate/reserved/referent/coverage failures, standalone map parameter/result/global rejection, permitted coarse local descriptor, copied lifetime, unchanged failure outputs, allocation failure/retry and public refusal are covered.
- Old declaration/private-layout neighbors: PASS 2 methods, 3.495s. Old ownership contracts: PASS 2443 checks.
- Strict Clang and GCC16 `-std=c11 -D_DEFAULT_SOURCE -Wall -Wextra -Werror -fsyntax-only` pass both changed C providers and the new C harness. Exact commands/status are in `neighbors-final.json`.
- Final six old budget profiles match the previous self-tail receipt byte for byte. `Analysis` remains 39840 bytes; complete values are in `old-six-budget.stdout`. No old allocation/work boundary changed.
- Map maximum observed accounting is 787873 bytes / 3607975 logical work units, with 128-byte scratch. The small exact boundary is 1258 bytes / 252 work; one below refuses and recovery succeeds. Twelve map allocation positions are tested in both failure modes with independent retries. Immutable field getters require caller precharge of 20 bounded logical probes, not a claim of 20 machine operations.
- All 37 retained compiler/link-provider hashes match before/after. Changed ownership/layout providers compile directly into isolated test products. `environment.json`, command receipts and `source-pins.json` identify inputs. No installed tools, compiler generations or aggregate objects were changed.

## I preserve earlier failures

The first old declaration neighbor expected UNKNOWN for a nested ARRAY referring to U8. Exact HEAD `030f1afa9` reproduced PREPARED before this product change (`baseline-neighbor.*`). I corrected only that stale expectation to PREPARED with exact type/referent/binding getters; all other malformed/unknown controls remain. This is not a map-runtime regression. The initial failed neighbor and final corrected terminal are retained separately. One final runner invocation from the unrelated default cwd failed to import `tests.native_process`; the same script ran successfully from this isolated worktree, without product changes.

## I preserve the phase boundary

`test-map-declarations` is wired into `test-units`. It uses the existing DECLARATION_CC/CFLAGS/OBJECTS/LDFLAGS interface; `environment.json` is the exact local invocation environment. The map classifier is framing-only: ABSENT is not semantic validation. Kind 5 remains rejected by old public NULL routes. Unsupported well-formed pairs can be described but cannot execute. Existing simultaneous extension maximum four remains unchanged.

The next phase remains contract-only: exact map origin/write closure, STRING-key retain/replacement/headroom, absent GET behavior, copied-module lifetime, private/public VM and forced-GC native identity require independent design and qualification. DELETE and runtime/publisher admission are excluded. Full #1001 and exact hosted platform qualification remain open.
