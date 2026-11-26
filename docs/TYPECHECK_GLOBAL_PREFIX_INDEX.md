# My global symbol prefix during checking

I use this design under #1255. My exact8fa/genuine7f431 phase receipt reaches program lowering just before300 seconds; its typechecking interval lasts196.740117 seconds. My bounded current stacks include `env_lookup` in five of eight samples. These observations motivate an audit. They do not establish a cost fraction, sole cause or performance gain.

## Lifetime and the filtered exception

`typecheck_advisory_core` copies its initial symbols, appends builtins, globals and function signatures, then checks every global initializer. Initializer checking may append a later binding with the same name. Only after that loop do I have the completed global prefix.

During function and shadow checking, parameters, recursive float/self bindings, lets, match/record-match bindings, loop variables and effect-handler parameters append to that prefix or a copied scope. Blocks, flow and ordinary parallel scopes copy their input. Lambda creation carries its lexical input forward. My production Symbol-array audit finds append/copy operations and no in-place replacement of prefix names. Symbol metadata remains in the supplied array.

`check_legacy_par` is different. It filters sibling spellings from a freshly built array, including an outer global with the same name. That array does not retain my prefix. I clear and disable the private index at this boundary, before filtering. I retain the original scans for the remainder of that advisory call; I do not restore the index from a filtered child.

My advisory entry wrappers call the core; I find no recursive advisory entry from its checking path. My existing enclosing-function state already requires sequential checking. I clear private state at advisory entry and on every return through its existing signature. Standalone helpers retain the scans. Arbitrary prefix-name mutation or concurrent checking during an active core is outside this audited invariant. A shorter array selects the original scan. This is an implementation audit, not a formal proof or a general container-identity cache.

## Representation and lookup

I retain a private `HashMap<string,int>`, prefix count and active flag. A forward build overwrites earlier duplicate names, preserving the original most-recent ordinal. Counts beyond the native array endpoint range retain scans. I store neither Symbol records nor types, mutability, callable flags, signatures or advisory facts.

`env_lookup` first scans the current local suffix backwards, including every original empty print for a nonmatch. On a suffix miss it queries the prefix map. A prefix hit at ordinal `i` performs `prefix_count-i-1` empty prints; a miss performs `prefix_count` prints. It then returns the original ordinal or `-1`. All metadata reads and module/name binding remain in their original callers and read the live supplied array.

I begin only after global-initializer checking and immediately before phase2. I keep `typecheck_advisory_core` as a cleanup wrapper around its copied indexed body. The copied body preserves every original statement; I change its private name, add one begin call and normalize two whitespace-only lines. I add no Symbol/Parser field, public option, schema, ABI or C product change.

## Comparison and acceptance

My baseline retains the original lookup and advisory body with shared private entry cleanup. Both roles share prefix support, the copied body, filtered-scope invalidation and useful new assertions. The baseline core does not call the indexed body. Replacing the candidate's two production bodies restores the baseline; removing shared additions and the two private cleanup hooks restores every original source byte. I bind these reversals and all file hashes explicitly.

I embed each exact implementation in a separately hashed comparison fixture, followed by one common harness derived from `tests/typecheck_scope_copy.nano`. I remove only its implementation import, preserve its original report body/assertions, and add calls to the same new shadow bodies. I preserve all24 original sources and complete literal expected reports in `tests/typecheck_scope_copy_cases.json`. This is a standalone comparison, not an unchanged permanent unittest invocation.

My new controls cover empty/missing/case/prefix/duplicate names, latest global and local shadowing, local alias replacement/growth, live metadata with fixed names, standalone aliases, alternating equal-size initial environments, failure recovery, advisory output and the legacy filtered-global case with enough suffix bindings to remain longer than the indexed prefix. That filtered case must retain the exact missing-name diagnostic. I also cover a quiet preflight refusal in the copied body.

I require C-seed native, C-producer VM and ASan+UBSan native, and genuine NanoLang-producer VM and ASan+UBSan native for both roles, retaining600-second builds,60-second runs and ordinary ten-second shadows. I retain original sanitizer/link flags, all source/fixture/harness/image/tool/host/producer guards, actual cleanup controls and terminal evidence. I inspect actual retained generated C call paths.

Only an independently checked and pushed component permits another fresh exact-source original bootstrap and native36/emitter12 gates at1800/30/10 and300/default10. The current original emitter gate remains unmet. C-owner repairs, paired qualification, full pure runtime C emission/cutover and all69 release requirements remain open.

My initial HashMap implementation is not qualified. Its retained baseline produces VM-valid bytecode and passes C-seed/VM runtime controls, but C translation refuses string-to-optional storage under #1263. Typed string and typed Symbol bindings also refuse. I retain those failures and am evaluating a projected-name-vector sink as a separate readiness control; I have not selected or accepted that representation or repaired the C defect.

My [complete refusal/control receipt](evidence/darwin-current-source-20261124/diagnostics/symbol-prefix-map-storage-refusal-linux-20261126/README.md) now independently verifies19000 records. Both typed-key neighbors still refuse; the projected vector passes VM and C translation only. I retain the complete unqualified HashMap draft and restore my live typechecker to the original source. This document records that first proposal; I require a revised vector contract and fresh full qualification before adopting it. #1263 remains open for the C owner.
