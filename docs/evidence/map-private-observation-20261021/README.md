# Private map copied observation qualification

Base: aggregate `48e1508ae`; reviewed contract `ad02d1538`. This component adds private copied observations, byte lookup/string copy and two-path identity comparison. It does not enable public, native or publisher admission.

## Final commands and results

`final-corrected.py` supplies the exact retained environment and four original modules, then runs `python3 -m unittest -f -v tests.test_record_map_observation tests.test_record_map_runtime tests.test_record_array_vm tests.test_record_map_heap`. Seven methods passed in 63.333s. Observation switch/computed-goto each passed 19,208 checks; prior map runtime each passed 2,714. Old array VM and map heap neighbors passed. Providers/harness retain ASan/UBSan instrumentation; the environment, source and provider hashes are retained here. Original inputs are the unchanged seed, C-hosted Nano and retained Stage1/Stage2 STRING-to-INT modules; this is not a fresh frontend qualification.

`strict.py`: all 14 Clang/GCC16 ordinary/array/preparation/runtime/harness/heap syntax cases passed. `budgets.py`: all six legacy rows match `record-array-self-tail-20261021/old-budget.stdout` byte-for-byte, including Analysis size 39840. `make -n -f Makefile.gnu test-record-map-observation` retained the actual prerequisite plan; no broad provider build was run. The dedicated target is also a test-units prerequisite; aggregate execution remains a separate gate.

## Boundaries and measured observations

Runtime owner remains 818512 bytes; copied observation is 32 and identity comparison is 16. Queries allocate and retain nothing. Measured logical-work examples: observe 33; string slice 36, empty copy 34, exact control-byte payload 40; simple map lookup 229; same-identity 430; collision lookup 34950. Each corresponding one-below control preserves output on LIMIT. Work headers, alias/range overflow, idle/thread state, topology corruption and selected nominal/depth/leaf constraints fail closed. Identity exposes only a Boolean and epoch, never a pointer or reusable trust token.

All five primitive leaf kinds at depths 1–3 pass selected-path controls. The largest constructed valid graph has 60 records, three arrays and one map (64 origins), with scalar path length 63. Syntactic path length 259 refuses; no claim that a valid 258-edge graph exists. An unbound empty root ARRAY does not invent a nominal field depth. A field-selected wrong-depth empty ARRAY refuses, while an unrelated malformed sibling is not recursively traversed without charge.

Four original programs execute and their copied result zero is observed after original module/source-plan destruction. Detailed map/alias/collision observations use explicit synthetic private fixtures. Failed later execution preserves the prior result identity but aliased global mutation remains visible; epoch advances and no transactional rollback is claimed. Exact NUL, quote, backslash, UTF-8 and newline bytes and FLOAT negative zero are preserved.

## Retained first failures

`selected.log`: test harness compile failed because vm_array_new lacked its capacity argument; corrected to zero with no production change. `final.log`: the first empty-root fixture lacked existing nested ARRAY authority and correctly refused declared array shape. The corrected fixture consumes an already-admitted depth-three tree before returning the empty root, preserving the original proof limits. Both failures remain distinct from the final passing gate.

Prior runtime count 2714 matches the final component and aggregate: removal of a redundant CHECK(argc>1), not allocator behavior, explains the earlier intermediate 2715 count. All four originals remain supplied explicitly here.
