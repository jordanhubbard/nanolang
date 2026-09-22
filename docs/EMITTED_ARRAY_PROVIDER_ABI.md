# My emitted array provider declarations

I retain hosted9386 run35755222839 userguide-export failures under task_7e9bb711c28f42f3abf1b6ae71e52a9c. The selected provider is the generated static nl_os_walkdir, not the module export fs_walkdir. Only the latter had an ABI2 declaration. I do not make a missing declaration compatible.

I give each actual emitted helper a static descriptor containing its exact function pointer and the compiled DynArray version. Static descriptors cannot collide when root and dependency translation units each emit the helper bundle. The local guard verifies the selected pointer, version and defining image. My existing dynamic lookup guard remains unchanged for real foreign declarations. Process-run has a conditional provider: when the real process header supplies it, a NULL local descriptor deliberately routes to the original dynamic guard.

My name mapper carries an explicit local-provider result only from the actual builtin registry mapping branch and its finite emitted-helper catalog. User/qualified/alias declaration routes and generic/list specialization routes do not acquire this result from a matching C spelling. A captured lexical callable clears the function metadata, so it cannot borrow this authority. I retain the mapping result before argument lowering.

I audited every registry row containing an ARRAY parameter or result:

| Registry names | Selected provider boundary |
| --- | --- |
| str_split, str_join | Actual static nl_str_split/nl_str_join have local declarations; source-declared str_split keeps its earlier exact-declaration route. |
| bytes_from_string, string_from_bytes | Actual static byte-array conversion helpers have local declarations. |
| array_slice, array_sort, array_reverse, array_contains, array_index_of | Actual static helpers have local declarations; specialized builtin lowering and selected source declarations retain their existing routing. |
| fs_walkdir, file_read_bytes | Actual static DynArray wrappers have local declarations. |
| process_run | Static wrapper has a local declaration only without the real process header; imported provider retains dynamic ABI admission. |
| dir_list | Existing ARRAY registry versus generated char* text mismatch is separately open as task_be61e571e5a24cdaa77d80a6a7e80912. I attach no array declaration to it. |
| array_length, array_new, array_set, at, array_get, array_push, array_pop, array_remove_at, array_concat, array_map, array_filter, array_fold, filter, map, reduce | Typed/specialized lowering, runtime functions or actual selected declarations; no new static-provider permission. Genuine foreign references keep dynamic admission. |
| hashmap_keys, hashmap_values, map_keys, map_values | Runtime/module provider references; no new static-provider permission. Genuine foreign references keep dynamic admission. |

My additive controls check matching static provider admission, wrong function and wrong version refusals, retain every original missing/foreign-image/version refusal, and compile/run the actual imported walkdir wrapper with an exact one-file result. That compilation also links independently emitted root/module helper bundles. Source review precedes execution. Full hosted userguide-export and the independent dir_list repair remain separate obligations.
