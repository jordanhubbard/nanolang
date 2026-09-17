# My native record-array growth

MAC `task_617006c46f9746da9694c6a4e0a0ceaf` records the fixed 256-record limit reached by the native compiler while tokenizing its own source. The retained `record_array_growth_257.nasm` fixture passes in my VM and aborts in unchanged native code. The preceding collection-debt repair emits byte-identical C for this map-free fixture, isolating the storage limit from collector scheduling.

I now keep a stable array handle with a separately allocated buffer of by-value records. A private owner tracks the buffer capacity and whether it owns the handle. Growth checks element-count multiplication, preserves the original buffer until `realloc` succeeds, and copies borrowed backing storage before growing it. Cleanup frees each owned buffer, owned handle and owner node. I retain shallow record-value semantics and trace current array elements after buffer relocation.

I reject missing backing storage before record reads, record writes and root traversal. I retain record width/kind checks, tagged-array guards and the existing missing-read policy boundaries. This changes my generated C representation; it does not change NanoISA bytecode or publish a new foreign record-array ABI.

Four focused regressions run under ASan, UBSan and leak detection:

- I append 257 records through one handle and read preserved boundary values through its alias.
- I grow an array to 600 records containing owned string values while intervening collections trace those edges.
- I grow a stack-backed borrowed array by copying its elements, preserving its handle alias and leaving the original stack storage intact.
- I inject `realloc` failure and request an overflowing capacity. Both stop before changing the old pointer, capacity, length or eight existing values; the failure harness then releases the retained allocation and checks for leaks.

My existing native translator suite passed 2386 checks and shape constraints passed 1092 checks. I adjusted one C harness that previously relied on inline record storage to supply an explicit backing array; its value/tag assertions remain intact.

I also translated the exact pinned compiler bytecode used by the failed run into a separate native executable. Its unchanged-source run gets beyond the former `nrarr_push` limit and continues tokenization. The original executables and failure stack remain intact. This progress does not yet prove completed self-compilation or byte-identical bootstrap convergence.

Local evidence: `/tmp/nanolang-native-record-growth-{focused,native,integration}.log`, `/tmp/nanolang-bootstrap-8d769-stage1-growth-progress.log`, and `/tmp/nanolang-bootstrap-8d769-stage1-debt-stack.log`. The focused regression is `tests/test_native_record_growth.py` in `make test-one-ir-compiler`.

The progressed full-source run later faults while parsing. A smaller independent control isolates that boundary: depth-4 and depth-12 prefix additions compile with both the unchanged fixed-capacity compiler and this growth compiler; depth-24 and depth-40 crash both, with fewer than 256 tokens. A diagnostic child process with a larger stack lets the unchanged compiler finish depth 24. I recorded native stack use as `task_4c5aeade9fa944dd80eebfbd1ae91072` and preserve the normal stack limit in acceptance. No completed native self-compilation claim follows from this capacity repair. The comparison is retained in `/tmp/nanolang-native-parser-depth-comparison.log`.

My first broad run passed 44 of 46 methods, including both compiler-product gates. Two returned-array checks exposed an allocation-counter harness assumption: it counted `malloc`/`calloc` and `free`, but the new payload allocation uses `realloc(NULL, ...)`. I added `realloc` accounting while retaining the zero-live-allocation assertion and sanitizer leak checks; a non-null resize preserves the allocation count. Final corrected checks are recorded separately.

My corrected final run passes all 50 methods in 583.535 seconds, including both compiler-product gates, the four growth sanitizer methods, returned allocation teardown, mutable root tracing and collection scheduling. The final native suite also passes all 2386 checks. Logs: `/tmp/nanolang-native-record-growth-final-suite.log` and `/tmp/nanolang-native-record-growth-final-native.log`. The independent native parser stack task remains open.

After a clean rebase through the allocation-harness repair, I rebuilt the translator with `make nvm2c` and reran all four growth methods plus returned-allocation teardown: five methods pass in 0.939 seconds. The product/runtime diff is unchanged by that rebase.
