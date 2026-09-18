# My builtin host-string ownership boundary

My bounded child `task_7f2b7373646341d5a317f374d302e390` adopts known builtin allocations and facade snapshot copies. At this child checkpoint, parent `task_d5f899966241452a900422938fff3265` remained open for artifact ownership. My later [complete native inventory](native-host-result-inventory.md) records the merged provider contracts and narrow parent acceptance.

| Adapter | Incoming ownership | My returned storage |
| --- | --- | --- |
| argv, getenv, tmp directory, current directory | Borrowed argv/environment/stack text | Checked string-pool copy; borrowed input untouched |
| Builtin lexical normalization | Scratch indices plus newly allocated result | Result allocated directly in the pool; indices freed normally |
| Builtin file read | Owned growable malloc/realloc buffer, including empty/failure result | Pool copy; exact temporary buffer freed once |
| Builtin shell capture | Owned fixed-capacity output buffer | Pool copy on success or pipe-open failure; exact temporary freed once |
| Builtin temporary directory | Newly allocated mutable template | Pool allocation before mkdtemp; directory filesystem lifetime unchanged |
| Character conversion | Existing pool-owned result | Unchanged; not adopted twice |
| Facade load-print/pretty/last-error and module-artifact snapshots | Borrowed transient/TLS result | Pool copy; original never freed |
| Generic artifact string result | Existing independent borrowed-result contract | Unchanged; no ownership inferred |

I enable the existing string/root runtime when retained imports allocate these strings, even with no map or allocating string opcodes. I preserve complete tracing and all existing published-root safe points. Neither allocation, transfer from a builtin temporary, nor return handoff collects. The pool is defined before host adapters, so exact helpers can allocate directly or copy and consume only their known malloc-owned temporary. Checked size/debt accounting and final cleanup remain the existing pool mechanism.

Pool byte bounds cover retained pooled storage, not transient overlap while copying a file/capture buffer or filesystem resource cleanup. Generic artifact functions may validly return literals: the existing identity fixture does this for path_basename. Actual filesystem file_read additionally returns owned text (including empty success) or a borrowed empty literal on allocation failure. The later task `task_bdc323f270d44f02b38ba728f1f93184` preserves both contracts through a same-image cleanup companion whose provider recognizes its private sentinel; it does not change file_read to owned-or-NULL. I do not free pointers based on contents, symbol spelling or library filenames.

## My bounded checks

Four new methods execute strict native C with ASan, UBSan and LeakSanitizer. They cover 5,000 iterations of argv/environment/tmp/cwd calls without allocating string/map opcodes, caller/returned/non-self-tail/global aliases, file reads with missing/empty/embedded-NUL inputs, capture and temporary-directory results, 10,000 lexical-normalization calls, and copied TLS snapshots surviving 5,000 later calls. The artifact control also returns a literal and must not free it. Every native run asserts zero final pool bytes/owners and a retained pool peak at most 70,000 bytes.

Three methods execute the same bytecode in VM and native. The raw path_normalize builtin name is not exported by the VM, although native translation accepts it; the initial failure is preserved in `/tmp/nanolang-host-adoption-tests.log`. Task `task_4c21884a6a8b417eb6b07b7f82c80d55` records this distinct resolution gap. I split normalization into an explicitly native-only ownership test; I did not add an alias or change VM behavior here.

The four methods pass, and 23 host/string/aggregate/map/returned-allocation methods pass in 36.666 seconds. Logs are `/tmp/nanolang-host-adoption-tests-split.log` and `/tmp/nanolang-host-adoption-regressions.log`. My native core gate passes 2,412 checks and 1,092 shape checks (`/tmp/nanolang-host-adoption-nvm2c.log`). After integrating main `a515f0e0` and correcting an optional helper reference, all eight host/string methods pass again (`/tmp/nanolang-host-adoption-final-focused.log`).

I tested both unchanged calculator artifacts with argument `5`: C-seed bytecode SHA-256 `e930a023200986405b5b9011678fe3e3325d217920c9391fe07dbd55b1449b02` and canonical bytecode `674d2819e8aa08646106622996b33e8eeb7ea84384b8a04ac67df3872e0eb51c`. Both strict GCC builds print `Result: 3.14159`, exit zero and pass ASan, UBSan and LeakSanitizer without the prior two-byte argv leak. Both strict Clang builds also execute successfully. I selected the installed GCC 13 support directory explicitly for this host's Clang; its default GCC installation warning otherwise precedes source compilation. Clang then identified an unused optional transfer helper, which I corrected with the same explicit generated references used by other optional helpers.

I retain calculator commands, hashes, output and sanitizer results under `/tmp/nanolang-host-calculator-merged/`. The initial missing `nano_aot_runtime.o` link evidence remains there separately; building the required runtime target resolved that prerequisite. I have not rerun native full self-compilation. The bounded child is ready for review; parent host ownership and the explicit artifact ABI follow-up remain open.
