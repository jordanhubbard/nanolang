# My admitted native host-result inventory

I audited `src/nanoisa/nvm2c.c` at main
`308235c8f9033ed547664f9e77fa3cdcb02a11c3`, after PR553, PR583 and PR589.
This closes the stated native string-result scope of
`task_d5f899966241452a900422938fff3265`: I track adapter-owned strings, snapshot
transient borrowed facade results, preserve foreign allocator boundaries, and
retain bounded lifetime/escape tests with sanitizer cleanup evidence. I do not
claim that every host API or every consumer implements this ownership model.

## Builtin namespace

I match the exact builtin namespace and signature in `import_host`; I do not
assign ownership from a foreign library's symbol spelling. The following
inventory includes every entry in `host_adapters` at the audited pin. Aliases
share one emitted helper and its allocation policy.

| Emitted helper | Admitted symbols | Result storage |
| --- | --- | --- |
| `nhost_strlen` | `strlen` | I return a scalar value; no result buffer is transferred. |
| `atan` | `atan` | I return a scalar value; no result buffer is transferred. |
| `nhost_getcwd` | `vm_getcwd` | I copy the stack buffer into my string pool. |
| `nhost_getenv` | `vm_getenv`, `nl_os_getenv` | I copy borrowed environment text into my string pool. |
| `nhost_tmp_dir` | `vm_tmp_dir` | I copy borrowed environment/default text into my string pool. |
| `nhost_argc` | `get_argc` | I return a scalar value; no result buffer is transferred. |
| `nhost_argv` | `get_argv` | I copy valid argv text into my string pool; an invalid index returns a borrowed empty literal. |
| `nhost_file_read` | `file_read`, `vm_file_read`, `nl_os_file_read` | I copy the owned read buffer into my string pool, then free the original through nstr_take. |
| `nhost_file_write` | `file_write`, `vm_file_write`, `nl_os_file_write` | I return a scalar value; no result buffer is transferred. |
| `nhost_file_exists` | `file_exists`, `vm_file_exists`, `nl_os_file_exists` | I return a scalar value; no result buffer is transferred. |
| `nhost_dir_exists` | `dir_exists`, `vm_dir_exists`, `nl_os_dir_exists` | I return a scalar value; no result buffer is transferred. |
| `nhost_remove` | `file_delete`, `file_remove`, `nl_os_file_delete`, `nl_os_file_remove` | I return a scalar value; no result buffer is transferred. |
| `nhost_rename` | `file_rename`, `nl_os_file_rename` | I return a scalar value; no result buffer is transferred. |
| `nhost_identity` | `file_compare_identity` | I return a scalar value; no result buffer is transferred. |
| `nhost_destinations` | `file_compare_destinations` | I return a scalar value; no result buffer is transferred. |
| `nhost_normalize` | `path_normalize`, `nl_os_path_normalize` | I allocate output directly in my string pool and free temporary index storage. |
| `nhost_shell` | `nl_exec_shell` | I return a scalar value; no result buffer is transferred. |
| `nhost_capture` | `nl_exec_capture` | I copy the owned capture buffer into my string pool, then free the original through nstr_take. |
| `nhost_is_digit` | `vm_is_digit` | I return a scalar value; no result buffer is transferred. |
| `nhost_is_alpha` | `vm_is_alpha` | I return a scalar value; no result buffer is transferred. |
| `nhost_is_alnum` | `vm_is_alnum` | I return a scalar value; no result buffer is transferred. |
| `nhost_is_space` | `vm_is_space` | I return a scalar value; no result buffer is transferred. |
| `nhost_is_upper` | `vm_is_upper` | I return a scalar value; no result buffer is transferred. |
| `nhost_is_lower` | `vm_is_lower` | I return a scalar value; no result buffer is transferred. |
| `nhost_is_whitespace` | `vm_is_whitespace` | I return a scalar value; no result buffer is transferred. |
| `nhost_digit_value` | `vm_digit_value` | I return a scalar value; no result buffer is transferred. |
| `nhost_from_char` | `vm_string_from_char`, `string_from_char` | I allocate directly in my string pool. |
| `nhost_mktemp_dir` | `vm_mktemp_dir` | I allocate the mutable template directly in my string pool; directory removal is separate. |

`nstr_copy`, `nstr_take` and `nstr_copy_release` use the checked string pool.
The emitted module enables string-root support for every admitted string-return
import, including modules without map instructions (`nvm2c.c`, owned-runtime
selection near line 5343). Existing frame/global/aggregate roots and collection
safe points retain escaping aliases. Pool byte measurements describe retained
owned storage, not instantaneous copy overlap, RSS or filesystem resources.

## Artifact namespace

I admit the following complete `artifact_adapters` table. Exact path, import
kind, arity and type checks remain in `import_host`. I resolve an optional
`${symbol}__nano_string_release_v1` companion from the function's image and
check image identity with `dladdr`. I copy before invoking provider cleanup,
including copy-allocation failure; I never infer malloc ownership from a name.

| Admitted symbols | Result storage |
| --- | --- |
| `nlc_module_artifact` | I snapshot borrowed compiler-support TLS storage into my string pool; the provider owns its original. |
| `nl_nanoisa_load_print`, `nl_nanoisa_load_pretty` | I snapshot borrowed provider last-output storage into my string pool. This storage is process-global, not TLS, and the next facade output replaces it. |
| `nl_nanoisa_last_error` | I copy the borrowed static error buffer into my string pool. |
| `path_normalize`, `path_canonical`, `path_join`, `path_basename`, `path_dirname`, `path_relpath` | My real filesystem providers return owned-or-NULL results and export release companions. I snapshot, release through that provider, and retain existing NULL refusal. |
| `file_read` | My real filesystem provider exports a release companion which distinguishes owned text from its private allocation-failure sentinel. I preserve the non-NULL reader contract. |
| `fs_walkdir` | I use the separate array ABI: copy each string into tracked nsarr storage, then invoke fs_walkdir_release on the foreign array. This is not a scalar string companion. |
| `nl_nanoisa_assemble_save`, `nl_nanoisa_assemble_text_save`, `file_write`, `file_append`, `file_exists`, `file_delete`, `fs_mkdir_p`, `file_copy`, `dir_copy`, `file_compare_identity`, `file_compare_destinations` | I return a scalar value; no result buffer is transferred. |

I account for all 43 builtin entries and all 23 artifact entries
(11 string, 11 scalar and one array result). An artifact without
a cleanup companion retains its existing borrowed contract. Facade snapshot
adapters still copy those borrowed results. Other borrowed artifact strings
must be independent of argument storage and remain valid for the admitted
contract; general interior-pointer results are not admitted.

The native string adapters live in `emit_scalar_artifact_adapters` near line
4972; the independent array adapter is `emit_walk_adapters` near line 5014.
My provider declarations and implementations are in `modules/std/fs.h` and
`modules/std/fs.c`; the seven cleanup companions begin near fs.c line 238.
Borrowed facade storage is defined in `modules/compiler_support/compiler_support.c`
and `modules/nanoisa/nanoisa.c`. In particular, I do not transfer ownership of
the latter's `last_output` or static `last_error` to a consumer.

## Retained acceptance evidence

I reuse measured evidence from the merged implementations; this inventory
changes documentation only and does not rerun a compiler acceptance workload.

- [PR553 builtin/facade evidence](native-host-string-ownership.md): no-map
  repeated argv/environment/tmp/cwd results, path churn, file/capture/temp and
  facade copies, escaping aliases, and sanitizer final cleanup. Both unchanged
  calculator producer artifacts pass GCC ASan/UBSan/LSan and strict Clang;
  the earlier two-byte argv-copy leak is resolved history.
- [PR583 artifact protocol evidence](artifact-string-cleanup.md): twelve
  combined GCC/Clang methods, provider allocation/release counts, repeated
  intern hits, distinct images with the same symbol, borrowed results,
  same-image refusal, real file cases, and isolated copy-failure cleanup in
  both VM and native consumers. Existing native/shape and VM FFI gates pass.
- [PR589 filesystem provider evidence](filesystem-path-string-cleanup.md):
  fourteen combined methods pass in 15.134 seconds; six real providers retain
  aliases across 2,000 normal/empty rounds. Both new methods pass strict Clang
  in 2.092 seconds. The strengthened six-provider NULL control passes in
  7.722 seconds without sanitizer diagnostics. Generated native programs use
  ASan/UBSan/LSan; the ordinary VM CLI is not sanitizer-instrumented.

I found no remaining untracked owned scalar string result in this admitted
native inventory. That conclusion satisfies d5's exact adoption/borrowing and
bounded acceptance criteria. I leave arbitrary new artifact contracts,
callback/co-process/interpreter enrollment, array/map ownership, C-runtime
array-copy task `task_93bb44374587a757753418fc28c2095d`, concurrency of provider
storage, and full product/release acceptance outside this completion. I do not
attribute historical compiler failures or total process memory to these buffers.
