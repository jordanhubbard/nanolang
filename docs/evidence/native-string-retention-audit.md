# My native temporary-string lifetime audit

I inspected current native translation source at `e55855ba` and retained generated compiler C from code pin `8666cb09`. I did not execute a fault fixture, repeat the full compiler run or change production code. MAC `task_4d3105329e73454083846ad41473500d` tracks the proposed repair.

## What my source establishes

- In `src/nanoisa/nvm2c.c:4083`, `emit_nstr_storage` emits an append-only `nstr_owners` allocation list. `nstr_allocate` checks size overflow and allocation failure, but neither marks nor releases earlier allocations.
- `emit_nstr_concat`, `emit_nstr_substr`, numeric string formatting and the character adapter allocate through that pool. Each concatenation allocates the entire new result. Repeated replacement can therefore retain the sum of all intermediate result sizes, even when only the final result remains reachable.
- `nstr_release_owned` is emitted only in generated entry cleanup, after the entry function returns (`nvm2c.c:5471`). Ordinary function exit releases invocation storage and root-list metadata, not strings.
- `emit_map_roots` publishes locals and live operand slots around calls and loop/tail safepoints (`nvm2c.c:2151`). It is conditional on `has_maps`; string-only modules do not obtain this registration.
- `src/nanoisa/nvm2c_map_roots.inc` already follows string edges through tagged values, string arrays and nested records, but its marking and sweeping phases cover only `nmap_owned` and copied map-get `nvalue_owned` allocations. `nstr_owners` is absent. Existing map allocation debt therefore cannot reclaim the general string pool.
- Host snapshots (`emit_scalar_artifact_adapters`), `nhost_file_read` and path helpers also return separately allocated buffers. Their provenance differs from borrowed environment or artifact pointers; a general string free must not assume all `const char *` values are owned.

This is a static process-lifetime retention finding. It does not measure the share of the separately recorded 48 GiB compiler attempt attributable to strings. Record snapshots and array owners also have entry cleanup; their costs remain separate from this first slice.

## My proposed bounded repair

I first make the existing root publication/tracing infrastructure available to generated modules that allocate owned strings, even when they contain no maps. I preserve caller frames, globals, current mutable aggregate edges and only live operand slots. I add owned-string membership and marking without dereferencing a guessed allocation header before arbitrary borrowed pointers.

I accumulate checked allocated-byte debt and reclaim unreachable string owners only at safe points after fresh root publication. I do not collect inside `nstr_allocate`: its concatenation or substring inputs and unpublished return values may still be needed there. A call-boundary safe point is needed as well as loop/self-tail points to bound repeated allocation in ordinary call sequences. I preserve returned values until the caller has registered them and preserve simultaneous self-tail argument staging. Existing map debt and lifetime guarantees remain intact.

I first cover concat/substr/format/from-char owners. Host snapshots and file/path results require an explicit adapter-by-adapter ownership inventory before adoption; borrowed pointers and foreign runtime ownership remain unchanged. Allocation failure must retain the existing explicit failure behavior, without freeing live objects.

My acceptance cases should include normal large concat churn with bounded live bytes, string-only modules, caller operands, returned strings, globals, overwritten locals, arrays and nested records, tagged values, mutation between safepoints, self-tail replacement and host-call boundaries. I require final owner cleanup and existing map/array/string sanitizer gates. Bounded live-owner metrics are necessary because an exit-clean LeakSanitizer result alone does not establish bounded retention during execution.

## Existing task reconciliation

`task_93bb44374587a757753418fc28c2095d` concerns C-runtime `dyn_array_push_string_copy` allocating `strdup` payloads while `GC_TYPE_ARRAY` destroys only the array buffer. Its escaped-element ownership policy is not this generated native AOT string pool, and remains open. Completed character cleanup task `task_d2b7c2616e2148a1871c25e1a7ac127d` establishes exit cleanup, not early reclamation. I do not reopen or overstate that completed bounded fix.
