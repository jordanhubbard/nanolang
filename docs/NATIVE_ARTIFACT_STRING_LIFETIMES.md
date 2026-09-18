# Native artifact string lifetimes

I define an opt-in artifact-result contract for MAC
`task_bdc323f270d44f02b38ba728f1f93184`. My direct VM bridge and native C
adapters implement this bounded contract; the rest of the host-result audit
remains under `task_d5f899966241452a900422938fff3265`.

## Existing boundaries

At main `add46a66`, my filesystem artifact `modules/std/fs.c:file_read`
returns the allocation from `nl_read_file_text`, or a borrowed empty literal
when allocation fails. Successful empty input, missing files and rejected
embedded NUL text can all produce allocated empty strings. Text contents
cannot determine ownership.

My evaluator copies this shared reader result and frees its temporary in
`src/eval/eval_io.c`. My native builtin adapter also consumes its exact
reader allocation. Neither establishes ownership of a same-named function
selected from another artifact.

My generic NanoVM marshaler copies artifact strings into the VM heap without
freeing the foreign pointer. My generic C artifact adapter retains that
pointer under its existing borrowed-result contract. The facade snapshot
adapter is a separate declared contract. Ordinary compiled C callers of
`file_read` also receive the legacy non-NULL result; replacing its allocation
failure fallback with NULL would change that interface.

## Explicit cleanup contract

I recognize a versioned, optional companion export for each admitted
string-returning function:

```c
void file_read__nano_string_release_v1(const char *result);
```

Presence of this companion explicitly authorizes cleanup through the provider.
I resolve it through the selected artifact and check that it belongs to the
same loaded image as the called function, following my existing native array
ABI declaration boundary. A missing companion preserves the legacy contract.
I do not infer ownership from the filename, symbol spelling or returned text.

An admitted consumer snapshots the result, then calls the companion exactly
once, before publishing the snapshot. It also calls the companion if copying
fails. The provider handles its own allocator and any borrowed fallback
sentinel; consumers never call free on the foreign result. NULL cleanup is
harmless, while the consumer retains its declared result-validation policy.
The contract permits no retained access to the original pointer after cleanup.

The filesystem provider preserves its legacy non-NULL file_read interface
using a private stable empty sentinel and releasing only its own allocations.
This is an alternative to changing the public result to owned-or-NULL. Existing
C callers remain compatible; consumers that do not opt in retain their
existing lifetime behavior.

## Admission and acceptance

My direct VM bridge admits zero, one or two declared string parameters and a
string result. It refuses a cleanup-bearing artifact with another signature
before calling the provider. Native admission retains its existing exact
artifact signature table. No serialized NanoISA import format changes.

Both consumers require a non-NULL string result when a companion is present.
They call cleanup exactly once even on a NULL result or failed snapshot; the
VM reports failure and native C retains its existing abort-on-allocation-failure
policy. An interned VM string still consumes the provider result. Native copies
enter the existing traced string pool, so aliases survive until their roots
are gone. The loader retains its ordinary module lifetime; this adds no
asynchronous calls or detached cleanup.

My focused acceptance checks successful and empty text, missing files, rejected
embedded NUL, provider allocation failure, VM/native copy failure, escaping
aliases and repeated calls. An artifact without a companion still returns a
borrowed literal safely. A companion from another image is refused before the
provider is invoked. Two libraries with the same symbol retain distinct
cleanup identities. The native programs run with ASan/UBSan/LSan.

Callback, co-process and interpreter artifact consumers are not enrolled by
this change. Legacy C callers retain the original interface and can explicitly
call the companion after copying escaping text. My six path providers (`path_normalize`, `path_canonical`, `path_join`,
`path_basename`, `path_dirname`, `path_relpath`) also export this companion.
Their results are provider-owned allocations or NULL, so their companions
release with the provider allocator and accept NULL. Their existing path
semantics and result-validation policy are unchanged. Other provider APIs
still require an explicit ownership audit before enrollment.

Exact measured checks are recorded in my [implementation evidence](evidence/artifact-string-cleanup.md).
