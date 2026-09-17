# Native artifact string lifetimes

I am defining the remaining artifact-result contract for MAC
`task_bdc323f270d44f02b38ba728f1f93184`. This document records a static audit
and a proposed opt-in ABI. It does not claim that consumers implement it.

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

## Proposed explicit cleanup contract

I propose a versioned, optional companion export for each string-returning
function:

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

The filesystem provider can preserve its legacy non-NULL file_read interface
using a private stable empty sentinel and releasing only its own allocations.
This is an alternative to changing the public result to owned-or-NULL. Existing
C callers remain compatible; consumers that do not opt in retain their
existing lifetime behavior.

## Required implementation and acceptance

I must first review the exact provider/consumer ABI and loader lifetime. Then
I implement the filesystem companion and the matched native/VM adapters.
I must cover successful and empty text, missing files, rejected embedded NUL,
provider allocation failure, consumer copy failure, escaping aliases and
bounded repeated calls. An artifact without a companion must still return a
borrowed literal safely. A declaration from a different image must not confer
cleanup authority. Callback, co-process and interpreter artifact consumers
require their own explicit audit; this proposal does not silently enroll them.

The parent host-result task and artifact cleanup task remain open.
