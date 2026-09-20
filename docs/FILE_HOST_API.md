# I require a grant for the bounded File package

This source checkpoint implements the explicit byte APIs and package below. Its
joint public qualification is pending; the older private target and grant seals
do not establish installed/public acceptance. I accept only the exact catalog1,
nominal-v2, acyclic File profile, with zero-argument scalar entry and optional
zero-result VOID initializer. Generic VM/module APIs and unsupported consumers
retain their File refusal.

I build the explicit runtime with `make file-public-runtime`; `make install`
installs its archive and headers with nano_vm and nvm2c. I do not embed an
interpreter into generated File C. Generated functions use the checked native
arena and runtime, including a fresh exact-byte plan and ABI agreement on every
invocation.

```sh
nano_vm --allow-temporary-files program.nvm
nvm2c --file-temporary --entry-name example program.nvm -o example.c
cc -std=c99 -I"$PREFIX/include" example.c host.c \
  "$PREFIX/lib/libnano_file_runtime.a" -o example
```

An embedding host supplies its grant explicitly:

```c
#include <nanolang/file/nanoisa/file_public.h>
extern NvmFileRuntimeReport nvm_file_program_example(NvmFileHostGrant *, NvmFileScalar *);

int main(void) {
    NvmFileHostGrant *grant = NULL;
    NvmFileScalar value = {0};
    if (nvm_file_host_grant_create_temporary_files(&grant) != NVM_FILE_HOST_OK)
        return 1;
    NvmFileRuntimeReport report = nvm_file_program_example(grant, &value);
    NvmFileHostStatus cleanup = nvm_file_host_grant_destroy(&grant);
    return report.status != NVM_FILE_RUNTIME_OK || cleanup != NVM_FILE_HOST_OK;
}
```

No File or Result owner escapes this call. The public scalar is INT or canonical
BOOL. An unsuccessful invocation preserves prior output and reports the first
error/site plus secondary cleanup failures. Initializer cleanup finishes before
entry. Accepted close consumes the File even when its Result reports an error.
The host must inspect failure, not interpret an unchanged scalar as new output.

The grant permits only temporary-file acquisition, byte0..255 write, rewind,
read and consuming close. It grants no arbitrary filename, FFI import, dynamic
loader, callback, Socket or GPU operation. Creation opens no stream. Revoke is
idempotent; a revoked live grant cannot run. Destroy invalidates its pointer;
a copied dangling pointer is not a valid revocation test. A grant may be reused
for separate invocations, but live owners never cross invocations.

A single C11 gate in the linked owning runtime excludes public File grant/query/
execution intervals, including both VM and native. Contention or same-thread
reentry returns BUSY without changing the active call. Callers still owe ordinary
C pointer lifetimes and disjoint output storage, and external serialization when
mixing unrelated direct private query/core calls. No asynchronous, callback or
cross-runtime grant transport is admitted. Runtime detail headers are for trusted
generated C; native memory access is not a sandbox.

Each generated entry identifier has1..63 ASCII characters, starts with a letter
and continues with letters/digits/underscore. Its sole program export is
`nvm_file_program_<identifier>`. Two different identifiers can share the same
runtime and grant. Duplicate identifiers intentionally fail the link. Do not link
separate copies of the owning runtime into plugins and exchange grants between
them. No atomic layout appears in public or generated headers; only the owning
gate object needs C11 compilation.

The explicit VM CLI excludes repeat, debug, profiling, guest arguments, daemon,
COP, verify-only and shadow modes. Successful INT/BOOL becomes the low8-bit
process result; an execution or cleanup failure returns1 with diagnostics.
The C API preserves the full64-bit scalar. nvm2c's explicit File options request
nonexecuting translation only: the generated caller still needs its live grant.
Without these opt-ins the existing command paths continue to refuse File claims.

The File CLI reads at most16MiB of actual serialized input, separately from the
bounded runtime preparation arena. Native source emission retains its128MiB
output bound. An explicit `-o` stages in the destination directory and renames
only after checked write/flush/close; a new staged file uses mkstemp's private
permissions. It does not promise durable fsync publication or rollback of bytes
already observed on stdout. Input-file and generated-output I/O are CLI actions,
not guest service acquisition, and qualification counts them separately.

Loops, indirect calls, richer borrow sets, paired executable NSI source bindings
and complete selected-source shadows remain required follow-up acceptance.
This first public profile does not close the larger File or5.1 roadmap parents.


## I require a separate cyclic choice and explicit fuel

My new cyclic public source checkpoint is awaiting fixture qualification. The
existing acyclic forms above keep their profile and ABI. I do not retry a refused
acyclic program through cyclic execution. My cyclic wrappers use the same owning
grant and freshly verify every selected variant, edge, startup and direct call
before beginning effects.

```sh
nano_vm --allow-temporary-files --file-cyclic --file-instruction-limit 1000 program.nvm
nvm2c --file-temporary --file-cyclic --entry-name example program.nvm -o example.c
```

I expose `nvm_file_execute_cyclic_bytes` and `nvm2c_emit_file_cyclic_bytes` in
`<nanolang/file/nanoisa/file_cyclic_public.h>`. The generated export is
`nvm_file_cyclic_program_example(grant, options, scalar)`, returning
`NvmFileCyclicExecutionReport`. Every execution receives a non-NULL
`NvmFileCyclicOptions` with revision1 and an explicit unsigned limit0..1000000.
Zero starts no opcode; I never substitute a default. Native emission has no fuel
argument because each subsequent invocation supplies its own limit.

Initializer, entry and all direct callees share the counter. Exhaustion reports
LIMIT, the pending function/PC, the started count and `fuel_exhausted=true`.
Cleanup is not charged and still finishes before a failure returns. Other LIMIT
causes retain `fuel_exhausted=false`. Clean scalar publication follows complete
destruction; failure preserves output. BUSY precedes reading any caller argument
and consequently reports a zero supplied limit and zero starts. Other early
refusals retain the supplied numeric limit when valid option storage is present.

The existing runtime archive gains separate cyclic VM, emitter and ABI members.
Generated native C uses real functions and labels; its link must not extract a VM
engine. Fuel is an opcode budget, not a host-call deadline or security sandbox.
Closed-indirect calls, richer reference profiles and paired cyclic source
production remain required later work and are not enabled by this checkpoint.
The [complete contract](NANOISA_FILE_CYCLIC_PUBLIC_ADMISSION.md) records the exact
ABI, gate precedence, installed closure and pending Linux/Darwin acceptance.
