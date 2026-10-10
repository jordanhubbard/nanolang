# My mixed WebSocket source integration

I extend the runtime checkpoint at `8d4a0d33d` under #990 to both source
producers and product routes. I admit mixed File/TCP/WebSocket and repeated
WebSocket instances. My C producer writes canonical absent import slots;
my Nano producer advances import offsets by each catalog's actual method
count, retaining the absent fifth WebSocket slot. File/TCP bytes retain their
existing ordering.

I parse WebSocket connection permission independently from File/TCP permission
and create explicit copied mixed grants. Selected shadows, in-process product
execution, raw `nano_vm --services` execution and generated native launchers
carry separate connection/lookup/helper options. I retain the File/TCP-only
shadow wrapper. Native products obtain fresh authority at invocation; compilation
does not embed its permissions or resolver helper.

My paired source test passes WebSocket/File/TCP/WebSocket and five-WebSocket
programs, with direct and indirect Message factories, embedded-NUL literals,
borrowed send/receive bodies and consuming close. I compare exact C-produced
bytes with the independent Nano lowerer executing in NanoVM and LLVM native
with ASan/UBSan. Actual non-network execution performs File acquisition/close
and WebSocket hostname lookup denial. It does not exercise successful network
branches.

My initial paired fixture stops at the C test harness's eight-allocation prefix
limit. Repeated decoded string literals require more allocations. I retain the
failure and measure the successful lowering's allocation count, then inject
every prefix through recovery while preserving prior output. The corrected
paired test passes without changing its source assertions.

My corrected source suite passes four non-network methods; its required live
method fails at localhost bind with EPERM. The additional resolver-policy method
passes: compilation with explicit lookup/helper options never invokes the
helper, invocation without lookup denies before the helper, and explicit VM
and native invocation reaches the chosen helper once for each WebSocket.
The helper deliberately exits without resolving; this checks policy propagation,
not successful DNS traffic.

I also compile the complete current `src_nano/nanoc_v06.nano` into NanoISA,
translate it with `nvm2c`, and compile that generated compiler with LLVM
ASan/UBSan. C-seed, NanoVirt, this fresh compiler in NanoVM and its native form
all pass the mixed/five-instance product permission matrix with identical wire
bytes. All four routes pass nonexecuting publication, failing-shadow output
preservation and invocation-only resolver selection. This is fresh compiler
source qualification, not a Stage 1/Stage 2 fixed-point claim. My four-route
product run takes 550.444 seconds: four methods pass and only the required
live-peer method fails at bind with EPERM.

My five mixed flow/grant methods pass with LLVM sanitizer and leak checks,
including 35,841 linked and 37,462 instrumented WebSocket checks. My shadow
supervisor controls pass. Seven adjacent methods pass: the resolver-policy
method, five standalone WebSocket driver methods and repeated File/TCP product
permissions. GCC 16 compiles the changed policy, product and shadow sources
under C99 with strict warnings; this is compilation evidence, not GCC leak
qualification.

I retain command results, initial failures and source/generated-artifact hashes
beside this report. My live test remains registered in the ordinary unit and
compiler-generation gates. It requires numeric and DNS-host lifecycles for
mixed and five-WebSocket programs, selected dependency/root shadows, VM/native
execution and exact echoed-message counts. This environment cannot run that
acceptance test. Final Linux/Darwin qualification and the historical full native
reader/GCC leak incidents remain open; I do not close #990 or claim release 5.1.
