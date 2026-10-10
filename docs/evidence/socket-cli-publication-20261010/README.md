# Explicit TCP source publication and invocation

I connect my checked TCP source bytes to the existing staged publication and
supervised shadow path. C and Nano source drivers accept
`--allow-tcp-connections`; File authority remains independent. I select the
catalog from decoded main bytes and require the selected public emitter and
execution consumer to validate each module independently. I grant no network
execution merely because bytes decode.

I create a fresh grant per selected shadow and per invocation. My supervisor
retains its whole-suite deadline, durable selection/start/done records and
process-group cleanup. The old File API selects catalog 1; the new catalog
adapter can select File or TCP. All catalog-specific calls retain their own
typed options, reports and grants.

I require separate authorization for generated launchers and VM invocation.
`nano_vm --allow-tcp-connections --socket-instruction-limit N` accepts explicit
fuel from zero through 1,000,000 and refuses conflicting modes. My nonexecuting
`nvm2c --socket-tcp --entry-name IDENT` path validates before staged output.
I document these interfaces in `docs/SOCKET_HOST_API.md`.

I preserve failures rather than erase them: test discovery imported unrelated
TestCase classes, a File test expected an older catalog-specific diagnostic,
my first native Nano driver link omitted its host runtime object, and I changed
File shadow-failure wording before restoring its existing diagnostic. The
native driver uses the same runtime support object as my bootstrap recipe.

One Nano VM root-only TCP shadow invocation returned supervisor SYSTEM after
SELECT without START. The exact command passed on bounded replay. I retain
both reports and add child setup/exit/report diagnostics; I do not claim a
root cause or a fix for that historical incident. It remains tracked under
#990. I require a complete passing TCP corpus before accepting this batch.

My complete C File driver corpus passes 16 tests in 111.648 seconds. Three Nano
File tests pass in 49.637 seconds through VM and native driver forms: generated
shadows/native authority, dependency/root-only selection and emission without
selected shadows. The File/TCP supervisor fault corpus and byte-only bridge
allocation tests also pass.

These are local Darwin checks of one catalog per source graph. I have not
qualified a fresh Stage 1/Stage 2 bootstrap, mixed File/TCP execution, DNS,
WebSocket, the complete platform matrix or release publication.

My final TCP corpus passes all three methods in 51.690 seconds and retains
154 CLI command reports. Every method exercises nanoc_c, nano_virt, the
Nano driver in NanoVM, and its separately linked native executable. IPv4/IPv6
outputs are byte-identical across all four drivers. I retain exact sources,
bytecode, emitted C, invocation arguments, expected/actual statuses and stdout/
stderr in `tcp-cli-artifacts.tar.gz`. Ports are specific to that run; the test
starts fresh listeners. The previously failing root-only case passes in this
full corpus, with the new diagnostics enabled. The historical incident remains
unexplained; I do not convert a passing replay into a root-cause claim.

I build the local Nano driver with nano_virt, translate with nvm2c, and link
its generated C with `bin/nano_aot_runtime.o -lm` using LLVM clang at O2. The
retained build log records the exact commands. I run the TCP corpus with
`NANO_TCP_DRIVER_MODULE` and `NANO_TCP_DRIVER_NATIVE` selecting those artifacts,
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`, and
`NANO_SERVICE_DRIVER_RETAIN=1`. The permanent `test-nano-service-driver` target
also runs this corpus using its bootstrap-produced driver. I have not run that
fresh bootstrap gate in this batch.
