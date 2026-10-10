# My mixed checked execution checkpoint

I build on `070d60f9a` under #990. I connect the private mixed value carrier to
retained checked plans, shared frames, VM dispatch and independent generated C.
I do not expose public mixed host grants, source publication or installed mixed
packages in this checkpoint. My complete 5.1 scope remains required.

## Matched execution

My actual Make target passes on Darwin with Homebrew LLVM Clang:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-services-dispatch
```

I retain [the terminal](mixed-final.log), [67 build/run commands and their
outputs plus generated C](mixed-artifacts.tar.gz), [artifact locations](artifact-roots.json)
and [source hashes](sources.sha256). The final target passes in 70.854 seconds.
My mixed runtime, flow/nominal adapter, value cores, host adapters, capability
core, VM dispatcher, emitter and generated C use ASan/UBSan with leak detection.
The transport/compiler support objects use their ordinary repository build;
this is not whole-program instrumentation.

I execute fourteen VM cases and fourteen generated-native cases: both IPv4 and
IPv6, direct/indirect owned and borrowed helper calls, ordinary/permuted type
and import maps, assertion failure with live owners, and fuel limits zero and
45. Each successful program acquires two distinct File owners and one TCP
connection before it begins their operations. All three remain tracked while
helpers borrow and return them. File write/rewind/read and loopback TCP
finish/send/receive/close Results are consumed and checked. Native and VM
publish 42 only after clean cleanup; failure leaves 999 untouched. Connection
readiness loops remain bounded by the same invocation fuel.

I require eight observed NUL-byte transfers for each address family, with no
peer error. A trapped invocation and fuel exhaustion also drain outstanding
owners. I check native symbols to exclude VM execution and the mixed emitter entrypoint. I reuse
one native executable per program across fuel limits, after comparing the newly
emitted C bytes, instead of compiling that unchanged program again.

Two additional generated-provider controls substitute a different valid
instance import and falsify a retained import-map expectation. I require TYPE
and UNRESOLVED respectively, preserved output and the expected cleanup reports.
A `tmpfile` host probe observes zero File acquisitions in both refusals.
The wrong-map control refuses before core creation; the wrong-instance control
creates the checked context but refuses before resource acquisition.

My fixture also refuses truncated bytes and invalid option revisions without
acquisition or output mutation. The nonexecuting emitter preserves prior output
for truncated bytes. Two instrumented fixture runs cover all six runtime arena
allocation failures and all ten per-instance core-creation failures, followed
by successful preparation/creation. Destruction preserves output and leaves no
sanitizer-reported leak. The separate mixed lifetime gate covers all 193
allocation prefixes for a 64-instance core table.

GCC 16 accepts the final carrier, VM, emitter and allocation-enabled fixture
with C11, `-Wall -Wextra -Werror -fsyntax-only`.

## Retained failures and correction

My first mixed compile found one remaining direct `.borrow.epoch` access and
an emitter-only helper unused by the VM. I route the epoch through the instance
adapter and make the pure instance-count helper inline. GCC separately rejected
a misleadingly indented fixture loop; I separate its following assertion.

My [first fixture link](fixture-link-failure.log) omitted VM providers required
by retained transport-test functions. I add those providers only to the fixture;
standalone generated products continue to exclude VM execution and the mixed emitter entrypoint.

My [initial matched corpus](initial-mixed-pass.log) passes before adversarial
provider controls. The [wrong-instance baseline](wrong-instance-baseline.log)
then exposes a real boundary gap: substituting another valid acquisition import
is detected by the next checked frame state, after service dispatch, with STATE.
I match the requested service import and catalog method to the current checked
instruction before host effects. The corrected control requires TYPE and now
also observes zero `tmpfile` calls. I retain [the first corrected-import
pass](checked-import-pass.log) and the expanded final gate separately.

## Adjacent gates and remaining work

My [initial adjacent gate](initial-regressions.log) passes TCP dispatch, both
File indirect-dispatch methods, both mixed lifetime configurations (881 and
1,515 checks), and both mixed-flow configurations (12,184 and 11,123 checks).
It predates the final exact-instruction guard. My [final unchanged single-catalog dispatch gate](final-regressions.log)
passes after that guard: TCP takes 29.953 seconds and both File methods take
161.276 seconds. I retain their [commands, generated providers and results](adjacent-artifacts.tar.gz).

These checks establish private bytecode execution on this Darwin host. They do
not establish public per-instance grant enforcement, paired source lowering,
source shadows, installed packaging, DNS/WebSocket, Linux or exact-candidate
release qualification. Those requirements remain open under #990/#982/#976.
