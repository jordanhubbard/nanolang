# My immutable compiler-input bridge

I add an explicit owning snapshot context and six exact artifact signatures in
9a51b6382. Its [bootstrap manifest](bootstrap-9a51b6382/manifest.json) retains all
seventeen successful steps. Raw Stage1/Stage2 module bytes compare equal and all
manifest source hashes match when collected. I qualify that bootstrap pin, not
the subsequent native/VM adapter changes or a final release candidate.

My follow-up admits the same exact signatures in nvm2c, exports provider-owned
string cleanup, and checks opaque/int/int string calls in NanoVM. I retain first
failures: incorrect fixture argument counts, initially unregistered source and
AOT signatures, missing release companion, and the VM cleanup signature limit.
I retain those terminals rather than claiming the initial probes passed.

The corrected [thirteen-method suite](paired-and-neighbors.log) passes with
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`. It compiles the actual
module through C seed native, nano_virt, Stage1 and Stage2. Each bytecode product
is verified, executed in NanoVM, translated by nvm2c and executed as native C
with address/undefined sanitizers and leak detection. Copied raw/interface/source
views outlive their context; independent contexts and full counted acquisition
are checked. Six wrong artifact signatures refuse without replacing prior output.

Provider cleanup neighbors preserve owned/borrowed results, originating-library
identity, null refusals and allocation-failure cleanup. The new indexed-context
probe passes a null context, negative index and INT64_MAX exactly, counts one
release in both runtimes, and counts one release when native copying fails.
I fix the older test harness to honor its selected compiler and link the current
File CLI/runtime into its fault-injection VM; no assertion or sanitizer is removed.

My [final reader/bridge fixture](reader-final-sanitizer.log) passes with fresh
LLVM ASan/UBSan and leak detection, including full-byte copied-view comparison
after context destruction. This fixture instruments the reader/bridge and strict
preparer closure. The paired native test instruments generated native code, not
every dynamic module provider; I do not conflate those scopes.

The actual source drivers still do not acquire companions. Complete namespace
resolution, independent File lowering, driver lifetime/budget accounting and full
Linux/source/release acceptance remain required under #989/#976. This bridge
supplies compiler input data, not service execution authority.
