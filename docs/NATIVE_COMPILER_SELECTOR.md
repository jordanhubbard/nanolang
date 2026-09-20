# My aggregate native compiler selection

My projected record-array global and tuple-value tests now honor
`NANO_NATIVE_TEST_CC`, using the same explicit compiler selection as my other
native tests. When it is unset, I retain `cc`. I preserve every assertion,
sanitizer flag and original 120-second subprocess deadline.

I reproduced both projected native timeouts with Apple Clang on Darwin. A live
sample places the store-first process inside recursive ASan initialization
during dyld/malloc startup, before main. A separate minimal executable also
times out under Apple ASan/UBSan, while Homebrew Clang enters main with the same
source and flags. That diagnostic uses a separate 10-second limit. I retain
these failures; I cannot attribute the older, deleted binaries retrospectively.

At fixture pin `3ef5d2261826a55941e399a155cb5bb178b821c3`, both methods pass on
Linux with default `cc` and on Darwin with
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`. Together they execute eight
generated native binaries successfully, including both function orders and both
tuple producers. I reuse the previously qualified File integration compiler
providers; the tuple emitter and comparator are freshly built against those
inventoried providers. This is selected-method acceptance, not a fresh canonical
bootstrap or the complete 90-method suite.

My [independent audit](evidence/native-compiler-selector/nanolang-native-selector-independent-audit.json)
rehashes both Darwin archives and 412 manifest-listed files across six report
groups, verifies equal input/provider endpoints, and mechanically reverses only
the selector/import edits to recover the prior fixtures exactly. Darwin tuple
execution retains its existing `detect_leaks=0`; I claim no tuple LSan result
there. My [report seal](evidence/native-compiler-selector/seal.json) retains
commands, statuses, diagnostics, source/tool identities and qualification drivers.
My [archive identities](evidence/native-compiler-selector/archives.json) identify
the retained generated binaries and other products under `/tmp` on this host;
the two Darwin archives are also retained on puck. They are local evidence, not
published release assets.

I still require the fresh complete suite after the canonical admission and
constructor repairs merge. Task `task_2f52721aac374ac592b61438315dc981` remains
open for that integration acceptance, and my 5.1 publication remains held.
