# My allocation-amortized collection candidate

Under #1161 I retain a private scheduling repair based on my production
packed-leaf collector. I preserve the complete trial-deletion graph, reference
counts, explicit collection and4096-suspect minimum. Automatic passes also
wait for allocation activity equal to one quarter of the preceding pass's
tracked retained bytes. Successful object/storage allocations revisit pending
pressure. My schedule lives after the root slots in their existing allocation;
heap and VM structure layouts remain unchanged. A completed pass can retain
a small empty root buffer carrying that schedule. I free it when a pass empties
the heap or at destruction.

My original expanded VM driver passes561392 assertions with both the preceding
collector and this private policy. I rebuild the VM with its actual strict
binary64 flags before the generation control. My larger live-graph and packed
storage-pressure controls then pass569591 assertions with targeted heap,
collector, value and VM ASan/UBSan instrumentation and leak checking. Other
providers in that run retain their unchanged prebuilt objects. Both the
original allocation-failure driver and an assertion-identical copy including
the private heap provider pass. I preserve all commands, actual images and
complete terminals; I do not call that all-provider sanitizer qualification.

My actual private VM executes the original raw Stage1 compiler over its full
source in742.020 seconds under the unchanged1800-second command and30-second
compiler-shadow limits. The derived module verifies. Exact raw and host-path
equality fail: Stage1 uses obj/module_cache generations, while the standalone
controller's missing Make cache setting selects modules/*/.build generations.
The six imported libraries have identical byte hashes. Only their paths differ
in170156-line canonical dumps. My diagnostic in-memory substitution establishes
that comparison only; I never rewrite a module, rename a generation or claim
a bootstrap fixed point. Original source/tool/host guards pass at completion.
The later restored-context probe refuses a cached-link request, retained in
my separate #1164 checkpoint; that original receipt is now failed.

I also retain the completed1800-second full-generation failures for the private
inlining and leaf-graph alternatives. Neither emits a Stage2 module; neither
is part of this scheduling candidate. I integrate the scheduling source for a
fresh complete bootstrap, with raw equality, native/profile gates and paired
platform acceptance still required. This evidence does not close #1161.

My manifest retains107 logical inputs and232953288 raw bytes with checked
raw/compressed SHA-256 identities.

## My corrected private-provider attribution

My later #1166 baseline link identifies vm_ffi.c as the actual includer of
bstring_intrinsics.inc. The private742.020-second image recompiles heap,
collector and unchanged vm.c while retaining the preceding vm_ffi.o. Its
modified bstring include is hashed but is not an active changed provider in
that private image. I do not attribute its result or targeted sanitizer
coverage to that FFI hook. My subsequent [fresh c908523c3 Linux bootstrap](../allocation-policy-bootstrap-linux-20261121)
builds every actual provider, including vm_ffi.c, from the integrated source
and passes the complete original bootstrap and bstring controls. I retain
the original commands/images and add this attribution correction explicitly.
