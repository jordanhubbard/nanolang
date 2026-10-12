# My owner observation lifetime repair

I reproduced two accepted-invalid instruction sequences on 70cdc4a89. A resource
array observation survived move/store rebinding. A record observation survived
unpacking its original owner, discarding its field, and constructing a replacement
in the same local before reading the observation. I retain both failing regression
runs; I did not execute either unsafe sequence in a VM or native product.

I now reject moving or unpacking a root while a stack observation refers to it.
The same helper checks existing exclusive-borrow conflicts. This preserves the
observation's lifetime without permitting it to resolve against a replacement.
I cover both consumption paths, resource arrays, completed observations and
operations on unrelated roots. Rejected cases leave all instruction facts unchanged;
record cases also refuse VM verification and native emission. Positive record
controls still verify and emit. Typed array execution remains explicitly refused.

My normal suite passes 2,125 checks and my allocation-fault suite passes 3,062.
The same 3,062 checks pass with ASan/UBSan/leak detection on the analyzer, state,
ownership reader, verifier and harness; other linked objects retain their existing
instrumentation. The sanitizer command manifest records exact compile/link commands.
`make -f Makefile.gnu test-affine-effect-runtime nano_virt` rebuilds affected tools
and passes all 26 existing VM/native effect/fault fixtures. All five shared C-source
effect methods pass with `python3 -m unittest -v tests.test_owned_effect_c_source`.
These regression checks do not qualify a fresh self-hosted compiler product.

The original array permutation refusal now occurs earlier at the owner move. I
update its expected diagnostic without changing the rejected sequence or permitting
execution. Exact empty array construction, resource-element extraction/disposal,
producer lowering and collection VM/native admission remain open under #976.
