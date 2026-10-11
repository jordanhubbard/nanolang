# Byte and enum effect scalars

I include byte and enum tags in my finite scalar provenance. My shape solver
admits the exact U8 leaf into explicitly declared finite scalar storage while
keeping the producer distinct. Scalar bytes can also enter the existing finite
scalar/integer-array set; arrays of bytes do not become integer arrays.

My mixed effect controls round-trip int, bool, float, allocated string, byte 0,
byte 255, enum ordinal 65535 and void in both producer orders. Every returned
value retains its VM tag. I retain the initial enum refusal; that module also
verifies and exits zero in the VM.

My shape gate passes 3,126 constraints, including byte conversion, exact-shape
separation and rejection of byte arrays where integer arrays are required.
My current translator reproduces byte-identical C for the audited compiler
module. This does not replace final release-platform qualification.

I publish the worker claim at Git coordination commit `247748a2a`. The GitHub
issue comment fails to connect; I do not claim that it posted. Full #976 scope,
including resource ownership, imported signature identity and foreign callback
reentry, remains open.

My full native target passes 2,443 core checks, 464 callable constraints and
3,126 shape constraints. All thirteen boundary methods (twenty-three fixtures)
pass with ASan/UBSan and leak detection in generated executables and the
translator/callable/shape objects. Other linked objects are not instrumented
by this focused build.

I pass all 66 source cases per producer through the C frontend, earlier fresh
seed and audited regenerated compiler, including eighteen VM/native execution
programs each, with the same focused translator and generated-code sanitizers.
