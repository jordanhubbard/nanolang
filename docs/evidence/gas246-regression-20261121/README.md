# My native GNU GAS regression comparison

I extend #1157's retained one-condition candidate with a complete native
AArch64 GAS regression comparison. This is upstream component evidence, not
GNU2.46 product admission, a replacement CI toolchain, or release acceptance.
Sparky hands the compiler-capture lane to my Puck session in
[#1158](https://github.com/jordanhubbard/nanolang/issues/1158#issuecomment-7075244116).
Its translator work and every existing qualification root remain separate.

## My exact inputs and results

I copy the retained upstream2.46.0 archive, SHA256
`d75a94f4d73e7a4086f7513e67e439e8fcdcbb726ffe63f4661744e6256b2cf2`,
into two independent source/build roots under
`/home/jkh/.worktrees/nanolang/gas246-regression-20261121`.
All29637 regular source files match except `gas/input-scrub.c`. My candidate
is the already-retained include-state condition, SHA256
`6ffe0ed82d02c8f92198b4c2b5200a8093b7eea0a07a96bf951eecabf021b036`;
my original source is SHA256
`40e7e14b5845e3f48b3be1cde0ea0b073f3e787864bdeb9d06dea9b545acfa07`.
Both assemblers report `2.46.0.20260210`. That shared version string cannot
identify the corrected binary or justify admitting other binaries with it.

I build original GAS and binutils tools independently with at most two workers.
My full unfiltered native test commands are identical for both images:

```text
make -j1 check-gas RUNTESTFLAGS=CLANG_FOR_TARGET=/usr/bin/clang-18
```

Both return0: **1239 expected passes,10 unsupported tests**, and no unexpected
failure, unresolved case or unexpected pass. All1249 outcome lines match after
mapping only each independently extracted source-root prefix to the same label.
The original summaries, logs, raw path differences and outcome audit remain.
The supported native target is AArch64 Linux; this does not cover other GAS
configurations or qualify NanoLang's original source/platform gates.

I then run12 fresh direct-assembler comparisons: the four exact retained
preprocessing streams through installed qualified2.42, freshly built original
2.46 and freshly built candidate2.46. The original2.46 image loses payload.S
identity on the complete witness; the candidate and2.42 retain it. All three
neighbor controls retain identity under all three images. I retain actual ELF
objects and complete decoded-line/debug-info reports. All source/macro/tool
hash guards match afterward. I change no witness flags or expected decisions.

## My preserved setup outcomes

My first full test attempts stop before any GAS case: the upstream Tcl loader
rejects stderr from the installed development Clang's LLVMgold metadata query.
Both assembler builds already passed. The same query on installed Clang18
returns0 without stderr. I preserve those failed summaries/logs and resume only
the unexecuted full test phases with that explicit auxiliary compiler. I change
no assembler/GCC image, test selection, assertion or expected result. I do not
change the compiler selected by NanoLang's hosted user-guide gate.

My first fresh witness fixture omits its relative macro include and fails to
assemble. My first fixture correction has a Python replacement error and starts
no assembler. I retain both scripts and terminals. The corrected fixture copies
the original macro bytes; all12 comparisons then run with complete inputs.
No successful witness or completed regression suite is repeated for green.
My initial outcome audit compares raw absolute source paths and refuses; the
retained raw diff establishes the exact prefix mapping used above.

I download Expect/DejaGNU packages and extract them privately. Their bytes match
APT's recorded SHA256 metadata. I run no system installer and replace no system
compiler, assembler or shared library. Private runtime paths and versions are
recorded. The original retained source archive, private precursor images and
all older qualification paths remain unchanged. The worker disk warning remains
visible; no cache or evidence is deleted.

## My retained evidence and limits

My [manifest](manifest.json) retains125 logical inputs,115196914 raw bytes and
58406431 unique compressed bytes, including the complete upstream source archive,
exact source delta, private dependency packages/metadata, native assembler and
readelf/objdump images, configure/build/test logs, failed setup outcomes, fresh
witnesses and controlling scripts. Every compressed/raw SHA256 and length is
independently checked on Linux and again on Darwin after copying. Tool versions
and hashes bind the observations; this is not a relocatable installed toolchain.

I preserve ordinary NanoLang assertions, capture guards, flags, deadlines and
GNU admission. #1157 remains open: the actual hosted distro assembler is still
unqualified. #1159's full private Clang observation, #1158's Darwin qualification,
#1171's actual user-guide gate and full #976 acceptance remain separate.
