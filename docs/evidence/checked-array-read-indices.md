# Checked raw and tagged array reads

I preserve raw missing reads as `void`. I require integer index tags and check
the full signed 64-bit value before narrowing. Negative, length-sized and very
large indices cannot wrap into an existing element. My VM releases rejected
heap-valued indices and receivers; successful reads retain their payload.

My baseline VM accepted a boolean index as zero. The native matrix also exposed
six wraparound failures: 2^32 and INT64_MIN read element zero across integer,
boolean and string arrays. I preserve both failing logs below. Existing native
missing-read tests now require full-width absence instead of accidental wrap.

Thirteen VM cases cover valid/missing reads, boolean/string/void/float/array
indices, wrong receivers, empty arrays and both integer limits. They check
stack/frame recovery, reference counts and zero residual heap objects after
cleanup. My complete VM suite passes 272,579 checks both normally and with
ASan/UBSan, leak detection and all warning checks at the repository's default
O3 optimization. A separately attempted O1 sanitizer build stops at an existing
NanoCore exporter format warning; I recorded it as task_c08bf552c04740daa060cbc5759eb19d
without suppressing warnings or claiming a runtime defect.

My native gate passes 2,333 checks and 1,092 shape checks. Thirty-eight actual
VM/native bytecode fixtures cover the three primitive element types, valid and
missing full-width indices, dynamic wrong tags and invalid receivers. The
unchanged full compiler bytecode-to-native-to-program gate passes after a fresh
build in 85.366 seconds. This does not complete the NanoISA-only fixed point.

I separately track source/static-native bounds semantics as
`task_c9561b3912a84a67a491cf9a78c1cc4b`; those paths currently panic while raw
reads preserve a missing value. I do not redefine that boundary in this repair.

Logs: `/tmp/nanolang-array-read-baseline-vm.log`,
`/tmp/nanolang-array-read-baseline-native.log`,
`/tmp/nanolang-array-read-fixed-vm.log`,
`/tmp/nanolang-array-read-sanitized-o3-vm.log`,
`/tmp/nanolang-array-read-final-native.log`,
`/tmp/nanolang-array-read-parity/result.log`, and
`/tmp/nanolang-array-read-full-compiler.log`.

After rebasing onto main `a16b2d8a`, the fresh full compiler gate fails before
C emission: `I cannot widen an exactly constrained string destination`. The
same retained current compiler bytecode fails unchanged pre-read-fix nvm2c,
isolating this baseline shape regression from the read runtime repair. I keep
that failure open as `task_031b36c92dbe44e49cea3888878d1963`. Current evidence:
`/tmp/nanolang-array-read-integrated-compiler.log` and
`/tmp/nanolang-array-read-integrated/baseline.log`; the retained module is
`/tmp/nanolang-array-read-integrated/compiler.nvm`. I do not claim the current
full bridge gate is green.
