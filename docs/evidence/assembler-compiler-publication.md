# My complete compiler assembly boundary

My self-hosted compiler emits 1,246,064 bytes of assembly containing 671
functions and 1,996 string directives. Publication exposed two assembler
boundaries: comment markers were stripped inside quoted strings, and my
fixed symbol table reported capacity exhaustion as a duplicate at `s1377`.
I separately reproduced the 4,095-byte decoded literal limit.

I now strip comments only outside quoted text, preserving escaped quotes
and backslashes. String scratch storage is checked against the u32 length
boundary, allocated from the source length, and freed on parsing or pool
insertion failure. Other directive-specific limits remain unchanged.

My symbol, label and forward-patch tables grow with checked sizes and signed
lookup-index bounds. Failed growth retains the prior allocation and reports
an allocation error. Genuine duplicates keep their own error category.
Forward patches are never silently discarded. Cleanup releases all tables.

Validation:

- 2,691 NanoISA checks pass.
- 210 canonical round-trip checks pass, including quoted comment markers,
  import/module names, malformed quotes, 4,095/4,096/16,384-byte literals,
  and 2,200 symbols, labels and forward patches with every jump checked.
- The allocation harness checks scratch/pool failures, table allocation and
  growth failures, duplicate distinctions, index bounds, cleanup and recovery.
- The round-trip suite and allocation harness pass ASan/UBSan with leak checks
  using separate objects in `/tmp/nanolang-asm-sanitizer-obj`.
- The exact captured compiler assembly assembles, verifies in my VM, and
  canonical dump/assembly reproduces the same bytes. The sanitized CLI repeats
  that full artifact round trip without a report.
- The resulting bytecode compiler runs `--help` in my VM and exits zero.

The captured module is 350,652 bytes. Its SHA-256 is
`dc517d54b0417b4934907ea565d34bdb23110da7116bef3f5713faa48746d4ee`.
Logs and artifacts: `/tmp/nanolang-fullcompiler-quote-capture.log`,
`/tmp/nanolang-fullcompiler-quoted.nasm`, `/tmp/nanolang-asm-tables-gate.log`,
`/tmp/nanolang-asm-sanitizers.log`,
`/tmp/nanolang-asm-fullcompiler-sanitizers.log`, and
`/tmp/nanolang-bytecode-compiler-help.log`.

I track these repairs as `task_c77ac0644fda463a8a2d0ae7dd735908` and
`task_74ee50b905d242e68c92abf41b427e15`. Assembly round-trip equality does not
establish compiler bootstrap equality. Actual source compilation through this
bytecode compiler and the self-hosted stage comparison remain separate gates.
