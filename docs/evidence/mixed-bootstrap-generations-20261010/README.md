# My mixed-service compiler bootstrap

I qualify the compiler/runtime sources introduced through `f5696fdc6` under
#990/#982. Later `5cf2528e7` evidence and `0d74f2462` test-fixture changes do not
change the bootstrap's input map. All 17 bootstrap steps pass. Both raw stages
contain 644,408 bytes with SHA256
`cc374edc2728e1d992bca389eeebbaf4ad3f218a6dd8b5aee5f184b0e12bd69c`.

I run:

```sh
env PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_CFLAGS=-O2 \
  make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  NANO_BUILD_CACHE=/private/tmp/nl51-mixed-bootstrap bootstrap3
```

Stage1 self-compilation takes 522.626 seconds and Stage2 takes 515.215 seconds.
These are concurrent-workload wall times, not isolated performance benchmarks.
The final installed compiler compiles and executes hello with `bin/nanoc_c`
removed; the Make terminal retains that check. I preserve the five tool hashes,
six immutable host-library hashes, 1,139 source hashes and generated compiler
hashes; all match at the archive boundary in `input-comparison.json`.

I retain `/private/tmp/nl51-mixed-bootstrap`: both compiler generations embed
paths to its immutable host libraries. The raw equality is a local fixed point,
not proof of semantics or reproducibility across hosts. The complete 5.1 scope,
DNS/WebSocket, hosted and exact-candidate release gates remain required.

## Service qualification

I run the complete File, TCP and mixed CLI suites through each generation's VM
and native compiler. Stage1 passes all 24 methods in 824.027 seconds, with all 547 retained
commands matching their expected status across 24 fixture roots. Stage2 passes the same 24 methods in 815.236 seconds, with all 547 commands
matching their expected status. My final comparison retains unchanged sources,
tools, hosts and compiler artifacts across both generation suites.
They include C seed/bytecode controls where their harness defines those drivers,
and use `obj/mixed-service-test-install` for installed mixed runtime tests.
