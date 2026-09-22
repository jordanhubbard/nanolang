# My PR522 integration evidence

I repair candidate `9d36ed18c`, integrate main `e0c7eb76d` and PR948
`82c30f5cc`, and retain the failed checks before correction. My implementation
pin after the follow-up repairs is `ac0f2b60a`.

I preserve raw logs as gzip files. `logs.json` records each uncompressed size
and SHA-256. I distinguish earlier focused checks from final integrated gates.

I observed these failures:

- Run `35577220556` rejects newly supported nested-float results in two stale
  tests, selects an unsupported Darwin leak checker, reports Forth signed-shift
  and empty-copy undefined behavior, and exceeds the full sanitizer deadline.
- PR948 run `35690205396` exposes nominal empty-array inference, checker leaks,
  and a scalar reconstruction worker exceeding twenty minutes. PR522 already
  contains the nominal inference repair. Its older checker allocation repair
  does not cover the reproduced float-record leaks.
- Integrated run `35791213789` rejects a positive reconstruction fixture that
  discards its arithmetic result. Reconstruction correctly removes that dead
  expression. I retain the computed bits in the corrected fixture.
- Local Homebrew LLVM instrumentation reports 219 leaked bytes in nested float
  records and 18 bytes in the global-record case. I release overwritten parser
  names and owned record metadata, and borrow nominal names from their owners.
- Strict Clang UBSan stops bootstrap at a short foreign call to
  `vm_mktemp_dir`, through an incompatible integer-returning function-pointer
  type. I use the existing typed libffi path for every arity, including provider
  string results, and retain the provider cleanup contract.

My corrected checks include:

| Check | Result | Scope |
| --- | --- | --- |
| Emitter and native-array cases | 20 methods pass | First integrated compiler; Homebrew native compiler and leak checks |
| Partition inventory and refusal controls | 9 methods pass | Complete target union, unchanged flags, aggregate failures, Darwin Make formatting |
| Ordinary canonical bootstrap | Pass | Before the later metadata and foreign-call corrections |
| Verifier | 98 tests pass | Homebrew ASan and strict UBSan |
| Forth session and double-number corpus | Pass | Homebrew ASan and strict UBSan; zero reported corpus errors |
| NanoISA including empty code append | 2,988 assertions pass | Homebrew ASan/LSan and strict UBSan |
| Float records and empty record fields | 4 methods pass | Corrected instrumented frontend; float-record leak checks enabled |
| Foreign-call suite | 29 cases pass | Corrected dispatch, instrumented short string/integer providers, ASan and strict UBSan |
| Binary64 facts and reconstruction | 4 methods pass | Live arithmetic and both prior-output refusals |

I compile instrumented objects in a separate directory with Homebrew Clang,
`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
-fno-sanitize-recover=all`. I set `UBSAN_OPTIONS=halt_on_error=1`.
I preserve the repository's existing compiler-wide leak policy and enable
`detect_leaks=1` for the NanoISA and float-record checks above. The first
ownership-control command omitted the existing sixty-second bootstrap shadow
budget and stopped at the ten-second product default; its corrected invocation
then exposed the foreign-call type error. I retain both failures.

My strict bootstrap completes after the foreign-call correction. The parser
and type-checker controls pass. The scope control then catches my attempted
free of a borrowed record module name; I register only checker-created copies
with their allocation owner instead. The unchanged standalone scope control
passes all 45 assertions under ASan/UBSan after that correction. I retain the
failed combined gate as `metadata-ownership-final.log.gz` and the corrected
scope result separately. The replacement complete hosted matrix remains pending. A passing focused check is not release
acceptance. I require all release checks before merging or tagging.
