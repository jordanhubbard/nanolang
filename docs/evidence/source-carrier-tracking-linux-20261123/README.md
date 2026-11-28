# My private source-carrier tracking checkpoint

I qualify source commit `933083ea5f86f0b40252ff2423f4979267381d52` on Linux ARM64 with GCC 13.3.0 and GNU assembler 2.42. I enable tracking explicitly in external qualification observers after the original source checker. Ordinary environment creation and my CLI do not enable it.

I preserve the complete original source fixtures and assertions. My record/union fixture completes six shadows, my scalar fixture seven, and my typed-list source witness two. Both fresh normal and ASan/UBSan+LSan builds pass. Their actual teardown counters are respectively `19 registered / 18 destroyed / 1 transferred`, `18 / 16 / 2`, and `12 / 12 / 0`; all finish with zero live tracked carriers. I retain caller-return, borrowed-value, nested-signature constructor and typed-list pointer-identity controls.

My final cohort records 41 commands. The original 134 evaluator and 88 checker cases pass in both profiles; these two broad sanitizer suites explicitly disable leak detection. The focused source, metadata, engine, storage, binding and borrowed-value sanitizer controls enable it. I preserve all four expected default-mode failures: the scalar fixture leaks 1,857 bytes in 78 allocations, the record fixture leaks 3,600 bytes in 153 allocations, and both API replacement and checked-source typed-list routes still report use-after-free. I do not accept default typed-list ownership or default source leak cleanup.

Before my residual repairs, the original default scalar and record reports were 1,937 bytes / 82 allocations and 3,720 / 159. Private tracking removed their carrier leaks but exposed separately retained match scratch-vector leaks of 80 / 4 and 120 / 6. I now release only those copied pointer/value vectors, preserving their borrowed entries.

For #1212, I register only explicitly allocated generated-list descriptor metadata with my existing checker-allocation owner. The original checked witness exposed 379 bytes / 14 allocations. An eight-environment, 104-instantiation baseline reproduces 36,632 bytes / 1,456 allocations after its assertions finish. The repaired normal and sanitizer test passes with leak detection, checking all four generated signatures, duplicate registration, source-name independence, descriptor stability through function-array growth and borrowed declaration storage. My original checker and checked-source witness also pass.

I have not implemented collection during execution. A separate actual-source observation runs two additional batches of 64 original record calls. Before teardown, live carriers grow from 18 to 210 to 402 with no destruction. Final teardown then reports `403 registered / 402 destroyed / 1 transferred / 0 live`. This is a retained failure of bounded live-heap acceptance, even though LSan is clean. #1116, #1140 and #1211 remain open. Complete in-flight roots, typed-field/list facts, host aliases and external list/GC lifecycle protocols precede incremental collection and default activation.

I add one private pointer to `Environment`; I preserve `Value`, container and FFI layouts. I capture explicit evaluator allocation sites and environment record copies rather than automatically adopting arbitrary SDK constructors. Complete low-level IO/FFI constructor and host-call ownership qualification remains outstanding. Puck-owned merge, nvm2c, regex, capture and CI inputs remain unchanged. I claim no current-source paired bootstrap, Darwin or whole-release acceptance. All native commands in this checkpoint are terminal; no worker reservation remains mine. Frozen roots, audits, stashes and NAS/TCC/migration pauses remain intact.

I retain 1,673 hash-verified records, including 1,405 exact Git inputs and 15 executable images. Other complete evaluator/checker/engine/storage/binding/borrowed executables and provider objects remain in their frozen private roots with commands, full streams and identities recorded; I do not claim their bytes are all copied here. Earlier source snapshots, unsuccessful reports and development observations remain in the archive. My archive reuses hash-bound objects from the preceding storage checkpoint.

I verify compressed objects, exact Git blobs, fixture bodies, shadow events, source/tool guards, ELF architecture, command outcomes, observed counters, retained negatives and the default-disabled boundary with:

```sh
python3 docs/evidence/source-carrier-tracking-linux-20261123/read_archive.py
```
