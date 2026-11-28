# Checked record-list loops and complete tested native names

I qualify source commit `0319ae8c5f8bbc77ff36cff9059f75a7c74dcb7f` on native Linux ARM64. I preserve my preceding [compiler STRING checkpoint](../string-scope-compiler-linux-20261123/README.md) and bind its manifest by hash.

I retain known record element names through checked List bindings, parameters, fields and computed producers. My interpreter reads borrowed record headers from existing integer slots without adopting caller storage. My native loop uses the existing value-returning getter and canonical helper names. My driver owns complete detected names instead of copying them into 64-byte storage. I retain existing detection counts, schema/runtime selection and refusal paths.

I independently verify this archive with:

```sh
python3 docs/evidence/record-list-loop-linux-20261123/read_archive.py
```

I verify 2,543 records, 1,423 exact Git inputs and 51 distinct native ELF images. My content-addressed gzip objects retain raw and compressed hashes, sizes, scripts, command arguments, streams, providers, intermediate images, failures and native products. Intermediate checker/emitter reconstructions are explicitly derived afterward and match recorded hashes; I do not describe them as original snapshots.

| Check | Observed result |
| --- | --- |
| New checker/private-runtime contexts | Four complete contexts per mode; ten checked shadows per runtime context; 256 caller-owned record calls per runtime run; normal and ASan/UBSan with leak detection pass |
| Caller lifetime boundary | Stack record header and managed text identity/refcount survive; no host storage adoption; both list/environment teardown orders; final GC bytes/objects and carrier live count zero |
| Original private STRING and loop metadata targets | Normal and sanitizer pass, including the original ten STRING shadows and 32 loop shadows |
| Original evaluator/checker | All 134 evaluator and 88 checker cases pass normally and under ASan/UBSan; these broad sanitizer runs explicitly disable leak detection |
| Actual native payload suite | Five, then six, then final seven methods pass; original four methods retain their complete ASTs, including assertions and budgets |
| Name boundary | Distinct 66/126-character names sharing a prefix pass shadows, native publication and execution; same-name loop binding also passes without a scope repair |
| Instrumented native payload suite | All seven pass with changed driver/emitter and COMMON/runtime providers instrumented; remaining backend providers are ordinary; leak detection explicitly disabled; this is not a fully instrumented compiler |
| Composite source | All ten shadows, native publication and execution pass, including fields, aliases, computed producer, nested/control/return/empty/mutation routes |
| Full compiler source, before name repair | C-seed source stage exits 0 in 110.660s, all 1,306 dependency/root shadows pass |
| Full compiler source, final image | Exits 0 in 111.057s, all 1,306 shadows pass; original 600-second parent budget and default shadow deadline |
| Final product | 26,105,208 bytes; SHA256 `fe3ea46d79353d7c5b7df3f1ced6a617b1c5ba8c929bfba626e54f460f687e7e`; actual `--help` exits 0 |
| Source/object confirmation | Fresh driver/transpiler recompiles are byte-identical to the objects linked into the final compiler; stage source guards match before/after |
| Policy and frozen tools | 3,468 static fragments, zero reserved names; five original tool images unchanged |

My final compiler image is SHA256 `4d6731c342206bbb6c4d0db672ca898516df01ee2304d53ab5cac728db6c7ffd`. My original compiler source is SHA256 `173e8e59512357b7d32e21ea770beae9df2ce1dbd110a083eacf00961ebe69a1`. The stage guard map omitted main.c; my separately retained byte-identical object confirmation binds its final source instead. I do not claim a before/after main.c stage guard that I did not collect.

I preserve failures and setup corrections. The original loop first failed checker admission; after metadata repair it reached interpreter refusal; after interpreter support it reached incorrect uppercase native helpers and an invalid dereference. Each stage and prior-output marker is retained. The 66-character source then passed its shadow and aborted in `__strncpy_chk` copying 66 bytes into a 64-byte driver temporary; the actual bounded GDB trace is retained.

My first new host test used nonexistent `dyn_array_free`; I corrected that fixture call to the existing `gc_release`. A later private-runtime test omitted `type_check_root_shadows`: ASan caught a List_length use-after-free in its first unchecked shadow, before the initially suspected mutation route. I added the actual CLI shadow-checking phase, preserved its assertions, and retained both failures. This setup failure does not establish an ordinary-production collector failure or a mutation-specific cause.

I close only #1220's C checker/interpreter/native iteration boundary and #1221's demonstrated generated-name defect. I do not activate private ownership by default, change Puck's sources, infer arbitrary unlimited host names, or claim other backend parity. #1213/#1217/#1211/#1116, raw/self-hosted stages, paired platform bootstrap and the complete 69-requirement release remain open. Timing observations establish no historical timeout cause or performance gain. All native jobs in this receipt are terminal; I hold no Darwin reservation and perform no NAS action.
