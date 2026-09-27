# Focused tuple ownership evidence

I retain the original e456006ad and corrected73b61d06f Puck focused runs here.
I prepared fresh ordinary COMMON_OBJECTS/RUNTIME_OBJECTS at each exact pin.
I did not run bootstrap or the full source/native corpus in these runs.

Both runs select exactly the checked storage/scheduler/allocation-prefix method
and the array intrinsic/declaration identity method from GenericRecordLists.
The five fixture helpers rebuild their documented owning translation units;
other prepared providers remain ordinary. Apple ordinary and Homebrew
ASan/UBSan use explicit selected tools, current SDK, strict flags, leak checking,
and the original fixture and supervisor deadlines.

Original e456 preparation passed in5.381s and ordinary focused checks passed
in7.260s. Its sanitizer phase failed in7.418s: the identity child completed its
assertions but LeakSanitizer reported746 bytes in20 allocations. Both the child
and outer nonzero terminals remain. I do not claim the unreached sanitizer
array-allocation child passed.

Corrected73b preparation passed in5.158s, ordinary focused checks in7.011s,
and sanitizer focused checks in19.183s. The correction frees owned StructDef
auxiliary vectors, preserves borrowed complete annotations, and corrects fixture
cache ownership. It does not resolve the separate module-name/duplicate-row
ownership follow-up. Root source-reviewed73b before this execution.

I copied the persistent Puck reports and products to
`/home/jkh/nanolang-qualification/tuple-{e456,73b}-{prepare,focused}-puck-copy`.
The unified local artifact store is
`/home/jkh/nanolang-qualification/tuple-focused-artifacts`.
Persistent originals remain under `/Users/jkh/nanolang-qualification` on Puck.
No temporary-directory-only evidence claim applies to these retained roots.

`report-manifest.json` describes405 retained report files; large reports are
reversibly gzip compressed with both original and stored hashes.
`retention-audit.json` and `audit-retention.py` rehash4,045 unique artifacts
(82,808,530 bytes), resolve131,283 artifact references and compare30 unchanged
input-map pairs. They distinguish nested `returncode` from outer `status`:
58 terminal records include the two nested/outer records of the original leak
failure. This is a local integrity audit, not independent source review.

The newer39ad emitter/rollback source and1b29 null-name correction have separate
qualification roots. These earlier passes do not qualify those changed sources
or establish full native tuple/list/callback parity.
