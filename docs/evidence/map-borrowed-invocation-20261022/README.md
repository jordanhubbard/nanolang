# Private borrowed-state invocation checkpoint

I implemented checkpoint2 after ordinary adapters a454075e9 and contract7abaf9f7f.
A private module-taking entry owns one fresh preparation through import, stack
activation, initializer/entry, assertion resumes and cleanup. Its40-byte typed
view borrows the ordinary VmState; the unchanged14512-byte preparation owns the
same proof/tables. No caller can pass executable authority. Public VM selection,
loader/save/native and publisher admission remain closed.

The core has a distinct sixth context argument; existing five- and four-argument
wrappers forward NULL. Borrowed preflight runs before dispatch indexing and before
IP/profile/operand mutation. Ordinary record ordinals, global capacity, cycle
collector metadata and globals survive. Map operands remain actual stack roots
through fallible checked operations. Root entry stays zero-arity INT; initializer
VOID/INT residue is drained. Failed selected invocations unwind owners/frames,
not already committed alias mutations. Success leaves the actual INT stack result
for caller consumption before reuse.

Review caught recursive private array validation before any build. The borrowed
route instead checks shells/selected edges; nested shape closure relies on full
historical proof, fresh import and closed transitions. I explicitly do not claim
shallow checks recover arbitrary runtime depth or permit concurrent host mutation.
Private recursive checks and old wrappers retain their behavior.

## Actual gates

- First switch gate:142 checks,8.222s, four unchanged originals twice.
- Expanded switch/goto:1836 each,16.213s; all eight original/source-global inputs.
- Real-source U8/map and defensive expansion:2076 each,17.104s.
- Final source:2112 each,16.638s, all twelve raw inputs twice plus actual ordinary
  NULL-route execution through the newly compiled vm.c. No retained CLI binary is
  substituted for that wrapper evidence.
- Old private map:2714 each,16.286s; old preparation:12495 each,16.558s.
- Old private array:2390 each switch/goto,15.809s.
- Six rebuilt proof allocation/work rows exactly equal the retained baseline,
  including Analysis39840. Sixteen GCC16/Clang syntax cases passed for macro-off,
  array-only and combined source, switch/goto, and the new harness.

Runtime gates used O1 ASan/UBSan with nonrecovering sanitizer errors, actual retained
link inputs and the source-including harness. Commands/statuses and selection files
are retained here. No production change followed the reviewed shallow correction;
subsequent changes were tests/target/receipt only. No runtime/build failure occurred
in this checkpoint. Review findings are not described as executed failures.

The actual stack capacity is1048576 values. First seed reservation is84068462B /
16980536 logical work; the first repeated no-growth reservation is67225966B /
16915222. Later identity-table collisions can change repeated import work by two
logical probes; these measured pointer-dependent costs are not invariant budgets.
The owner/view sizes and unchanged caps are asserted. The full measured invocation
has96 allocation positions, each exercised with transient and persistent denial,
idle cleanup, independent retry and zero tracked allocations after destruction.
Nonessential collector-buffer allocation failure may succeed; the test does not
mislabel every denial as required guest failure.

Controls include initializer root-tail/assert resume, missingGET INT|VOID, exact
false assertion and alias value2 surviving unwind, host reset/retry, unconsumed
result refusal, nonidentity ordinal/layout, CALL/TAIL17 distinct/duplicate owners,
cumulative tail retain overflow, actual stack realloc failure, invalid view/DAG/
current-function, cache handle/recipe corruption and restoration, and selected vs
unselected shell checks followed by fresh import refusal. Globals are not reset.

The new retained source `tests/nanoisa/fixtures/map_nested_u8_borrowed.nano` keeps
all nested[-1,256,300]→255/0/44 assertions from the existing accepted U8 test and
composes a STRING→INT map field. All four existing producers compiled the exact
file; original verifier and ordinary VM passed before annotation, with producer
hashes unchanged. The raw modules are retained under
`/private/tmp/nanolang-map-borrowed-20261022/u8-*.nvm` and hashed in final-pins.json.
The test adds exact private declaration metadata only after original verification;
it makes no source-publisher authority claim. The earlier synthetic U8 composition
is separate evidence and does not replace this real-source gate.

## Reproduction

Use `final-environment.json` for the exact twelve module selectors and retained
compiler/link closure, then run:

```
python3 -m unittest -f -v tests.test_record_map_borrowed
```

Required Make target is `test-record-map-borrowed`, wired into test-units with the
existing runtime prerequisites and inherited compiler flags. It accepts zero,
four, eight or twelve raw inputs honestly; aggregate qualification supplies all
twelve. The final source/module/link hashes are publication pins, not an invented
before/after claim for every retained object. The source producer before/after
pins are separately recorded in u8-source.json. No frontend/bootstrap was rebuilt.
