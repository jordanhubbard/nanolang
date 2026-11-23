# My Linux REPL source journal

I pin production source `9904098108f2bb19a3446808c1bb6558e91023d4` and
documentation checkpoint `21a17124a4cecf1e261048e28d1145c1f29c7c67`. My reader
verifies 2804 records, 1437 exact Git inputs, 22 distinct native ELF images,
39 independently decoded successful saves and six preserved I/O fault cases.

Under #1225 I replace typed-only save history with a versioned journal of
accepted input/load/reload/clear operations. I retain source bytes and absolute
origins, replay function-only reloads in order, and preserve earlier computed
values. An initial origin control observes alias-sibling 71 during admission
where physical-sibling 67 is required. I preserve that failure/image/source and use one captured
origin during admission and replay, including replacement of the old pathname.
Plain NanoLang files still load. My [session contract](../../REPL_SESSIONS.md)
defines the format, dependency/effect boundaries and failure behavior.

I retain the original production zero-fragment save through my base
[module/REPL receipt](../module-check-rollback-linux-20261123/README.md).
My 16 new session methods pass alongside the unchanged original three callable
and fifteen state methods: 34 controls normally, under ASan/UBSan with leak
detection enabled, and with unchanged production O3/Werror flags. The session
observer retains actual stdin, fixture/save bytes, before/after file inventories,
native output, selected environment guards, image hashes and original 30-second
native deadlines. Every profile completes 36 observed session-native commands.

I check loaded-source removal, typed/loaded mutation order, rejected and duplicate
loads, rejected reloads, earlier results versus later function bodies, ignored
reload effects, clear, moved journals/relative imports/Unicode and newline origins,
symlink admission and replaced-origin replay,
malformed/overflowing/truncated/NUL framing, capacity refusal, partial semantic
replay, zero-accepted-record failure, and flattened load/save/load round trips.
Invalid framing applies no prefix. An operational failure reports the already
accepted prefix and excludes its failed operation from accepted history.

I stage output beside its destination and check writes, flush and close before
rename. My Linux external-only descriptor observer injects real short-write and
close failures into the selected native save stream. All profiles retain prior
destination bytes, report failure, emit no saved-success message, and remove
their staging file. A directory destination independently exercises rename
failure and preserved prior content. A subsequent ordinary write/load control
passes. No fault channel enters production source, and I make no power-loss
durability claim.

The journal work exposes two REPL admission defects. Under #1227, rejected
ordinary source leaves nominal declarations visible; the retained production
probe subsequently accepts a Phantom-typed function and returns 7. I apply the
existing environment checkpoint around REPL input/file/reload admissions and
preserve previous scope, variables and functions while withdrawing failed
declarations/aliases. My first checkpoint selected the last explicit owner,
while imports select the first; an actual retained refusal observes stale 11
instead of 22. The corrected checkpoint matches the import owner.

Under #1228, valid Point bindings lose their checked nominal name before the next
input. I copy that owned name onto the evaluated binding without changing its
value ownership. All seven admission methods pass normally, with production
flags and under ASan/UBSan with the broad legacy leak boundary disabled. Four
independent admission methods pass with leak detection enabled, including the
multi-context alias case, prior-context teardown and a loaded record binding.

I preserve every complete aggregate failure. The first sanitizer admission run
fails four methods, including a 10-byte REPL current-module owner leak corrected
at session teardown. The complete final seven-method LSan run still fails three
aggregate methods at 168/254/254 bytes. The stacks identify unchanged checker
record-parameter scaffolding and runtime argument/local copies under #1149/#1116.
The original six-method production baseline fails every new semantic control.
I retain the full failed fixtures alongside the additional broad/focused runs.
Aggregate/default ownership remains open.

Only the REPL implementation changes in production. I compile its new object in
three original profiles and link separately named images to exact retained
shared providers. Every retained object/archive is hashed before/after, and its
bytes match the base receipt. Compiler/evaluator/runtime sources, public tool
images and host-library closure remain unchanged. I carry the base's source
qualification only for that unchanged shared code; current-source raw/bootstrap,
paired-platform and whole-release acceptance remain open.

My reader checks committed Git inputs, compressed/raw archive identities,
command streams, fixture products, independently decoded successful journal
records, unchanged original test sources and explicit failure scopes. Initial
passing 14-method source snapshots are reconstructed against their recorded
SHA256 guards; failed admission source/images were retained before correction.
Mutable component outputs were rebuilt between observations, so I do not claim
all intermediate passing images survived. Identical older objects are referenced
by verified relative paths. Run
`python3 docs/evidence/repl-session-save-linux-20261123/read_archive.py`.

Complete #1205/#1140/#1116, aggregate LSan, default STRING ownership and release
acceptance stay open. All frozen roots, audits/stashes, Puck-owned source/capture/
regex/CI and NAS/TCC/migration pauses remain unchanged. I perform no Darwin job.
