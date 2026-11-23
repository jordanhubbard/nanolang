# My bounded Linux REPL repair

I pin source `cbc2cccbde9dad3821770defec810959de266fe7` and verify 3090
records, 1428 exact Git inputs and 73 distinct native ELF images with my
standalone reader. I restore my strict standalone REPL build under #1205. I use my current
environment/module APIs, retain parsed source and path owners until environment
teardown, check executable input before mutation, and clear variable owners
without retiring declarations. My function-only reload checks compatible
signatures and patches existing slots without replaying top-level state changes.
It preserves callers, namespace targets and imported aliases exercised by my
controls. I refuse additional parsed inputs at the existing 4096-entry bound.

I retain the original strict compile failure and three discovered product
failures: an eight-byte loader context leak (#1222), a malformed trailing-escape
buffer overread, and an unchecked assignment that replaced an int with STRING.
My corrections pass the original three callable methods and fifteen additional
state methods normally and under ASan/UBSan with leak detection enabled. All
eighteen also pass with unchanged production optimization and warning flags.
I wire both REPL targets into my ordinary unit prerequisites.

My actual production build first failed at tuple carrier-edge allocation under
the original O3/Werror flags (#1224). I validate signed counts, vector presence
and every child before allocation or evaluation, and refuse allocation failure.
Eight forked ordinary/private negative controls require deliberate exit1 and an
atexit marker after checking allocation counters and an empty carrier registry.
A sanitizer crash cannot satisfy that marker. Empty and valid tuple controls
perform normal teardown. I do not claim leak qualification of fatal child
teardown. I retain two unsuccessful versions of this new test: one incorrectly
expected recoverable refusal; another inherited the parent's allocation counter.
Neither changes the original product refusal policy.

My complete original evaluator134 and checker88 controls pass normally and under
ASan/UBSan. Leak detection is disabled for these broader historical suites;
the focused REPL/ownership matrix runs with it enabled. Original record/list,
STRING scope, active-root, loop-metadata, collection scratch, native-byte,
heap-engine and adapter controls pass in both focused profiles. The original
native-byte workload remains at zero second-batch growth with its original
1024-byte criterion and two-MiB positive control.

Separately named ordinary and explicitly enabled STRING-candidate C seeds pass
the unchanged seven-method payload suite. Each compiles the complete original
compiler source within its original 600-second bound, with all 1306 selected
shadow starts matched by successful completions and an executable --help result.
I preserve source/tool identities and do not activate private STRING ownership
in the ordinary environment. Original module introspection and embedded fixture
policy also pass. The five frozen public tool images remain unchanged.

This is partial REPL acceptance. My actual C/API diagnostic still demonstrates
that a rejected imported module exposes a declaration whose body AST was freed
(#1223). The context-string correction does not repair it. My production :load
followed by :save also writes zero fragments (#1225). The typed-input save round
trip is narrower than loaded-fragment export. Both defects and complete #1205,
#1140/#1116, current-source raw/bootstrap, paired-platform and release acceptance
remain open. I make no full-release claim.

I archive source snapshots, failed inputs/reports, command streams, captured
temporary fixture bytes/products, distinct retained executables and final
normal/sanitizer/production providers. Mutable provider paths were rebuilt
between intermediate observations; I do not claim every intermediate object or
passing executable survived. The manifest pins committed source bytes and
retains independent command-time source guards. Older identical compressed
objects are referenced by relative path and verified by both compressed and raw
hashes. Run `python3 docs/evidence/repl-restore-linux-20261123/read_archive.py`
from this repository to verify the archive and exact source boundaries.

I preserve all frozen roots, the dirty primary audit and stashes. Puck's source,
capture, regex and CI ownership remains separate. I perform no NAS/TCC/migration
action or Darwin native job.
