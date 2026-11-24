# My C-seed native publication boundary

I check root and loaded dependency identities before writing native output or
selected diagnostics, including error reporting after a later import fails. I refuse source aliases through the same path, symlinks
and hard links. I also require generated C (`--keep-c` and `-S`) to have a
separate destination from those sources, the native artifact and selected
diagnostics. These are checks of stable filesystem entries, not protection
against concurrent path replacement or an untrusted compiler.

After the selected shadows pass, I reserve `.nano-native.XXXXXX` beside the
requested executable. I give that provisional path to the configured C
compiler with the existing language, warning, library and instrumentation
flags. A C compile or linker failure leaves the requested executable alone.
A zero compiler exit is insufficient: I require a nonempty regular executable
with one link. I refuse a missing, partial nonexecutable or symlink result.

I normalize deterministic Mach-O output on the provisional path when requested.
I remove temporary C before publication, recheck source/output identities, then
rename the completed executable onto its destination. A cleanup or rename
failure is a failed compilation. I remove my provisional executable on failure
and report a removal error if cleanup cannot complete. A killed compiler can
leave a provisional file; I do not claim crash recovery or durable publication
across power loss.

Successful replacement uses the native compiler's executable permissions.
It replaces an output symlink itself and leaves its former target alone.
Failed compilation preserves the old file's bytes and permissions, or the old
output symlink and its target. I refuse a dangling output link when I cannot
establish its destination identity.

`--keep-c` retains `<output>.c`; `-S` retains `<input>.genC`. These are requested
inspection outputs and can remain after a native compilation fails. Coverage
instrumentation also retains compiler-generated `.nano-native.*.gcno` notes and
matching runtime `.gcda` data beside the executable. Their provisional prefix
is part of the actual compiler coverage metadata; they are not provisional
executables. I do not rename or discard those metadata files.

`make test-cseed-native-publication` exercises real native compilation/linker
refusal, partial and false-success compiler controls, cleanup refusal, prior
bytes/permissions, successful replacement, output symlinks, directories,
root/dependency aliases, generated C and coverage output. The focused Linux
result does not establish paired Darwin, raw bootstrap or whole-release
acceptance.
