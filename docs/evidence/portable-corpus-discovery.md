# Portable corpus discovery

I tested candidate `6bf184796991c23fb0d9469c7cb7292bf585dfa7` on puck,
macOS 26.6.2 arm64, with Apple Clang 17.0.0 and Bash 3.2.57. I built
`stage1 nano_virt nano_vm` in a fresh isolated worktree.

My module compiler invocation tests passed 24/24, including an executed
assertion from an imported filename containing quotes and control characters.
My empty-record-array regression compiled, ran shadows, and executed with both
the native compiler and NanoVM.

My compiler-selection regression exposed silent omissions in both corpus
runners: Bash 3.2 does not implement `globstar`. I now discover recursive
fixtures with NUL-delimited `find` output and read them in the runner's shell,
so counters retain their values. An empty negative corpus fails explicitly.
I pass runtime commands as argument arrays and use Perl's direct executable
form, preserving spaces in artifact filenames on all three backend paths.

After this repair, all four compiler-selection methods passed on Linux arm64
and macOS arm64. They cover direct and deeply nested fixture paths, spaces,
C/VM/daemon selection, compiler-link changes, selected-compiler failure, and
empty negative discovery. My real Darwin negative corpus passed 37/37.

Puck logs:

- `/tmp/nanolang-darwin-6bf18479-build.log`
- `/tmp/nanolang-darwin-6bf18479-tests.log` (original portability failures)
- `/tmp/nanolang-darwin-portable-corpus.log` (four repaired methods)
- `/tmp/nanolang-darwin-portable-negative.log` (37 contracts)
