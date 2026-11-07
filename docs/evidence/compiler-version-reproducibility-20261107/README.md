# My deterministic compiler version report

Under #1045 I remove ambient build dates and times from my C and legacy hybrid
version reports, their shared version header and future generated release headers.
My C report now consists of `nanoc 5.1.0` and `nanolang compiler`; `-v` and
`--version` agree.

My regression recompiles the real C entry point and freshly links two actual
compiler binaries against the same frozen providers. SOURCE_DATE_EPOCH differs:
946684800 and 978307200. Before the repair, the two reports differ as January 1,
2000 and January 1, 2001; `before.log` retains that failing negative control.
After the repair, the same test passes in 4.264 seconds, requiring identical
reports and verifying every shared object/archive and original compiler binary
keeps its bytes and modification time. `after.log` also retains six passing
release-version/package-generator tests and the release workflow checks.

My fast gate passes 223 Clang and 223 GCC compiles plus its bootstrap-tool
control. I do not claim bit-for-bit native binary reproducibility or normalize
path-dependent error text here; retained-policy review remains under #1039.

While checking the touched hybrid entry point I reproduce #1073: its transpiler
call lacks the current input-file argument and both native command formats omit
one root argument. `hybrid-before.log` retains the strict GCC failure. I supply
the missing arguments; strict GCC 16 and Apple Clang C11/O3/Werror syntax checks
then pass. This does not qualify a complete legacy hybrid bootstrap or execution.
