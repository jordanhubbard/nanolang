# My standalone embedding Darwin qualification

I finish the remaining platform boundary of
`task_a522629205b440f28204162fba9d3960` with the existing fixture at canonical
`5f988ed79dd1f343fd9dca3d9b40c928264e668c`. I change no production or test source.
My [embedding contract](../NANOISA_MIXED_SAMPLES_RUNTIME.md#my-standalone-native-runtime-embedding-prerequisite)
and [corrected Linux qualification](managed-native-embedding.md) remain separate.
Compared with Linux fixturefe321533, the current fixture changes only the
previously reviewed platform dependency-inspection branch and its report.

I run all three methods once on arm64 Darwin25.6.0. They pass in4.991s; the
bounded runner records status0 after5.115s. Exact-byte/hash checks, regeneration
and refusal/output-preservation controls pass. The standalone harness builds
with Homebrew GCC16.2 and Clang23.1 at both O0 and O2, retaining strict C11,
Wall/Wextra/Werror/pedantic flags. All four fresh programs pass the existing
parser-guard, array alias, nominal map, zero-roots-before-disposal and four
arithmetic-provider assertions. All four `otool -L` reports list only
`/usr/lib/libSystem.B.dylib`. I claim no additional sanitizer coverage here.

I explicitly select both compilers, Python and Xcode SDKROOT. The
[environment](managed-native-embedding-darwin/environment.json) records resolved
selection and versions; tool maps hash actual compiler binaries, GCC cc1,
Xcode assembler/linker/otool, xcrun, Python, SDKSettings.json, libSystem.tbd and
the exact [runner](managed-native-embedding-darwin/runner.py). All11 entries
match before/after. These are selected tool/SDK identities, not a hash of every
transitive system header or library. All eight source inputs match canonical
Git bytes and their before/after maps. This is a minimal source snapshot,
not a full checkout/compiler build. Temporary regeneration controls are fresh;
no historical failed executable is replayed.

My [result](managed-native-embedding-darwin/result.json) and
[log](managed-native-embedding-darwin/gate.log) retain the first terminal.
The [manifest](managed-native-embedding-darwin/manifest.json) checks28 reports
and artifacts inside [the archive](managed-native-embedding-darwin/artifacts.tar.gz),
including generated C, all four executables, build/run/dependency logs and
per-route exact commands/hashes. Archive SHA256:
`64b33290e5f3a9e1b260ec88329c9d058e96b7312dff36ac7322ec7769212c63`.
I transferred it from the isolated Darwin directory
`/private/tmp/nanolang-embedding-darwin-5f988` and independently verified all
28 hashes on Linux. The source archive hash is recorded in environment.json;
its selected input bytes are recoverable from the canonical pin.

This completes a522's measured acceptance. Its ledger closure waits for actual
canonical evidence merge. I reconciled only documentation taskd8aa through
actual PR831 merge5f988ed79. Parent4be, owner ARRAY430220, full-product gates and
release publication remain open. I did not rerun a compiler bootstrap or the
product suites.
