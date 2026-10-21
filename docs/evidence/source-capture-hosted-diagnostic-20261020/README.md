# I distinguish selected assembler failures

My bounded diagnostic workflow run39543871312 atf90b2d769 completes on Linux
job119037526729 and macOS14 job119037526996. Workflow success means diagnostics
were captured, not every compiler command passed. This is a current hosted
snapshot, distinct from historical qualification39330564592 at271c88986.

Linux PATH clang is Ubuntu18.1.3(1ubuntu1); its integrated -cc1as invocation,
direct backend and fresh shared-library compilation pass. The production
module_assembler_report selector currently accepts only Apple21.0.0 and
Debian14.0.6. The historical Linux failures are confined to eight integrated
Clang/path/debug methods while GNU-as replay passes. This establishes a concrete
compatibility gate to qualify before extension, not permission to ignore reports.

Darwin PATH cc/clang/as are Apple15.0.0(clang-1500.3.9.4), selected Xcode15.4.
The external driver emits as -Q -arch arm64. Both the actual verbose assembly
command and direct selected backend fail: `can't specifiy -Q with -arch arm64`.
Fresh external shared-library compilation fails identically outside NanoLang;
integrated compilation passes. Banner acceptance alone cannot repair this
external backend. I do not strip -Q or substitute an integrated implementation
under an external-assembler claim. Historical compiler.txt records Homebrew23
for the main build, but the source tests independently select PATH cc/clang.

The diagnostic also intentionally records unsupported GCC -fno-integrated-as
and backend -### requests. Linux assembling an integrated-Clang-generated .s
with GNU as fails on .addrsig, while fresh external C compilation succeeds;
that diagnostic mismatch is not the original capture failure. All commands,
statuses, stderr/stdout and artifact hashes remain in the per-platform records.
No runtime, capture admission or original test expectation changed in this
checkpoint. The next compatibility/toolchain choice requires scoped review.

## I qualify the narrow Linux report extension

At `1f7460393903712d909528488363ae8f37a7dc34`, run `39544234552`, Linux job `119038657601` builds the production capture probe and passes all six selected report/integrated-capture methods in 89.790 seconds. I retain commands, current identity, build and test logs under `corrected-linux/`. The diagnostic manifest covers its diagnostic phase; subsequent build/test logs are retained separately. macOS remains queued at this checkpoint; no cross-platform or external-assembler success is inferred.

## I retain the macOS backend failures after report parsing

Run `39544234552`, job `119038657893`, finishes with six methods in75.042s: the exact report control passes, but actual capture has22 failures and2 errors. Apple15 text expansion/replay now reaches empty symbol labels and non-local `.no_dead_strip` errors. I retain the complete log rather than call the integrated path qualified. Installed Homebrew23.1.0 (SHA256 `2562143125f133005a4505f7ffd463a2e52e9a24dbae58cc425374bcbb0ea44f`) still selects `/usr/bin/as -Q -arch arm64`; standalone assembly, fresh shared build and direct backend all fail through active Xcode15.4. A compiler installation or selector alone does not repair that backend.

The next bounded diagnostic inventories installed Xcodes with explicit `DEVELOPER_DIR`, hashes each actual compiler, probes external assembly and exact debug text expansion/replay, and selects a capture tool only if those real probes pass. No product report list or assembler flags change in that diagnostic.

## I inventory installed Xcodes before changing the hosted target

Run `39552342366`, job `119064139683`, stops at the candidate-selection gate: no installed Xcode passes every required probe. Xcode16.1/16.2 actually accept unchanged external `-Q -arch arm64`, but their cc1as debug text still has empty symbol labels and cannot replay. Homebrew23 integrated replay passes. Direct Xcode16 shared linking additionally lacks explicit SDKROOT (`System` library not found); I correct that diagnostic environment separately rather than attributing it to capture. The retained representative original/expanded assembly makes the independent text loss concrete.

I next run the same bounded probes/eight capture methods on the official `xcode-27` arm64 image with explicit SDKROOT and DEVELOPER_DIR. This is diagnostic evidence only, not a retirement-matrix change or a macOS14 success claim. I retain all old failures and unchanged flags. Official label source: https://github.com/actions/runner-images#available-images .

## I isolate the Xcode 27 external alias failure

Run39554417586/job119070724147 at8ffc01629 passes all six integrated/parser methods, but its two external methods fail twelve subcases (eight tests553.791s). Actual selected Apple21/Xcode27.2beta2 external assembler and debug expansion/replay probes pass. The retained capture-selection record pins compiler cb5fd7fc6cdf43f60a06a6038e49958d6364c73c6a5b05a507d45dbd36f2eef6 and the observed SDKROOT. This is not full capture qualification or a release matrix change.

I reproduce the external failure locally: a direct cc symlink makes Clang emit the otherwise identical unused-fPIC warning with cc: instead of clang:. The strict report parser refuses that text. An executable selector that execs the resolved absolute Clang with unchanged arguments restores its genuine diagnostic name without widening parser acceptance. I retain the failing alias, successful capture and actual library result42. The first local selector probe omitted SDKROOT and reached a separate System-link failure; the corrected selector retains explicit SDKROOT. An extra evidence check initially treated the probe-returned relocatable object as a directory/shared library; I retain that failed check separately from successful libanswer_native.dylib execution.

The corrected hosted diagnostic will retain executable wrapper hashes alongside the actual compiler hash and SDKROOT. Its eight-method step has15 minutes within the unchanged35-minute job: the preceding assertion-failure run already spent553.791s, and successful external cases add real work. Guest/tool budgets and all eight methods remain unchanged.

The two unchanged actual external SourceSnapshots methods pass locally in58.048s with the corrected executable selector, explicit SDKROOT and current aggregate production probe. I retain command/env/pins and the complete terminal. Hosted eight-method retry remains pending.
