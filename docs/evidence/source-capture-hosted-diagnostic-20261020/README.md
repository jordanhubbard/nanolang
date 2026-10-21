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
