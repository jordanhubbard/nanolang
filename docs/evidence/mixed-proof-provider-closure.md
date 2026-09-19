# I restore both mixed-proof provider lists

I qualify source `64db8f922e7a24e5e4b86794f66fe3d1f0c076b9` for MAC
`task_f8f7c5140f1949b3b8bf876de58e54d8` and [PR817](https://github.com/jordanhubbard/nanolang/pull/817).
My [manifest](mixed-proof-provider-closure.json) records first-failure and corrected
reports, complete tracked-source maps, actual tool hashes and retained binaries.

My first Darwin STRING bootstrap at `e543f78b` stops on missing mixed-proof
symbols. The wrapper-only correction `27a8464ca` passes my Linux wrapper gate
in13.046seconds, but bootstrap stops after43.162seconds because the NanoISA
module manifest independently omits the same source provider. I preserve that
checkout and its first terminal logs; I do not replay its failing artifact.

My corrected fresh Linux checkout adds the object to the native wrapper list
and the source to `modules/nanoisa/module.json`. Independent static review checks
both lists and their direct dependencies. Existing wrapper tests pass in13.048
seconds; full `make bootstrap` passes in247.110seconds. Source, tool and HEAD
maps remain unchanged during both gates. Bootstrap success is not a canonical
bytecode fixed-point claim. I retain existing generated-C const warnings in the log.

My corrected Darwin wrapper gate passes five C cases and seven Python checks
in7.53seconds. Fresh SDK-aware bootstrap passes in301.32seconds with Apple
clang21.0.0 and SDK27.0, including both compiler stages and installed/no-C-seed
smokes. I independently hash all5326 current tracked files,12 selected built
tools and six actual tool files through SSH against the preserved maps; the
initial/frozen source maps and normalized actual tool hashes agree. The
wrapper-only Darwin attempt at27a remains a preserved bootstrap failure after
104.51seconds, despite its8.01second wrapper pass.

I copy and hash those reports in the manifest. Homebrew LLVM23.1.1 is reserved
for subsequent sanitizer source gates, not mislabeled as the bootstrap compiler.
Darwin STRING prerequisite setup separately hit Apple Make3.81's unsupported
`--eval`; the peer preserves that harness result and uses a supplemental Makefile.
It does not invalidate these completed wrapper/bootstrap gates. Full STRING
source/runtime acceptance remains task793523 work.

I add no runtime/source admission and do not publish a release from these checks.
My provider-closure implementation and platform gates are complete; canonical
landing is recorded separately in MAC.
