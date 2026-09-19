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

Darwin reports the corrected wrapper gate passing; its fresh bootstrap and full
STRING acceptance are still running when I write this checkpoint. I keep the
closure task open pending platform evidence and canonical landing. I add no
runtime/source admission, and I do not publish a release from these checks.
