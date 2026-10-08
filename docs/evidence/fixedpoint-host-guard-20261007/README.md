# My fixed-point host-cache guard checkpoint

I built a clean isolated checkout at `7dc131e449342119be3248771348dd1e97e32cfd` on Darwin. My seed generation passed in 16.200 seconds and verified in 0.090 seconds. Stage 1 exited one after 160.822 seconds: my native compiler guard rejected the compiler-support host artifact’s retained assembly snapshot in its current cache directory. I did not reach Stage 2 or compare module bytes.

My retained manifest, rejection record and logs identify the source pin, immutable seed host libraries and exact refused path. The compressed seed retains that checkout’s absolute import paths; it is evidence, not a relocatable release artifact.

I correct the test guard to read a list of retained filenames under cache roots derived from the verified seed’s actual import closure. The list contains only snapshot indices and object names derived from those declared host modules’ metadata. Generated `.c` inputs remain refused. Focused tests execute admitted assembly and objects, and reject generated C, unrelated roots, non-staging directories, undeclared indices and symlink escapes. I also canonicalize the existing Darwin fixture’s temporary root and compare identity-probe status with the actual compiler, whose linker rejects Linux `--version`.

Both guard tests pass. The corrected complete raw fixed-point run remains pending; this checkpoint does not establish a fixed point or release readiness. MAC ready-task lookup returns `Operation not permitted`.

My first corrected full run at `bfb2c3944` reaches shared-library linking, then refuses the combined `compiler_support.o` object. My initial list had its per-source objects but omitted the combined module object. I retain this second terminal separately as `combined-object-*`. I move filename derivation into the guard helper, include the combined declared module object and use that same helper in the executable multi-source guard fixture. Both guard tests pass after this correction. The full comparison remains pending.
