# My retained native callback signatures

I compare complete annotation trees when matching a named function, a local function value, or a forwarded function-valued parameter against a declared callback signature. This includes nested arrays and concrete generic arguments. I reconstruct an owned signature from a function declaration, release temporary signatures on both outcomes, and share recursive copying with the interpreter and module metadata pipeline. `type_infos_equal` provides structural comparison; callers resolve nominal aliases before using it.

My C callback typedefs use retained parameter/result annotations. A callback returning `Box<int>` now returns the concrete native union instead of an unspecified scalar C type. My existing selfhost typedef mapping already preserves that spelling after the companion identity repair.

Ten callback methods exercise 30 C-seed/Stage1/Stage2 decisions: ordinary generic results and parameters, local and forwarded values, nested arrays, and rejected wrong named/local/forwarded/nested signatures. Rejected programs preserve an existing output artifact and fail before C compilation. Four adjacent selfhost generic-context methods also pass; the combined 14-method run takes 61.299 seconds.

I passed a fresh default-budget native bootstrap and the complete parser/typechecker suites after integration. Parser tests compare nested generic annotations, mutate parameter/result trees to require mismatches, deep-copy signatures, reconstruct signatures from declarations, and free the original AST before reading the copies. The parser suite also passes ASan/UBSan with the existing separate legacy leak boundary; I do not claim leak-free whole-checker execution. The first-class user-guide example compiles, checks its shadows and executes.

This completes MAC `task_cf555f2a672e43d9921ca44b817ec631`. Parent `task_e05a42e2e09b47cc9c53fa6923eeeaef` remains open for complete indirect-call argument/result metadata, qualified-module call signature checks, serialization and conservative resource callback rejection. A qualified imported callback still reaches a separate basic argument-checking path; the module metadata work does not close that checker gap.

Evidence is retained in `/tmp/nanolang-callback-foundation-integrated-{gates,drivers}.log`, `/tmp/nanolang-callback-signature-{parser,asan}.log` and `/tmp/nanolang-callback-firstclass.log`. Independent source review found no blocker within this bounded scope.
