# My retained native callback signatures

I compare complete annotation trees when matching a named function, a local function value, or a forwarded function-valued parameter against a declared callback signature. This includes nested arrays and concrete generic arguments. I reconstruct an owned signature from a function declaration, release temporary signatures on both outcomes, and share recursive copying with the interpreter and module metadata pipeline. `type_infos_equal` provides structural comparison; callers resolve nominal aliases before using it.

My C callback typedefs use retained parameter/result annotations. A callback returning `Box<int>` now returns the concrete native union instead of an unspecified scalar C type. My existing selfhost typedef mapping already preserves that spelling after the companion identity repair.

Ten callback methods exercise 30 C-seed/Stage1/Stage2 decisions: ordinary generic results and parameters, local and forwarded values, nested arrays, and rejected wrong named/local/forwarded/nested signatures. Rejected programs preserve an existing output artifact and fail before C compilation. Four adjacent selfhost generic-context methods also pass; the combined 14-method run takes 61.299 seconds.

I passed a fresh default-budget native bootstrap and the complete parser/typechecker suites after integration. Parser tests compare nested generic annotations, mutate parameter/result trees to require mismatches, deep-copy signatures, reconstruct signatures from declarations, and free the original AST before reading the copies. The parser suite also passes ASan/UBSan with the existing separate legacy leak boundary; I do not claim leak-free whole-checker execution. The first-class user-guide example compiles, checks its shadows and executes.

This completes MAC `task_cf555f2a672e43d9921ca44b817ec631`. Parent `task_e05a42e2e09b47cc9c53fa6923eeeaef` remains open for complete indirect-call argument/result metadata, qualified-module call signature checks, serialization and conservative resource callback rejection. A qualified imported callback still reaches a separate basic argument-checking path; the module metadata work does not close that checker gap.

Evidence is retained in `/tmp/nanolang-callback-foundation-integrated-{gates,drivers}.log`, `/tmp/nanolang-callback-signature-{parser,asan}.log` and `/tmp/nanolang-callback-firstclass.log`. Independent source review found no blocker within this bounded scope.

## My indirect and qualified call follow-up

I retain an owned checked signature on indirect calls so later bytecode
publication can recover concrete result metadata after lexical scopes end.
My AST copier and destructor copy and release that context. I compare full
argument annotations, including nested array literal leaves, and route C
qualified calls through the ordinary argument checker. My self-hosted
qualified call checker compares callback signatures explicitly.

My first repaired native bootstrap passed. Its paired gate exposed that
qualified imports still accepted a mismatched callback in Stage 1 and Stage
2; I repaired the separate self-hosted branch. Independent review then found
that a record literal without retained TypeInfo needed a nominal declaration
comparison. I added matching and mismatched literal regression cases.

After integrating the resource callback boundary, a fresh default-budget
native bootstrap and the complete parser/typechecker suites pass. Thirty-five
paired methods pass in 117.266 seconds: 18 callback methods, 13 resource
callback boundary methods and four adjacent generic context methods. These
include computed callees, nested literal rejection, qualified import execution
and rejection, and VM execution with retained callback signature metadata.
Rejected programs preserve their existing output artifacts.

This completes `task_3ca0e46fbbc64aa8bc39bfdaf65b8833` after merge. I do not
infer complete callback ownership from these ordinary call checks. The failed
paired log is `/tmp/nanolang-callback-call-context-all-drivers.log`; final
bootstrap, unit and driver evidence is retained under
`/tmp/nanolang-callback-context-final-{bootstrap,units,drivers}.log`.
