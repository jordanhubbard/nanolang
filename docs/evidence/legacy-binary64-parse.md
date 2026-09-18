# My legacy binary64 parsing evidence

I implement required companion `task_9e93c1badb1a4da093a737b3a2c15ef7` after
canonical PR657. Contract `ad2a2598` precedes production `eefa97f7`.

I share exact prefix values through a checked C-string adapter in my AST
interpreter and both legacy generated C runtimes. My evaluator's strict
`cast_float` still requires a nonzero endpoint at NUL and returns positive zero
with a diagnostic otherwise. My `string_to_float` routes still accept a prefix;
I do not add string admission to compiled numeric `cast_float` helpers.

My optional parser endpoint includes leading whitespace/sign only when there
is a conversion, consumes only complete decimal/hexadecimal exponents (also
for zero mantissas), and distinguishes full infinity and valid nan payload
suffixes. An absent conversion has endpoint zero. Both outputs publish only
after checked parsing succeeds. Existing value-only callers use the same API
wrapper and retain their previous exact results.

My endpoint sanitizer gate passes 26 ordinary value/endpoint controls. The
source fixture passes C-seed, Stage1 and Stage2 compilation/execution with
mandatory shadows; the paired two-method log is
`/tmp/nanolang-legacy-parser-paired.log` (9.156 seconds). Fresh three-stage
bootstrap, parser and typechecker gates pass in
`/tmp/nanolang-legacy-parser-bootstrap.log` and
`/tmp/nanolang-legacy-parser-parser-checker.log`. The unchanged 1,131-case
value corpus also passes native sanitizer and import-free Wasm core checks in
`/tmp/nanolang-legacy-parser-core-reference.log`.

I retain a separate full evaluator gate failure: the run stopped with SIGSEGV
at existing `test_eval_shadow_tests`, before reaching the new conversion test.
I did not re-execute that failed artifact or establish attribution. The log is
`/tmp/nanolang-legacy-parser-c-gates.log`; the preserved binary is
`/tmp/nanolang-legacy-test-eval-failed-791a` with SHA256
`e3067a5c49d2a6d470cbbbab977ae578e74808d889d690a79eed9b32ee01b83a`.
No core file was located under the host's apport configuration. Required task
`task_791a3d3d66b04c0caa892a3d5b99ccca` records this unresolved acceptance.

My focused evaluator driver invokes only the new conversion test through the
public REPL AST entry. It checks prefix results, strict suffix/no-digit errors,
embedded NUL, signed zero, exact NaN payloads and infinity boundaries. It neither
executes nor closes the separately failed general shadow gate. I retain this
scope distinction in the target name and source comment.

Darwin managed sanitizer task7ba and broad managed parent51da remain open.
No full evaluator, all-platform or release acceptance is claimed here.

My combined frozen gate passes the refreshed three-stage bootstrap, focused
evaluator driver (including exact special-value diagnostics), two legacy parser
methods across all three compilers, the adjacent legacy float-conversion method,
two managed package methods, three runtime-core methods, 22 canonical managed
methods and shared profile decisions. I retain the complete log in
`/tmp/nanolang-legacy-parser-integrated.log`. This result does not alter the
separate unresolved full evaluator failure recorded above.
