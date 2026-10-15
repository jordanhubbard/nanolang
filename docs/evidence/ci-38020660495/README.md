# I retain the failed verifier corpus at030b12862

[Run38020660495](https://github.com/jordanhubbard/nanolang/actions/runs/38020660495)
has a failed Memory Sanitizers job114120724818. Its verifier corpus selected185
inputs, verified184 and failed compilation of `tests/nested_structs.nano`.
I retain the job tail here. The complete failure log lives in the uploaded
`verifier-corpus-sanitizers` artifact under
`.test_output/verifier-corpus/corpus.7QP54x/0011-nested_structs.compile.log`.

Two artifact downloads failed connecting to api.github.com. The job summary
does not establish the cause; I track diagnosis under #982 and do not claim
release acceptance from the other passing jobs.

I independently reproduce E027 locally: legacy `println(line.start.x)` spelling
parses the parenthesized qualified name as a zero-argument call. I change the
fixture to canonical `(println line.start.x)`, preserving every field read and
shadow assertion. C bytecode compilation, `nano_vm --verify-only` and execution
then succeed with the retained expected output. This local result does not
replace the missing hosted compile log or the fresh hosted sanitizer gate.
