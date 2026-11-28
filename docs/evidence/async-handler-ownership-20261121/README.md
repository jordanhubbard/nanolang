# My async syntax and handler metadata ownership controls

I repair two measured leaks under #1169/#1170/#1149. My parser owns the function below `AST_ASYNC_FN` and the expression below `AST_AWAIT`, but `free_ast` omitted both children. My checker also copied handler parameter names, then restored `symbol_count` without releasing the metadata in those discarded rows.

## My bounded baselines and repairs

I add an ordinary parser regression that retains async wrappers, an awaited nested call and callable parameter annotations, then frees three independently parsed trees. I enroll it in the original parser suite. Separate bounded drivers execute that same regression and the unchanged original async-handler method, including all100 calls, trace/result checks and coroutine-slot checks.

My matching ASan/UBSan/leak baselines use the explicit existing sanitizer object root. The parser control leaks9294 bytes in84 allocations; the original async-handler control leaks4090 bytes in40 allocations. I first add only the two owned-child cases to `free_ast`: the parser control becomes clean, while the original async-handler method retains2 leaked bytes in one checker-created parameter name. I preserve that intermediate image/report before changing the checker.

I then release owned symbol names and nominal-name copies at the checker's existing handler-local discard boundaries, before restoring its original count. I preserve borrowed annotation and placeholder-value ownership and retain ordinary scopes as before. Both matching bounded controls now pass with no leak report. I preserve the exact original assertions and sanitizer options.

My first baseline builder accidentally omitted `OBJ_DIR` while slicing an earlier command. I retain its actual mixed-provider images, source and completed leak diagnostics, but exclude that attempt as a matching fully instrumented baseline. My correction uses the explicit original sanitizer object root before any parser repair. My post-gate provider inventory is labeled as collected afterward; I do not claim uniform project instrumentation or describe a later rebuilt global archive as unchanged earlier bytes. The actual baseline/intermediate/repaired images retain their linked code.

## My original ordinary and product gates

I use a fresh ordinary object root and rebuild the actual default-path native test fixture with matching flags before running ordinary checks. Every selected source and tool hash remains unchanged across those gates.

- My complete ordinary parser suite passes75 named methods, including the added regression.
- My complete ordinary evaluator passes all134 named methods.
- My complete ordinary checker and effect suites pass87 and35 named methods.
- Every method name is observed in the retained terminal; no source-main method is missing.
- My two bounded ordinary drivers and all twelve unchanged effect/handler ownership methods also pass.
- My original C-producer async/refusal/imported-identity/effect VM/native product gate passes48 methods in59.268s. This includes source-created modules and O0/O2 sanitized native scalar async products. I explicitly select the actual C producer for the parameterized effect fixtures; these results do not establish new self-hosted or Darwin qualification.

My ordinary Make gate deletes its temporary complete unit-driver binaries. I retain its complete commands, source hashes, discovery audit and terminals, together with the bounded actual driver images and source-built product tools. I do not claim that every successful temporary product remains available.

## My remaining sanitizer boundary

The complete unchanged original evaluator sanitizer run still fails:8909 bytes in286 allocations, with the last visible method `eval_match_miss_is_terminal`. I retain that failure, its actual complete driver and matching default-path native fixture. I do not claim all134 sanitizer methods completed. Broader callable/record/evaluator ownership remains open under #1149/#1140/#1116, and I add no suppression.

The [manifest](manifest.json) retains114 logical files,396462653 raw bytes and76916755 unique compressed bytes. I verify every raw and compressed SHA256 after writing. It includes exact original/intermediate/repaired sources and images, complete full-sanitizer/ordinary/product terminals, discovery, selected provider identities, builders and the working source diff from `bdffa4ccc`. My first archive inventory stops on a continued shell-command line; I retain that refusal, join continued lines and record any unparsed inventory lines explicitly. It changes no product check.

Both coordinated Darwin bootstrap handles and their frozen inputs remain untouched. They qualify their recorded older source pins, not these later parser/checker changes. Additional heavy Darwin gates wait for their terminal results. The installed host-library closure, complete pure translator, paired/final-head evidence and full release remain open.
