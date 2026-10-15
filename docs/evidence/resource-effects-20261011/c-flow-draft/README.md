# My C resource-effect ownership checkpoint

I analyze transferred effect arguments, resumed resource results and repeatable handler parameters in my C source checker. I refuse captured owner access, dropped or reused parameters/results, unrelated suspended owners, and staged resource values abandoned during later argument or constructor evaluation.

I compute escaping operations to a fixed point over direct calls, including transitive and mutually recursive calls. A local handler contains only its matching operations. I preserve lexical callback identity, inspect known higher-order callbacks, and conservatively refuse unresolved callable effects while an owner is suspended. I retain allocation failures as ownership errors.

My final source/native matrix passes 22 methods, including positive transfer, handler consumption, inferred Handle forwarding, lexical return, staged/spread ownership, local handlers, partial handlers, callback shadowing, and declared array operation effects. See c-declaration-final.log. The inferred Handle native binding defect is corrected in type-name inference; c-native-matrix.log retains the original failure.

My 36 adjacent C ownership controls pass. My allocation gate passes ASan/UBSan with leak detection, checking 300 owners and 1,832 injected allocation failures plus effect-summary allocation refusal. The final corrected run is c-lookup-final-asan.log. The C typechecker/transpiler gates and adjacent effect tests are retained here and in the array operation evidence.

I retain the development counterexamples: staged nested perform, staged/live owners across indirect calls, forward-declared local handlers, inferred native Handle bindings, and registered array aliases. The initially accepted indirect-call leaks are refused by my fixed-point summaries. The array alias defect and later declaration-priority defects are corrected and tested separately in [my array operation evidence](../../array-builtin-aliases-20261011/README.md).

This is a C source/native checkpoint. My self-hosted checker still needs resource handler flow, and my owned NanoISA source profile still refuses these programs before publication. I have not implemented affine bytecode handler/resumption contracts or qualified complete runtime cleanup. publication-boundary.json retains that refusal with prior-output preservation. Full resource-effect ownership remains open under #976.
