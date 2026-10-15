# My array operation identity checkpoint

I route array_map, array_filter and array_fold through checked map, filter and reduce operations without replacing their source spelling. I resolve lexical callables and real function declarations before builtin checking, inference, interpretation and native emission. My C checker also accepts these declarations in modules.

My eleven-method matrix passes through the C NanoISA producer and the native self-hosted compiler product, in VM and ASan/UBSan native execution. The same eleven methods pass direct C-native compilation and mandatory shadows. I exercise the alias pipeline, record-array inference, string accumulation, signature refusals with prior-output preservation, lexical callable shadowing, differing declared signatures and results, and qualified module declarations. See declaration-module-corrected.log and declaration-module-native.log.

My eight-stage compiler-product audit passes with unchanged inputs, including self-compilation (317.5 seconds), bytecode verification, C translation, native compilation, 29 effect/array methods and adjacent metadata tests. I ran that audit before the additional C declaration-priority fixes. Its manifest describes those exact earlier inputs; checkpoint-inputs.json identifies the later C correction. My self-hosted emitter is unchanged between them. The expanded matrix directly tests that native compiler product against the corrected C producer.

My final seed rebuilt by the corrected C compiler passes 1,151 mandatory shadows and all eleven expanded methods in VM and sanitized native execution. See seed-final-shadows.json and seed-final-matrix.log.

My C typechecker, transpiler, evaluator, environment-scoping and 28 native callable-array methods pass. Twenty-one adjacent effect/signature/import methods pass through the corrected C producer in VM and sanitized native execution. My source/native resource-effect matrix passes 22 methods; this does not admit resource effects through verified NanoISA.

I retain the failures that led to these changes: registered aliases initially lack native/NanoISA lowering; an initial include order breaks C compilation; an imported unittest class accidentally discovers installed-compiler tests; declared calls initially use builtin checks and shadow dispatch; and module registration initially treats the registry entry as a duplicate. The corrected runs are separately named. Full resource-effect ownership remains open under #976.
