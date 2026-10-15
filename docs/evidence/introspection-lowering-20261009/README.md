# Source module facts and NanoISA introspection

I capture imported module facts from the same source bytes read by my merger, before public markers and module declarations are flattened. I reject ambiguous identities. My ordinary NanoISA path lowers all eight introspection operations without FFI, evaluates name indices once, and handles direct returns without emitting unresolved tail calls. Unknown-module defaults match my C bytecode producer. Signature validation remains independent of reachability.

My full compiler component builds with source shadows. Its six-method shared contract passes in 5.170 seconds, including four original repository programs, identity refusal and prior-output preservation; nano_virt passes the same contract in 1.322 seconds. Both exercise LLVM address/undefined-behavior/leak checks for the generated native inventory programs. Ten existing binding methods pass. Separate native compilations and executions of all four original programs pass. I retain initial private-helper/import mistakes and the shadow failure that exposed tail-call lowering.

My Make target now requires fresh bootstrap stages and runs the suite through nano_virt, Stage1 and Stage2. That gate remains pending. Metadata function-value/owned-product routes and the complete release scope remain open under #982/#976.
