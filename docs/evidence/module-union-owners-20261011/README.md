# My module-owned union evidence

I continue #976 from 9c74deecb. I bind internal union declarations to their module owner, retain independent original-name metadata, and resolve qualified aliases against their exported owner. Same-owner aliases stay compatible; equal-shaped declarations from different owners remain distinct.

I recover match declarations from registered generic instances in both native and NanoISA emission. Splitting at the first underscore loses canonical owners and ordinary underscore-bearing names. Both parsers now accept qualified ordinary and explicit generic union constructors.

My C NanoISA and native routes pass 36 generic/union methods. The NanoISA route verifies and executes modules, then translates and executes strict ASan/UBSan C AOT products. A separate malformed-constructor method checks four refusals and preservation of prior output. All 37 methods pass with seven freshly instrumented C objects (environment, evaluator, module, nominal binding, parser, transpiler and checker); I use a 65,520 KiB stack and disable interpreter leak detection. This is not a leak-freedom claim or an instrumented NanoVirt audit.

My parser, typechecker, environment, metadata-lifetime and transpiler gates pass. The self-hosted parser component builds and executes its shadows. The metadata lifetime control also passes with the instrumented objects, independently releasing AST, environment and copied metadata in different orders.

I retain the adjacent suite's 17 failed subcases in installed generations. They concern record-array union literals, primitive-list payloads and by-value cycle refusals. Current-product attribution and exact release/platform qualification remain separate from these focused results.

My fresh self-hosted seed passes 35 of the 37 focused methods; equal-name merged declarations still misclassify contextual constructors. I retain that failure before correcting the parser. The separate duplicate-declaration-in-one-owner refusal passes. My adjacent current C run has two diagnostic failures for layout cycles; my fresh self-hosted run has ten failed subcases for aggregate payload lowering and cycle refusal. I keep those current-product gaps open.

My unchanged-source eleven-step compiler-product audit passes in 511.053 seconds (product/manifest.json). Both its seed and native self-hosted product reproduce the two contextual-constructor failures before the final parser correction. The correction keeps constructor syntax until owner binding; its isolated parser component shadows pass. The retained full audit predates that final correction and is not exact-source qualification of it.

My corrected fresh seed passes all 38 focused native methods and all 15 union methods through explicit NanoISA emission, verified NanoVM execution and strict ASan/UBSan C AOT execution. These include equal-name modules in both import orders, distinct layouts, same-owner aliases/callbacks, root collisions, ordinary and explicit qualified constructors, cross-owner and ambiguous/unknown-owner refusals, malformed constructors and duplicate declarations within one owner. Source-input hashes remain unchanged during this corrected build and qualification (corrected-source-inputs.json). The final exact-candidate Linux/Darwin gates remain open.
