# My record literal evaluation order

I keep task_865923a37b6d470c83326aed625b8897 open until actual paired source and backend checks pass. Static inspection found both NanoISA emitters visit declaration fields and evaluate the matching source expressions in that order. My evaluator visits explicit source fields in their written order. I preserve source order and evaluate every expression once before declaration-order packing.

I first resolve and validate the complete mapping from each explicit field name to one unique declared field. Missing/duplicate/unknown fields refuse before publication. I stage each source expression in a fresh anonymous local in source order, with its declared expected scalar type, then load those local roots in declaration order for AGG_PACK. My ordinary existing local-count limits still apply; I do not borrow a visible name or reuse a source variable's slot. After packing I clear staged locals with VOID stores, retaining the packed record as the sole result owner. This must work inside repeated loops without retaining the last temporary graph until function return.

For the C producer's existing spread form, I evaluate the base once first. I snapshot every inherited, non-overridden field into an owned local before evaluating explicit overrides, matching my evaluator's scalar snapshot order. Explicit overrides then execute once in their written order. I pack in destination declaration order and clear all private staging locals, including the base. Nested reference values retain their existing alias/value semantics; this checkpoint does not redefine deep-copy behavior. The independent Nano parser's missing spread syntax remains required full-graph work; I do not substitute the C producer for it.

I require actual side-effect traces with reversed declaration/source order, unchanged ordered literals, nested literals, overwritten spread fields, an override that mutates an inherited scalar in the source record, once-only base evaluation and repeated loop ownership. I preserve first failure terminals and original shadow policy. Generated C/LLVM/Wasm routes qualify where their existing record admission supports the exact fixture; any missing route remains a full5.1 obligation.

## My native producer prerequisite

The same static audit finds both native C producers put side-effecting field expressions directly in designated compound initializers. C does not establish my written source evaluation order for those expressions. Thus native output is not an independent correct oracle until it is repaired too. I extend the same task to emit a private typed record temporary and sequenced field assignment statements inside the existing expression-block mechanism. I snapshot inherited fields before explicit overrides; every explicit assignment retains destination C conversion semantics. Temporary names must be disjoint from source bindings and remain safe under nested literal scopes. Actual source-order assertions, rather than agreement with the old native output, are my authority. I review both native and NanoISA changes before testing.

## Native union initializers

I found the same static ordering gap in native union designated initializers. I include both native producers and exact payload evaluation traces in task_9a1ba277fe3e45d5913d03305ece1e06 before implementation. My NanoISA union staging already follows written order; this finding does not qualify native behavior.

## First NanoISA source checkpoint

I implemented complete field mapping, anonymous staging, written-order evaluation and declaration-order packing in both NanoISA emitters. My C producer additionally snapshots inherited spread fields before overrides. Declared byte destinations retain contextual narrowing, and staged locals are cleared after packing. Native producers and Nano spread parsing remain required.

The C producer builds and runs my new exact reversed/ordered/nested/repeated-loop source assertions successfully on Linux. The first full self-hosted emitter build refuses publication after four existing nisa_emit_function shadow expectations still require the old record constructor local counts. I retain that terminal. Each two-field literal now deliberately needs two anonymous slots; I will update those exact counts while retaining all payload/instruction assertions. This is not a full source/backend qualification.

## Corrected bounded producer check

At source79fe52cdb, my full C-seed build of the Nano emitter passes its mandatory shadows after the four exact local-count expectations include two anonymous roots. Both actual NanoISA producers emit, verify and execute the same unchanged record fixture: written and reversed order, nested construction, exact field payloads and20 repeated loop iterations. My automated test_record_literal_order regression also passes. Commands, first failure, corrected build log and products are retained in docs/evidence/record-literal-order.

This is a Linux check of C-seed NanoVirt and the C-seed-built Nano emitter. Stage1/Stage2-built emitters, Darwin, native producers, spread side effects, byte destinations, failure boundaries and complete generated backend acceptance remain required before closing this task. I also recorded native tuple/array ordering under task_0ef97cccc6c441c59bd9eb2ec7c5ff05; their designated/variadic expressions need the same written-order audit.

## Native source checkpoint

I now sequence C-seed and Nano native record fields and union payloads into private typed temporaries. Both C-seed union AST forms share one ordered emitter. C-seed spread snapshots inherited fields before explicit overrides; Nano spread parsing remains open. Temporary selection avoids visible bindings and function symbols; nested literals have separate C expression scopes. Empty records retain the existing GNU empty-initializer form. I preserve the existing flat record and GC ownership conventions; this sequencing change is not a new deep-ownership guarantee.

Static review also finds the Nano native global-literal fallback returns0 for a record initializer outside its explicitly runtime-initialized union route. Global record construction therefore needs a separate complete initializer correction under this task, with actual global source tests; I do not use local-only ordering checks to close global acceptance.

My first complete component build with native sequencing passes, including parser/checker/transpiler shadows. Actual C-seed native compilation and execution pass both record and union trace fixtures, including nested records, zero-payload variants and an empty record. Self-hosted native products still require fresh bootstrap and their own source runs.

## Complete nonliteral global initialization

Task task_57661a28bf114448a4ee14642e7e1b90 covers the broader same static fallback: a function call, field/reference, array, tuple or callable initializer can also fall through to0. I will classify only NUMBER/FLOAT/STRING/BOOL as direct literal initialization. Every other expression must use generate_expression with its declared expected type in the existing ordered startup function, including guarded primitive initialization. I retain declaration/prototype order, direct literal behavior and first-error refusal. Actual C-seed/Stage1/Stage2 compiler and source tests must cover observable initialization order and stored payloads; source-tree inspection alone is not acceptance.

Before executing the broad global fixture I also found C-seed global declarations discard record identity and omit tuple/callable typedef selection. I include exact annotation-based declaration emission and precollection of global tuple/callable types in the same initializer task. I retain ordinary runtime assignment order; I do not use inferred anonymous C types at file scope or erase nominal names.

The complete component build passes with the nonliteral classifier and C-seed global declaration correction. Actual C-seed native output passes global record, array, tuple, reference, function-call, function-value and mutable-call initializer assertions, including captured initialization traces21 and213. My new native gate requires all three real compilers and keeps source/output artifacts. Fresh self-hosted bootstrap and that complete three-producer gate remain required; neither the component-build banner nor C-seed execution substitutes for them.

## First three-producer native gate

My 216ab bootstrap passes, including installed-compiler independence. In /tmp/nano-native-literal-order-z2tus_tj I retain all gate commands and outputs: C seed passes all three fixtures; Stage1 and Stage2 pass record and union fixtures but reject global initialization with unknown nl_FnPtr_0. I generate globals after flushing the function-type registry. I will prepare global text before that flush, preserving emitted typedef/prototype/global order, and test the unchanged payload/effect fixture.

## Tuple and scalar-array sequencing

My corrected callable-global bootstrap and unchanged nine native producer/fixture combinations pass. Next, under task_0ef97cccc6c441c59bd9eb2ec7c5ff05, I replace native tuple compound initializers with typed temporary field assignments in both producers. For C-seed scalar arrays I stage each operand before entering variadic helpers. I retain their existing argument conversions and ownership conventions; I require trace and payload checks before qualification. My self-hosted array path already emits sequential pushes. Full contextual types, ownership, platform and optimization coverage remains required.

My first corrected C-seed tuple/array fixture compiles and runs successfully, with exact tuple21, array321 and nested-array123 traces and payload assertions. I retain its command/output directory at /tmp/nano-tuple-array-cseed-first-rfoz7zvg. Fresh tuple/array bootstrap is running; no Stage1/Stage2 or Darwin tuple/array acceptance claim yet.

## Scalar variadic boundary

My ordered staging preserves operand C types, but helper varargs read fixed types. I ledger task_8c8a62af03c043e49594d49f9c74a70d before changing this boundary. I will explicitly convert U8 through uint8_t then promoted int, float through double, bool through int and string through const char*. INT already converts through int64_t. I do not run mismatched old varargs to establish this static defect.

My dbac tuple/array bootstrap and all twelve Linux producer/fixture combinations pass. I preserve commands, logs and product hashes in docs/evidence/record-literal-order/native-tuple-array. This checkpoint covers integer/nested-array ordering and prior record/union/global regressions. The next scalar ABI fixture extends float, string, bool and byte payload/effects; it does not replace the earlier evidence.
