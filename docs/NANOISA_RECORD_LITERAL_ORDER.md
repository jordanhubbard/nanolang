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
