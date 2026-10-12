# My imported effect implementation evidence

I continue GitHub #976 without closing the full 5.1 contract.

I bind effect declarations to their defining module, retain the original spelling, expose public imports under their selected names, and filter handler inference by visibility. Both parsers accept qualified performs. My self-hosted binder also rewrites operation parameter/result nominal types in their defining module.

My paired-imports and paired-regressions logs cover C and self-hosted producers with VM and sanitized native execution. I include same-named effects, distinct imported record signatures, qualified performs, selective aliases, private local handlers, and prior-output-preserving private and type-mismatch refusals. My self-hosted seed passes 1,149 shadows. The full C typechecker and transpiler gates pass.

My direct native matrix exposes two separate type-name lifetime defects: a handler result borrows an annotation refreshed during nested checking, and a wrapper result borrows the shared formatting buffer changed by its parameters. I retain both failures. My final native-route-final.log passes all ten methods after retaining owned names at both boundaries. native-route.py runs the permanent matrix through nanoc_c rather than the default NanoISA route.

My final eight-step compiler-product audit passes with unchanged source inputs. I compile myself to NanoISA, verify the module, translate it to strict native C without a VM wrapper, and execute the focused effect/union-array matrix through the resulting compiler. The entry matrix includes both producers and the latest distinct argument/result record fixture; the resulting product also passes 44 adjacent generic/purity methods. I retain commands, timings, source and artifact hashes in product-audit/product-manifest.json. This qualifies the changed implementation area, not my full release candidate.

My same-basename control passes both producers when modules declare distinct names. Without those declarations, my C loader rejects ambiguous introspection identity while the self-hosted producer accepts it. I retain that existing broader module-identity limitation in same-basename; I do not claim complete module parity.
