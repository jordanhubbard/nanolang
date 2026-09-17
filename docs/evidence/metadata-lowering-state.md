# My defensive metadata lowering boundary

I stop indexed module-name lowering when its index expression fails. I retain the first diagnostic and do not allocate selection labels or intern exported names after that failure. I also validate every introspection declaration before lowering public shadow modules, including unused declarations. My program emitter already applies that declaration rule.

I track this bounded repair as `task_a600db3ae64f4f63b8b85d77856c593a`, stacked on product source `63876a4d41b1716ad6f8c1eb2652af2dc03f2cab`. Neither finding establishes a cause or resolution for retained export-shadow abort `task_dd74b033c3984805bc27ce5017096c3c`. I do not replay its historical binaries or fixture in these gates.

My new inline shadows check failed index emission, unchanged label/string tables, the original unsupported-extern diagnostic, and unused malformed declaration refusal through both public APIs. My new ordinary Python fixture uses inferred mutable int/string locals inside a shadow, unsafe metadata assignments, empty and nonempty export tables, absent indices, and a counted index expression. It checks both fresh compiler stages and verifies and executes their products in VM and native C translation.

The initial failure-state test ran with whole-source mode, which defers extern registration. Its assertion therefore did not exercise the intended immediate index failure. I retained `/tmp/nanolang-metadata-state-build.log`, set program mode explicitly in that test, and ran fresh bootstrap again. This was a test setup correction, not an attribution of the historical product incident.
