# My resource effect baseline

I retain #976's full resource-bearing effect requirement. My transfer fixture constructs one Handle, moves it into an effect, resumes the same value, and consumes the result exactly once. Both current producers refuse it at ownership analysis: C reports a boundary needing ownership lowering, while the self-hosted producer reports expression node 42 without ownership lowering.

My double-use and drop variants also refuse and preserve prior artifacts, but this blanket boundary refusal does not establish the desired path-sensitive ownership rules. I require positive transfer/resumption, operation argument consumption, arm-local ownership, lexical-exit cleanup and invalid-use refusal across the frontends, verified NanoISA, VM and native execution. These baseline logs establish missing implementation, not completed acceptance.

I run the C producer bin/nano_virt and the qualified native compiler from /private/tmp/nl51-imported-effects/product-audit/compiler, at source checkpoint 60e6a0ae1. The current import-alias changes do not affect these probes.
