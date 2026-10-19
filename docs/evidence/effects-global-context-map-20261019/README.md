# I map checked global-effect context

The exact NanoVirt and fresh Stage 1 providers from `16eb3b1f9` execute both local-global fixtures successfully. For immutable reads the seed records effect mask 0; for mutable reads it records mask 2. The self-hosted producer conservatively records UNKNOWN (8) in both cases because its advisory analysis has no checked global symbol context. This is missing metadata parity under #998, not an execution or authority bypass.

I retain source, exact commands, producer hashes and canonical dumps. This probe does not qualify a fix. Retaining the actual checked symbol context must preserve work budgets, owner/generation validity, failure/reset behavior and raw/generic UNKNOWN.

I normalize trailing display whitespace in copied text. `retained-original-hashes.json` preserves original paths/hashes and identifies each normalized copy.
