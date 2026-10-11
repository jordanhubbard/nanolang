I continue #976 from 5c1bf97ea. I corrected native C layout-cycle refusal: direct self-edges now participate in dependency ordering, and stalled graphs fail before publication instead of being emitted in source order.

All six cycle controls preserve prior output with the intended diagnostic. The full 16-method layout suite passes under instrumented emission, including arrays, lists, callable signatures and generic-array neighbors; ordinary C controls and the transpiler unit gate pass. Evidence: docs/evidence/native-layout-cycle-refusal-20261011/README.md.

The self-hosted declaration validation gap remains open. This C emitter correction does not supply a language-wide frontend check or close aggregate payload/full release qualification. I coordinate through Git while the GitHub API is unavailable; MAC is not used.
