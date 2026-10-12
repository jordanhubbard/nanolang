# My same-basename control

I first omit module declarations: the ordinary C loader rejects the inferred duplicate introspection identity shared before effect checking, while my self-hosted producer publishes it. This is the existing broader module-identity limitation recorded in the roadmap; I do not claim to resolve it here.

I then add distinct declared introspection names LeftShared and RightShared, retaining those final sources here. Both producers compile and execute the same-named effects with result 8. The named-results.json receipt covers that positive VM control. I do not count the unnamed divergence as passing parity.
