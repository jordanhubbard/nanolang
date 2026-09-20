# My private Wasm restack after the shared C checker repair

I merge actual PR910 canonical6b86de83d into reviewed PR906 head6825961f7.
The sole conflict is the additive roadmap tail; I retain both ordered records.
My [identity record](portable-read-wasm-restack-910.json) proves all15 previously
reviewed non-Make production/fixture inputs and complete Make bytes unchanged.
Both original report manifests are byte-identical. Incoming typechecker.c,
general VM fixture and C match fixture are exact canonical copies.

The C checker correction extends match metadata lookup to call expressions.
It is not in the direct private Wasm guest/host or native adapter build. My
ordinary declaration query links NanoISA objects and utf8, without typechecker.o;
its providers, fixtures and recipes are unchanged. Broad historical source
inventories remain attached to their original pins; I do not relabel them as
this integrated tree. PR910 separately qualified its checker and fixture repair.
No affected compile/link closure requires another engine/query/sanitizer matrix,
so this restack adds static identity evidence and no replay claim.

Full b7ef/2d2 source, authority and installed acceptance remain open. I do not
claim previously failed full-suite or queued hosted checks now pass. Root review
and actual merge still precede bounded task reconciliation.
