# My collection/effect baseline

I ran four source cases through the C producer and the fresh Nano compiler from
my qualified global/effect product. Both reject direct array resumption and
array-valued performed arguments followed by lexical return at effect-signature
lowering. Both reject the resource-child/array payload earlier at retained record
construction. The wrong-element control reaches type checking and is correctly
rejected. Every attempt preserves the prior artifact. These failures establish
remaining work; they do not qualify collection effect execution.

I retain initial fixture errors separately: `>=` tokenization without a separator,
a direct resource-array declaration, and the reserved field name `handle`. The
final fixtures use separated tokens and a `token` field. Those parser errors do
not explain the final lowering failures.

My implementation has three disconnected requirements:

- `ownership_effects.inc` describes scalars, strings and exact record/union layouts,
  without an exact collection-element descriptor.
- The private array declaration projection in `ownership_contracts.c` explicitly
  rejects EFFECTS; its ARRAY_FIELDS contract does not grant handler authority.
- Owned-array origin/authority analysis and affine handler analysis must agree on
  transferred elements, captures, resumed values and lexical cleanup before either
  VM or native admission can expand.

I must preserve exact nominal element identity, source evaluation order, resource
obligations, unchanged outputs on refusal, and runtime cleanup on allocation
failure. I will extend declaration and flow contracts before connecting both
producers, then qualify VM/native execution and a fresh compiler product. Borrowed
continuations and the rest of 5.1 remain open.

`baseline.py` records commands, statuses and logs in `baseline.json`. Its driver
path names the retained compiler product; `inputs.json` pins both drivers and the
sources used here. No product source changed in this baseline checkpoint.

My later [array declaration checkpoint](contracts/README.md) retains exact
element tables; executable integration remains open.

My [typed-slot checkpoint](slots/README.md) connects the element table to
function results, parameters and locals, with executable admission still closed.
