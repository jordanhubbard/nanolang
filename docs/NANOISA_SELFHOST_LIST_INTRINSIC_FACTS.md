# My self-hosted list intrinsic facts

I record this repair before changing source. My cb489 fresh builds pass on both
hosts, but Stage1 rejects Stage2 with nineteen discarded `list_*_set` calls and
one `str_trim` array-destination error. My isolated 1dca diagnostic Stage1 builds
in 49.251s; its one original-source check exits1 in 163.650s with unchanged
inputs and no timeout. Every discarded error identifies a real set call: eight
in parser.nano, six in nominal_bindings.nano, three in nanoc_v06.nano and two
shadows. My old helper recognizes only new/push/get/length. I retain both first
preparation mistakes, both original bootstrap terminals and the measured log;
none is qualification. Task_873238ff98c845a9bf94ec78a03e2950 owns the prerequisite.

## My exact list contract

I check real supported list calls before the generic name-to-result fallback.
A visible local value, function or extern declaration keeps ordinary callable
checking and wins over an intrinsic spelling. Qualified calls resolve their
actual declaration; an unrelated module suffix never grants an intrinsic.
The call's declaration owner and existing nominal bindings resolve the encoded
element. I do not infer element identity from a current global name or accept
an unresolved leaf. Scalar lists retain their existing supported catalog;
ordinary record lists require one exact resolved record declaration. Enum,
resource, resource-bearing and unresolved record elements keep their existing
refusal boundaries. Full enum-list parity remains separate required work.

I preserve the existing shared foreign-record namespace: a uniquely registered
extern record may retain its unmangled identity where that namespace already
permits it, with competing extern or ordinary declarations refused. This is
not a fallback to any globally unique ordinary name. In a merged source graph
ordinary declarations require their actual module binding (including aliases).
Standalone parser controls resolve local declarations without inventing module
bindings. I audit the existing nb_register/nb_import/nb_type maps and canonical
record names before choosing the concrete helper, including imported schema
records used by the compiler itself.

| Operation | Arguments | Result |
| --- | --- | --- |
| new | none | exact List<T> |
| get, remove | exact List<T>, INT | exact T |
| pop | exact List<T> | exact T |
| push | exact List<T>, exact T | VOID |
| set, insert | exact List<T>, INT, exact T | VOID |
| length, capacity | exact List<T> | INT |
| is_empty | exact List<T> | BOOL |
| clear, free | exact List<T> | VOID |

The table follows the already supported C record-list operations; it does not
admit additional scalar operations lacking actual providers. Each selected
operation checks exact arity, receiver facts and all supplied argument facts.
Index facts are INT, not unknown or an enum relabeled INT. Writes require the
resolved element identity after an explicitly supported destination conversion;
I do not weaken global types_equal. A failed call emits a diagnostic and yields
unknown, never a guessed valid result. Get/remove/pop retain the exact element
inferred type. Bounds and mutation semantics stay with existing checked runtime
and emitter paths; this frontend repair adds no runtime call or opcode.

I place the selected-intrinsic dispatch before the existing argument scan so
its arguments are checked once in left-to-right order. Ordinary calls retain
their existing scan and contract checking. I remove the old unchecked name-only
fallback for the selected list operations so wrong receiver/argument facts
cannot bypass the new checker. Any unsupported operation stays refused.

## My separate trim signature boundary

C's registry declares str_trim(string)->string, and the existing evaluator and
native emitter already implement it. I restore exactly that selfhost call fact
only for an actual unshadowed intrinsic: one known STRING argument, STRING
result. I audit bare, qualified, declared and function-valued call paths so a
same-spelled declaration keeps its own signature and a qualified suffix is not
mistaken for this intrinsic. I do not give an identifier a callable signature
merely by returning STRING from a name lookup. Wrong arity/type is diagnosed;
unknown never becomes STRING to satisfy an array destination. Other missing
builtins and unsupported target routes are not silently admitted by this fix.

## My source and acceptance sequence

I submit the complete checker helper/dispatch source checkpoint before gates.
Meaningful shadows cover all supported operations, exact arities, INT indices,
wrong nominal receivers/elements, undefined and colliding declarations, module
aliases, extern identity, visible function/extern/callback precedence and exact
get/remove/pop result facts. Trim controls include typed array destinations,
wrong arity/argument, visible declaration and qualified-name precedence.

I keep every existing seventeen-method assertion and the original LexerToken
program. Additive source controls exercise local and imported record lists and
actual mutation/result behavior across the existing supported producers and
runtimes. A newly discovered unsupported backend route remains an explicit
prerequisite, not a removed assertion or claimed acceptance. Fresh schema,
providers, build/bootstrap, focused allocation/sanitizer matrices and the full
source corpus remain required on both hosts under unchanged deadlines and
capacity guards. Broader list/enum/full-Make/release parents remain open.
