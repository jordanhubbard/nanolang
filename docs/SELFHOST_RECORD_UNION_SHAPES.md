# My self-hosted record and union shapes

MAC `task_9f67cec0f9a440ff86983ba451645d4b` tracks thirteen unchanged #522
acceptance cases: record arrays in union payloads and concrete generic union
fields in ordinary records. At `a0dfa86be`, every original source passes my C
bytecode producer's shadows, VM execution and instrumented native execution.
My repaired self-hosted producer now passes these thirteen cases through both
fresh native stages. I retain [the source repair evidence](evidence/pr522-implementation-2026-09-23/record-union-source/README.md).

I admit finite ordinary records, concrete unions and nested arrays whose leaf
is a supported scalar, enum or ordinary record. Records and unions share the
ancestor path so recursion across either declaration kind is refused. I bound
shape traversal, retain exact generic arity and substituted field types, and
keep resource-bearing records and collections on their owned boundary. A
phantom generic argument does not itself store an owner. I preserve existing
extern record admission when explicit fields describe a supported finite layout;
my compiler contracts use this path for `List<CompilerDiagnostic>`.

I use a record field's declared type to lower an inline union constructor,
including empty variants and nested generic constructors. Context supplies
missing type arguments; it does not replace a conflicting explicit declaration
or argument. The constructor emitter still checks declaration identity, field
names, counts and exact payload types. Ordinary expressions still require the
exact declared type. Nested array literals retain their declared element type.

I require the thirteen original accepted sources through raw production and
shadow emission, VM and native execution, then all original accepted/refused
record-field and union-context cases through fresh native stages. I retain
wrong nominal identity, wrong concrete arguments, unknown fields, cycles and
resource-bearing global/collection refusals with prior output intact. Existing
selected generic ownership and scalar/array emission gates remain required.

This work does not clear the three retained aggregate/integer-selector match
result failures or the complete integration, fixed-point and release gates.
