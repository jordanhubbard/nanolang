# My map constructor parity checkpoint

I track the complete outcome in #1083, under #1081 and #976. My original
`tests/unit/test_map_constructor_contexts.nano` remains unchanged. I retain
all four int/string key/value pairs, every original context, enumeration
snapshot and clear assertion.

My checker currently returns an unknown named `HashMap` for the empty builtin.
A nested generic-union payload therefore loses its concrete expected type.
I refine only a zero-argument identifier call to `map_new`, with an exact
supported scalar-pair expected type and no real same-named nominal declaration.
I leave general unknown-type equality unchanged. My mandatory shadows include
all four concrete payload pairs and wrong-pair, unknown-value, wrong-arity and
same-named nominal controls. These shadows still require fresh execution.

I expand the existing Python matrix to require the seed NanoISA producer and
both fresh installed stages. Each emits a distinct verified module, executes it
in my VM, translates it to C and executes the result under ASan/UBSan. I retain
the original C-native seed counterpart and expand the existing refusal controls
to both stages. The Make target now requires these actual providers. My hosted
diagnostic workflow runs this complete matrix after fresh bootstrap even if a
focused string test fails, then still collects the complete original JUnit routes.

My local actual-lexer check accepts all 12 static Python fixture fragments;
Python AST, workflow YAML and diff-whitespace checks pass. These checks do not
establish compiler or native parity. Local disk pressure holds substantial builds.

My producer still needs all four typed key/value pairs, keys/values snapshots
and once-only clear. My C AOT analysis/runtime still needs corresponding
representation, alias, lifetime and allocation-refusal coverage. I keep #1083
and both roadmap rows open. This checker foundation does not complete the map
outcome, ordinary-record publication, native adoption, or release acceptance.

## My producer implementation pending execution

I now carry each of the four supported key/value pairs through my self-hosted
map type helpers, exact HM_NEW operands, key/value checks and inferred
keys/values array types. I refuse uncontextualized construction rather than
silently selecting STRING-to-INT. My map clear lowering saves the receiver once,
takes a detached key snapshot and deletes each key in reverse snapshot order.
It emits existing VM operations and leaves no result on the statement stack.

I add mandatory shadows for all constructor operand pairs, typed enumeration,
clear lowering, missing context and wrong arguments. I add runtime controls for
all four pairs: snapshots survive replacement/deletion, empty snapshots grow,
clear preserves aliases and evaluates a returned receiver once, maps can be
reused, and integer keys retain signed extremes. I retain every original
constructor assertion and seed-native counterpart. My Python actual-lexer check
accepts 20 static fragments. These source checks do not establish executed
producer semantics; the fresh hosted run and the C AOT/runtime work remain open.
