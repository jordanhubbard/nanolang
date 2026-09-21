# My merged function visibility

I record this precode dependency under the open File/SDK parent
`task_8bbc1cf5295b4b59b314640ef57c725f`. My frozen 6d7861 Darwin evidence is
`/tmp/nanolang-split-6d78-puck-evidence`: the direct context fixture passed,
C seed passed every 26-source control, and refresh1 passed eight positives
and five typing refusals before the unchanged private imported function
compiled successfully. I did not execute the invalid output. Refresh2 was
not reached. The paired outer terminal was exit 1, without timeout or cleanup
failure. I preserve this first terminal and every original assertion.

## My source boundary

My merger strips `pub` before the merged parser, my ASTFunction has no public
field, and `bind_merged_functions` currently creates import bindings for every
function belonging to the target owner. This loses an actual language fact.
My C checker already distinguishes same-owner access from public cross-owner
access; an extern declaration is not automatically public.

I retain public declaration text through merging and capture visibility in my
actual parser dispatch. I reuse ParsedDeclarations rather than invent a second
visibility grammar. My current parser_store_function returns result_kind -1;
I must not pretend the existing generic result tag identifies a top-level
function. I will return an explicit function index from the actual top-level
function/extern dispatch, with an absent value for other definitions. The
ordinary parse_definition wrapper preserves its Parser result. A top-level
function is stored after its body, so lambdas appended during that body are
not the top-level export. Shadows and global initializer lambdas remain
internal even when another declaration is public.

I extend captured definition metadata with this exact index. I keep default
parse_program behavior and the File descriptive capture limit of 4096.
Ordinary merged compilation requests a separately bounded complete capture
using token_count as an upper bound on successful top-level iterations; the
existing progress check requires each iteration to consume tokens. I refuse
an incomplete capture before binding. I do not impose the File-specific
4096 limit on ordinary programs, claim a total compiler heap bound, or change
existing source/token limits and process allocation-failure behavior.

Before mangling I build one visibility entry per parser function, initially
private. Each actual top-level function index must be in range, unique and
within its captured before/after extent. Its canonical owner comes from the
existing merged source-line map. I validate all facts before mutating function
names or publishing import bindings. Declaration spans and generated helpers
remain owned by the parser; temporary visibility vectors live through binding
and do not escape as borrowed metadata.

Same-owner bindings retain every function, including private helpers and
extern declarations. Cross-owner bindings include only explicitly public
functions under their original names, then apply actual import qualifier,
selective alias and wildcard rules. Qualified/private lookup cannot fall
back to a foreign unqualified declaration or builtin. I audit selective
symbol validation against C behavior before changing unused-import refusal;
this correction must not silently claim complete type/re-export namespace
validation. Public nominal type registration and its existing import policy
remain distinct from function visibility.

## My acceptance dependency

I keep the original private Provider.str_split control unchanged and require
all original split controls on C seed and both newly refreshed Nano producers.
I add ordinary function names so the repair is not a builtin exemption:
public/private qualified imports, explicit selective aliases, wildcard
visibility, same-owner private helper calls, public wrappers calling private
helpers, private/public extern declarations, and a public function containing
a generated lambda. I assert nested/generated functions are never exported.
Parser shadows cover public syntax, comment/newline boundaries, exact capture
indices, partial parse failure and existing File truncation semantics. An
ordinary capture above 4096 declarations verifies that the File cap does not
narrow default compilation.

My source and complete fixtures require review before execution. Refreshed
Nano binaries must include this changed parser/driver closure; the previous
f946 binaries are retained evidence, not current visibility products. I retain
normal shadow selection, publication sentinels and ten-second bounds. Passing
this prerequisite still does not substitute str_split into my merger or close
the required later clean C-seed bootstrap and complete installed SDK corpus.

## My implementation checkpoint

My actual dispatch now returns DefinitionParseResult, with a function index
only for its function/extern/pure branches. The ordinary wrapper returns the
same Parser. ParsedDefinition retains that index separately from the legacy
result tag, which remains unchanged. Complete merged parsing requests the
actual token count as its capture bound; the File API still requests 4096.
I retain `pub` lines unchanged through merging, including comments and line
maps. Before function-name mutation I validate captured index uniqueness,
function extents and agreement between declaration-token and function owners.

My function visibility vector uses three states: absent top-level declaration,
private declaration, and public declaration. Same-owner rows remain complete.
After installing actual local/public imported bindings, I install inaccessible
import fallbacks in the remaining names. These point to an unbound name under
the existing collision-reserved prefix, not to a function or C symbol. A
separate complete original-name/owner table prevents unbound foreign lookup
from falling through to a builtin or an unmangled foreign extern. Lexical
values still take precedence through the existing checker/emitter scope
paths. Private fallbacks are installed last so they cannot hide an available
public import merely because a private declaration was visited earlier.
Qualified checking rejects these markers before its historical suffix helper
fallback. Identifier-value checking consults the resolved name for builtin
fallback too. I preserve private declarations for their own helpers/shadows.

I also retain same-owner extern/source collision refusal regardless of which
appears first; repeated extern-only declarations remain allowed. I do not
claim a new general signature compatibility rule for repeated externs.

My additive paired fixtures retain all original split controls and add five
positive visibility programs and seven private visibility refusals. Public
wrappers exercise private same-owner helpers, lambdas remain internal, and
public/private extern controls use the existing get_argc runtime boundary.
Positive imports cover qualified, selective alias and wildcard forms. Private
callable-value controls cover qualified names and a builtin spelling. Every
negative preserves its output sentinel and refuses before host compilation.
A separate actual-parser program parses 4097 declarations at runtime, checks
File truncation, complete ordinary capture and default parser retention. Its
safe main shadow does not perform that large runtime loop. Each producer
compiles with normal mandatory shadows; I retain full C-seed selected names
and completed JSON, compare the complete Nano selected multisets, and require
the specific declaration-capture shadows. No product gate has run at this
checkpoint.
