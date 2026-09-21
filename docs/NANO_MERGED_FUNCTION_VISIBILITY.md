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
