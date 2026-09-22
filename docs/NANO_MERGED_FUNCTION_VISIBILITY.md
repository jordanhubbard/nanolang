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

## My absent-binding correction

Root's de325 source review found scope pollution before execution: a namespaced
module containing private str_split caused my global foreign-name scan to deny
a legitimate root builtin call. I retain the finding and correct the exact
lookup order. The existing Nano checker now exposes a predicate composed of
its existing is_builtin_function and known check_builtin_function result;
I do not add a second list of exceptions in the binder. The driver records
that fact beside each original function name/owner. Absent an actual local or
import binding, a known existing checker fallback remains unchanged; a foreign
nonbuiltin remains inaccessible. This preserves the checker's existing helper
fallback vocabulary too, without claiming that every helper is a C language
registry builtin.

Explicit private selective/qualified rows still resolve to denial markers
before any fallback. A wildcard or plain import does not select a private
binding; it cannot use a private row to hide an unrelated builtin. I add actual
namespace and wildcard private-str_split positives with a root two-string
split and exact segment assertions. All original explicit private/imported
value negatives stay unchanged. The binding shadow independently checks
absent-binding builtin retention and explicit private denial.

I checked the get_argc fixture against C source: get_argc has no row in
builtins_registry.c and no checker builtin registration. Its evaluator/native
runtime handler does not itself confer source visibility. The explicit extern
fixture therefore remains a private-function refusal control to qualify,
not an exemption based on runtime C spelling. My new checker-predicate shadow
requires get_argc to remain outside this builtin fallback inventory.

## My first current-producer terminal

At 387fc0 I retained equal 3789-source/tool endpoint maps and copied exact
93c C-seed providers. Direct context passed in 1.793 seconds. Retained f946
refresh2 compiled the current refresh1 with normal shadows in 31.807 seconds.
That current compiler's self-refresh stopped at native shadow C compilation:
18 diagnostics refer to seven inaccessible shared compiler helpers. It exited
1 in 19.032 seconds, without timeout or surviving descendants. Evidence,
generated C and both endpoints remain under
`/tmp/nanolang-split-387f-puck-evidence` on Darwin and in the copied Linux tree.
No paired method ran and no failed output was executed.

The actual driver calls parser_decode_import_path from parser.nano,
cg_append/cg_build/last_dot_index from transpiler.nano, and
mi_scan/mi_unique/mi_emit from module_introspection.nano. Their implementations
and existing shadows remain authoritative. I propose declaring exactly these
seven intended shared entrypoints public, plus ModuleIntrospection as their
public result type. I keep their signatures, bodies, local helpers and shadows
unchanged; I do not export all compiler internals or exempt compiler source
from access checking.

My source audit also shows that an unqualified denied function call can reach
the existing unknown-call return without producing a checker diagnostic.
I propose rejecting a resolved private marker immediately in the real
AST_CALL checker, before builtin and ordinary fallback. Qualified calls already
have this guard. I add an ignored-result private selective call control so
refusal cannot depend on a surrounding return/assignment type mismatch.
The original private and builtin-fallback controls remain required.

## My C selective-import prerequisite

At c481 both current refreshes passed (30.888 and 32.168 seconds), and the
actual-parser limit method passed on C seed and both current Nano producers.
The original/additive corpus then stopped at the C-seed private selective
alias control: `from owner import hidden_value as chosen` followed by a call
to chosen compiled successfully. The output sentinel changed; I did not
execute that product. The outer terminal was exit 1 after 53.566 seconds,
without timeout or surviving descendants. I retain both host evidence copies
under `/tmp/nanolang-split-c481-puck-evidence`.

My C module alias loop finds the target and copies its Function without
checking is_pub. The copy retains is_pub and module_name, but
is_function_accessible immediately allows all functions when the caller's
current_module is NULL. That includes an unnamed root importing a known-owner
private function. The same shortcut predicts an ordinary wildcard private
call gap. Function-valued identifiers also return TYPE_FUNCTION without
calling the existing visibility checker. I do not need another failing run
to establish these source boundaries.

Before source edits I propose these owning changes:

1. Selective function targets use exact requested module ownership, never an
   ownerless/global same-name fallback. Preflight every selected function's
   public bit before publishing any function alias for that import. Type-only
   imports keep their existing resolver; this is not a new type visibility
   implementation. Public alias spelling and original target remain separate.
2. A function with no declaring module keeps existing root/builtin treatment.
   A known declaring module must match the caller for private access; an
   unnamed root is not the same owner as an imported module. I do not invent
   a root module label or change builtin registry precedence. Public functions
   remain accessible subject to existing selective import rules.
3. Selective visibility checks use alias_of when present, because the import
   records the original declaration name rather than its caller spelling.
   Actual function-valued identifiers pass through the same accessibility
   check as calls. Local variable/callback lookup stays earlier.
4. Alias name/original-name allocation is staged and checked before a new row
   is published. Failure follows existing bounded module cleanup and stops
   compilation; I do not claim rollback of unrelated existing module state or
   a new recoverable general env_define_function allocation contract.

The original failed control remains unchanged. Public alias, private wildcard,
private callable value, ignored-result call, same-owner wrapper, builtin
fallback and output-sentinel controls remain in the complete paired corpus.
I add an unnamed-root public function-value positive and private function-value
negative to exercise the actual C identifier path. Changed module/checker
providers require fresh C-seed rebuild and endpoint maps; current Nano refresh
reuse, if proposed, requires exact exclusion/dependency proof. Complete clean
bootstrap and full SDK still remain later requirements.

My implementation uses exact-owner lookup and a public-function preflight
before any alias rows for the selected import. The alias helper stages both
owned spellings, leaving the output unchanged on failure; it deliberately
borrows the same declaration/signature/body fields as the existing alias row.
The normal environment publication and its terminal allocation policy remain
unchanged. Caller visibility distinguishes an ownerless root/builtin function
from an imported known-owner function, including function-valued identifiers.

The ordinary call path previously tried a same-name declaration before its
lexical callback fallback. To preserve local callable authority under the new
access guard, I use its existing check_indirect_call path first when an actual
visible TYPE_FUNCTION value exists. Earlier reserved and intrinsic routes
remain in their original order. An additive private-name/local-callback
positive requires the local target's actual result, alongside public selected
function-value acceptance and private wildcard function-value refusal.

My new C fixture includes the actual module translation unit. It checks exact
owner lookup with a competing ownerless and other-owner function, selected
public/private preflight without row publication, and both strdup failures in
the actual alias helper. Failure leaves a memcpy-preserved output sentinel and
zero live helper allocations; each failure is followed by normal recovery.
Successful owned spellings survive changes to their source buffers. Other
signature/body/module fields retain the existing borrowed ownership contract.
The fixture uses complete ordinary provider objects excluding main/module,
with its own CLI state. It does not claim whole-provider sanitization or
unrelated environment rollback. No new product execution has occurred.

My 99a Darwin C visibility checkpoint passes the rebuilt C-seed corpus and
actual alias allocation fixture. The next original Nano record-initializer
negative compiles successfully; I do not execute it. The paired outer terminal
is exit 1 after 123.573 seconds, without timeout or cleanup error, with all
process groups and descendants absent. Complete source/tool endpoints remain
equal. My retained Nano producers remain attributed to c481; this is not a
fresh bootstrap. Evidence is retained in
/tmp/nanolang-split-99a-puck-evidence and
/tmp/nanolang-split-99a-puck-continuation on both hosts.

Before correction, my Nano record-literal checker returns the record identity
without traversing initializer expressions. My existing field mutation adapter
already compares actual and declared types, but constructor fields bypass it.
I propose a shared declared-field STRING-array compatibility boundary using
actual parsed field annotations and check_expr_node results. I retain complete
array<string> results from the existing call checker, including declared and
lexical calls; no str_split spelling shortcut supplies authority. Known
STRING-array mismatches must refuse in either direction. Empty array contextual
behavior and unrelated numeric/computed-array compatibility remain unchanged.
I require original record initializer/mutation negatives and additive positive
STRING fields, non-STRING declared-call fields, and nested record initializer
controls before another producer refresh. General record shape completeness
is not established by this bounded prerequisite.

My bounded implementation traverses each actual field expression once before
comparing it with its matching parsed declaration annotation. Recursive calls
through check_expr_node validate nested record literals. The new comparison
uses existing types_equal only when either side has a known STRING-array
result; empty unknown-element arrays retain contextual compatibility and
unrelated numeric comparisons remain unchanged. I retain the returned record
identity and existing field-shape policy. No initializer expression is executed
by this checker. A direct shadow checks nested mismatch diagnostics and empty
STRING-array acceptance using real parsed records; full paired programs add
matching nested fields, declared/local array-returning callees, reverse mismatch
and nested refusal while preserving the original record-initializer negative.
This checkpoint is source reviewed only until its next qualification approval.

The 1ee9 refreshes pass (29.674/29.994 seconds), but the new reverse-direction
record initializer reveals a C gap: [41] into array<string> is accepted. My
existing complete-result helper checks actual STRING-array metadata only.
Scalar literals do not retain that complete annotation; the later field check
compares only ARRAY. Before changing source, I propose keeping complete result
facts first and using existing infer_array_element_type for otherwise missing
array element facts. I compare known STRING against the other known element
kind in both directions at the existing contextual helper. Unknown/empty
elements retain contextual behavior; unrelated numeric compatibility remains
unchanged. This is checker inference, not executing an initializer. The same
existing lexical/declaration authority remains in inference. Original and
additive refusal programs remain unchanged.

My C correction keeps a complete result element (including UNKNOWN) when
present, and invokes existing array inference only when that view is absent.
The symmetric check runs only for two known element kinds with STRING on at
least one side. Empty/unknown and unrelated numeric cases follow the existing
remaining helper paths. I retain the original actual-STRING diagnostic and
add a precise expected-STRING diagnostic. Six direct parsed-literal controls
exercise both mismatch directions, matching STRING, empty STRING/INT and
unchanged numeric helper behavior without emitting or executing invalid code.
The original full paired reverse control remains unchanged. No execution has
yet qualified this correction.
