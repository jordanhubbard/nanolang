# I audit my original public C requirement

I audit task_6ade6d62ef644390bb9645b15077c8df against c4c039a4 (PR791, canonical merge pending at this report's first publication). The original task asks me to inventory supported options and expression contexts, replace GNU block-expression dependence with portable lowering **or checked refusals**, and run ordinary generated C99/C11 controls. It does not ask me to invent arbitrary missing source ABIs. This is a finite acceptance audit, not a whole-language conformance or release claim.

## I map the original clauses to evidence

| Original clause | My resulting boundary | Evidence |
|---|---|---|
| GNU AST_BLOCK expressions | Supported function-body values use real scoped statements and typed destinations. Continuing/control-only completion propagates through nesting; unsupported insertion contexts refuse before publication. | PR775; `evidence/public-c-expression-lifting.md` |
| Expression contexts | Exact scalar/local union block, match, IF and COND values can enter supported initializers, assignments, return operands, calls and nested eager/lazy operands. Repeated lifted loop headers and lifted global startup refuse; I do not replace these with comma-only block semantics. | PR775 source/interpreter/VM and API controls; finite table below |
| Union match typeof | Named declaration/variant payload types replace typeof. Scoped payload binding precedes guards; first successful named arm wins; exact coverage and one final wildcard are supported. Unsupported wildcard/no-success profiles refuse. | PR760,766; `evidence/public-c-match-payloads.md`, `public-c-guarded-matches.md` |
| Exact union results | Prototypes, definitions and direct calls share exact local nongeneric owner/result identity; scalar/empty payload ABI remains bounded. | PR769; `evidence/public-c-union-results.md` |
| Advertised options | Every CBOptions member has an explicit meaning or refusal; see complete option table below. | PR777; `evidence/public-c-profile-options.md` |
| Captured-lambda claims | Anonymous/captured metadata and callable-valued storage/calls receive checked refusal. Direct scalar/nominal declarations are not mistaken for unsupported function values. | PR777,780; `evidence/public-c-callable-profile.md` |
| Ordinary C99/C11 controls | Focused fixtures compile and execute at O0/O2 with GCC/Clang and the established sanitizer/strict-standard flags; selected source routes have independent VM/interpreter observers. | Each linked sealed evidence record; latest PR791 passes3 focused and63 adjacent methods per compiler, plus seven programs |
| No inference of whole-target conformity from scalar checks | I enumerate refusals and evidence limits explicitly. Source checker acceptance alone does not establish backend support; API fixtures do not establish source grammar. | This audit and all per-child contracts |

## I enumerate every current AST family

I inventory the 47 ASTNodeType entries in src/nanolang.h. “Admit” below always means the checked, exact supported-type profile, not every arbitrary AST combination. Unsupported expression dispatch records an error; statement default dispatch uses that same path. Top-level program declarations are processed by their declaration passes, not promised as arbitrary executable statements. I do not claim malformed-AST fuzz coverage.

| Nodes | Reached profile and evidence |
|---|---|
| NUMBER, FLOAT, STRING, BOOL | Admit exact scalar literals; float bit transport and decoded C-string bytes retain their documented boundaries. PR748,756 and binary64/literal evidence. |
| IDENTIFIER | Admit known lexical scalar/nominal values; unresolved or callable-valued references refuse. PR780,782. |
| PREFIX_OP | Admit existing exact scalar arithmetic, truth/comparison, ordered string equality/concatenation and supported lifted operands. Unsupported operand facts refuse. PR748,753,758,775; scalar contracts retain their own scope. |
| CALL, MODULE_QUALIFIED_CALL | Admit exact direct declared identities/signatures and qualified emitted aliases; selected builtin exact arity/types. Local/expression callees and missing qualified declarations refuse. PR780 plus conversion fixtures. |
| ARRAY_LITERAL | Refuse, including retained array metadata in annotations/calls and32 registered builtin array names, without a new array ABI. PR779. |
| LET, SET | Admit supported exact local storage and supported lifted initializer/assignment contexts. Scalar globals use ordered startup; unsupported types/global insertion contexts refuse. PR748,775,782. |
| IF, COND | Admit supported ordinary conditions/branches and typed lifted values with branch-local lazy preparation. PR775 and seven ordinary programs. |
| WHILE | Admit ordinary supported headers and scoped bodies with real enclosing break/continue; lifted repeated headers refuse. PR775,784. |
| FOR | Refuse at the profile boundary. Canonical range remains a valid source construct elsewhere; old bare-count C fallback was not a source-language contract. PR784. |
| RETURN, BREAK, CONTINUE | Admit in valid enclosing checked control contexts; lifted completion stops consumers/tails after control-only paths. PR775,784. |
| BLOCK | Admit scoped statements and supported typed lifted values; unsupported value insertion contexts refuse. PR775. |
| FUNCTION | Admit exact supported declarations/direct-call ABI; anonymous, captured, generic, callable-valued or other unsupported signature facts refuse. PR777,780,782. |
| SHADOW | Source-only C output does not execute shadows; the driver/source-test selection policy is separate. This statement marker emits no program semantics. PR777 scope and PERSONA test policy. |
| PROGRAM | Container for checked declarations, prototypes, ordered scalar startup and hosted entry. Both public entrypoints stage semantic output. PR748,777. |
| PRINT, ASSERT | Admit supported scalar/string observers and assertions, including lifted operand preparation in supported contexts. PR748,749,752,775,787. |
| STRUCT_DEF, STRUCT_LITERAL, FIELD_ACCESS | Admit exact local complete explicit-field records and checked resolved scalar/nominal fields. Prior declaration order is required for by-value fields. Spread, unsupported field types, absent/wrong owners and incomplete cycles refuse. PR782,791. |
| ENUM_DEF | Existing integer carrier/declaration spelling remains; PR782 API carrier controls are not ordinary enum-field source syntax qualification. No new enum ABI is claimed. |
| UNION_DEF, UNION_CONSTRUCT | Admit exact local nongeneric scalar/empty variants, named payload types and exact checked construction; invalid owner/payload facts refuse. PR760,769,782. |
| MATCH | Admit exact covered supported union cases, scoped guards and supported typed value lifting. Early/multiple wildcard and incomplete coverage profiles refuse. PR766,775. |
| IMPORT, MODULE_DECL | Frontend-resolved declaration metadata; no emitted runtime statement. Qualified calls still need exact retained emitted declaration identity. PR780. |
| OPAQUE_TYPE | Refuse the unsupported type profile. PR782. |
| TUPLE_LITERAL, TUPLE_INDEX | Refuse, including tuple-valued storage facts. PR777,782. |
| QUALIFIED_NAME | No generic expression lowering; default expression refusal. Qualified direct calls use the separately checked MODULE_QUALIFIED_CALL path. Static dispatch audit; no new ABI. |
| UNSAFE_BLOCK | Existing real scoped statement block; contained calls/types still undergo profile checks. This is not an FFI ABI expansion or safety proof. Static dispatch audit and existing ordinary statement paths. |
| PAR_BLOCK, PAR_LET | Existing serial binding/statement lowering; flow requests use passive_binding_order. The passive design specifies source order for par and stable topological order for flow. I make no new scheduler, speedup or full passive-contract claim from this audit. |
| TRY_OP, EFFECT_DECL, HANDLE_EXPR, EFFECT_HANDLER, EFFECT_OP, ASYNC_FN, AWAIT | Explicit checked refusal at reached top-level/statement/expression profile boundaries; no stub value or invented runtime ABI. PR777. |

The inherited PAR/UNSAFE/metadata rows describe existing statement/declaration handling from static source inspection; I do not relabel them as new focused all-combination execution evidence. The original defect's expression/option boundaries have dedicated controls. Public C remains a bounded checked-AST target.

## I enumerate all options and public publication paths

| Option/API | Actual contract and evidence |
|---|---|
| NULL options / all-zero defaults | Hosted output with ordinary wrapper and established literal lifetime; PR777 actual compilation/execution. |
| no_stdlib=true | Checked refusal; no freestanding implementation claim. Both APIs preserve old bytes. |
| no_main=true | Omit hosted wrapper only; keep language functions callable. Globals refuse because no initialization entry is supplied. Library linking, source main/no source main and uncalled main are tested. |
| static_strings=false/true | Explicit compatibility alias: both retain static literals; scalar conversion and concat snapshots use separate checked process-exit ownership. Actual literal contents observed. |
| verbose | Diagnostics only, no semantic profile change. PR777 option matrix. |
| c_backend_emit | Stage semantic output, close sibling temporary, rename only after success. Semantic refusal preserves existing path bytes. |
| c_backend_emit_fp | Stage semantic output before copying to caller stream. Semantic refusal preserves caller bytes; final external I/O failure can leave partial copied bytes. I do not promise transactional external streams. |
| Invocation reset/recovery | Independent namespace/error/planning state and valid same-process recovery after refusal, repeatedly checked in API suites. |

The CLI initializes defaults and exposes --target c, output selection and existing verbosity. This audit does not invent no-stdlib/no-main CLI flags. Generated-code strictness is C99/C11 -pedantic-errors plus established implicit-function/return/local-typedef errors and ASan/UBSan; API harnesses additionally use -Wall/-Wextra/-Werror. Global unused-helper cleanliness is not claimed.

## I keep broader work distinct

Task_d0c6c784248b4716a1e3c215d5568793 describes broader repeated lifted loop-header/global admission. Existing checked refusal satisfies this original portable-or-refusal requirement; it does not complete that independent admission task. Tasks477bdd/70c5 and parent7a99 retain shared wildcard/no-success policy obligations. They are not silently implemented by the public C common-profile restriction. Full scalar-policy, platform and release gates retain their own acceptance. I do not replay historical compiler failures or infer a fresh bootstrap from these C-target checks.

PR791 canonical merge and independent acceptance review remain prerequisites to completing original6ade. I propose closure only of the original bounded inventory/portable-or-refusal task once those prerequisites are established, not whole-target or whole-language conformance.
