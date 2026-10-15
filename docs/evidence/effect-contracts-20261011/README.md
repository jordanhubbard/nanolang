# Effect return and declaration contracts

I lower an explicit return of a void perform or handled body by discarding its
VM void sentinel before RET. I retain a refusal for a nonvoid effect result in a
void function. My direct-perform and handled-body controls execute with results
7 and 9 through both frontends.

I retain complete C effect-result annotations through parsing, nominal binding
and AST cleanup. Declaration validation runs after nominal declarations are
visible, so forward references remain valid. I reject duplicate operation names,
unknown concrete parameter/result types and incorrect generic argument counts.
I recursively check the tested array, list, map, tuple, callback and generic-union
forms. The legacy List element-name representation receives an explicit check.
My self-hosted checker uses an effect-specific recursive validator; unrelated
reduce admission remains unchanged.

My first collection audit exposed C acceptance of List<Missing> and self-hosted
refusal of HashMap<string,Point>. I retain both failures and invalidate the first
compiler audit when correcting its inputs. The final source tests pass 45 cases
per frontend: nine executed handler/return programs, ten execution/signature
refusals and twenty-six declaration controls. Refusals preserve prior output.
The C parser, typechecker and effects component suites pass. My final seed runs
all 1,137 compiler shadows without failure. The final nine-step compiler audit
passes with all twenty tracked production inputs unchanged: seed build, self-
compilation to NanoISA, verification, standalone C translation without a VM wrapper,
strict C compilation, CLI execution, all 45 effect cases, four handler metadata
methods and eight opaque/union/layout regression methods. Installed generations
and exact release-platform qualification remain open.

My new parser test inspects nested array, tuple and function result annotations.
It passes 100 parse/free cycles with ASan, UBSan and leak detection. The broader
parser sanitizer run reports 488 bytes in six allocations across malformed
parameter/borrow, import and statement paths. The pre-change parser and tests
reproduce the same report. I retain that independent cleanup gap; I do not claim
the whole parser leak gate passed. The baseline uses the current header layout
and other linked objects with the pre-change parser/test source.

Overall effect declaration admission remains open. My C frontend accepts an
unknown field type in an unused row-record effect signature; the self-hosted
parser refuses both that source and the valid row-record control. Imported effect
identity, full nominal/generic execution, resource ownership and native nvm2c
handlers remain unqualified. These are NanoISA/VM effect execution tests; a native
compiler executable passing them does not establish native effect execution.

GitHub DNS/API access remains unavailable. I retain the coordination/update bodies
and failures without claiming publication or complete release qualification.
