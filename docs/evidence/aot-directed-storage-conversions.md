# My directed storage conversions

I separate aggregate storage conversion from strict shape equality. Record
arguments and returns register directed constraints. I solve them after every
function has contributed its exact shapes, before resolving native storage.

I copy representation facts into independent destination nodes. A present
string can feed an optional destination's string payload. A destination string
inferred by conversion can widen to optional storage, with a fresh payload
node. An exactly constrained string destination cannot widen. Optional payloads
and map components retain exact compatibility requirements.

I use a worklist and retain aggregate cycles when copying missing destination
edges. Primitive payload nodes are not reused as optional wrappers. Repeated
passes propagate facts through conversion chains until no facts change.
Strict unification remains unchanged.

My graph tests cover both source orders, shared string payloads, recursive
record/array cycles, idempotence, later facts, incompatible optional payloads,
exact destination constraints and cleanup. My nested optional-return acceptance
fixture now assembles, executes in NanoVM, translates to C, compiles with
warnings as errors and executes natively in all four combinations of function
order and ordinary/tail calls.

Full compiler acceptance remains open. It passes the former `env_get_type`
tail-call conflict and now stops at `STORE_LOCAL` in `check_match_expr` (319),
offset 650, with optional/string field shapes. That assignment boundary is
tracked separately as `task_0557eb72ec60494983314ecb11a31a36`.

`make -j1 test-nvm2c` and `make test-nvm2c-sanitizers` each pass 1,572 AOT and
1,073 shape checks. Leak detection is disabled; I do not infer leak freedom.
`make -j1 test-one-ir-compiler` has one remaining failure, the full compiler;
both focused source/bytecode acceptance cases pass.
