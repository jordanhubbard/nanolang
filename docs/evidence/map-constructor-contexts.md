# Checked map constructor context

I apply declared key/value context before checking map constructors in returns,
arguments, globals, record fields, assignment, conditional values, and concrete
union payloads. My parsed record declarations retain complete field annotations;
my environment borrows those trees while the AST owns and frees them.

I retain only the checked int/string key and value kinds on constructor calls.
Those scalar facts survive AST copying without extra ownership. NanoVirt now
uses one constructor path everywhere and refuses missing checked context instead
of defaulting to string/int. Native C keeps the concrete constructor name.

My regression exercises all four int/string pairs, default shadows and actual
VM execution. It checks empty key extraction followed by append, insertion,
key/value extraction, clear, globals, returned constructors, direct arguments,
record fields, nested generic payloads, reassignment and both conditional arms.
Native C executes the same controls except the map local inside a selected arm:
its existing cleanup references that local outside its scope. I preserve the
full VM fixture and track native cleanup as `task_1edadd5eb33a445d9bf6516744bc405e`.

Missing context, unsupported bool key/value types, and wrong constructor arity
still reject while preserving an existing artifact. The old diagnostic test
that rejected a typed function return now belongs to the positive context gate.
Checking already-created maps against incompatible declared key/value types is
separate task `task_d0438e26b84147cdb9fd16b654c44a6a`; native AOT global transport
remains separately tracked. I do not claim either from constructor propagation.

Validation: six constructor/diagnostic/concrete-union methods pass; parser,
typechecker and all eight PGO unit methods pass; a fresh native bootstrap passes.
Forty ASan/UBSan parser/typechecker/environment/AST teardown iterations pass with
leak detection disabled under the existing metadata leak follow-up. These checks
do not establish NanoISA-only bootstrap or its bytecode fixed point.

Logs: `/tmp/nanolang-map-final-tests.log`, `/tmp/nanolang-map-final-unit.log`,
`/tmp/nanolang-map-final-bootstrap.log`, `/tmp/nanolang-map-final-asan.log`.
Task: `task_f4e1871af407805219770d7620d58349`.
