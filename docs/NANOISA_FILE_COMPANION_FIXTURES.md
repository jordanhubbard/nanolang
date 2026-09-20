# My companion resolution fixture checkpoint

I prepare these fixtures against reviewed source02b7318cb. No fixture, compiler,
bootstrap, service or new driver invocation has executed. My source remains
unqualified until independent fixture review and fresh acceptance below.

My `tests/test_file_companion.py` imports retained runner modules by module name;
I expose only my own TestCase. I reuse the durable file-backed command runner,
first-terminal recording, bounded process-group cleanup and empty LSAN_OPTIONS.
My nested invocation supervisors inherit that owning process group. Their retained
stdout/stderr files survive a supervisor or child failure; the outer terminal owns
timeout cleanup. I never retry an unchanged failed binary to manufacture a pass.

## My actual boundaries

- I build nine fresh snapshot/input/bridge/catalog/strict providers in linked and
  allocation-instrumented forms. I inject selected provider malloc/calloc/realloc/
  strdup failures and preserve actual frees. I model syscall failures before real
  acquisition, and modeled close reporting failure only after the real descriptor
  has closed. Input tests retain sentinels, partial/error cleanup, complete counted
  NUL/UTF-8 refusal, exact small extent,64MiB sparse extent refusal and64/65 EINTR.
  Snapshot controls retain request validation before all opens, second-request
  refusal, path components, count17 refusal, copied lifetime and getter sentinels.
- I retain every measured prefix and single transient allocation terminal. Successful
  snapshot recovery must retain exact generated source and full catalog bytes.
  Graph recovery must retain all row counts and a checksum over every original
  visibility tuple and every plan field. Ordinary parser/resolver allocation remains
  outside this injection domain. In the graph fixture only, canonical path storage
  allocated by inherited resolver/libc can be freed without appearing in my selected
  allocation table; actual free still executes. I do not claim whole-compiler OOM.
- I compare complete C and independent Nano reports: canonical origins and retained
  source bytes, visibility tuples, original target tuples, declaration IDs, all plan
  fields and all four snapshot payloads. Independent expected ordinary IDs/kinds and
  absence of locals supplement byte equality. Actual Nano collector refusal has an
  `ok` boundary rather than the C status enum; I retain that distinction explicitly.
- I use actual publisher output and compare its complete golden bytes. Graph cases
  include qualified/selective/wildcard/nested re-exports, distinct modules with the
  same catalog, repeated modules, collisions, private selection, ordinary declaration
  kinds, metadata refusal, NUL/UTF-8, configured limits and ordinary/alias boundaries.
  Large qualified service graphs can reach the alias cap before the request cap;
  I do not relabel those as isolated request-limit proof. The snapshot API separately
  exercises its request-count preflight. Existing strict/source-plan boundary suites
  remain required neighbors for their own unchanged document/text/work budgets.
- I invoke all three actual drivers with and without the source opt-in. Service and
  invalid graphs must preserve output sentinels and never call an observer compiler,
  including a compiler that would fail. Ordinary no-service programs must still
  compile, select their real shadows and execute37 with the option and by default.
  These are ordinary programs, not File service execution.
- I compile both complete report probes through C-seed/Stage1/Stage2 and compare exact
  selected shadow multisets from frozen transitive sources with C success JSON and
  self-host trace names. The fixture renames only the copied driver's entry function
  and its corresponding shadow; actual graph/parser/provider helpers remain intact.
- I exercise the actual native provider preparer through each producer: identical
  canonical requests, distinct same-basename sources, final ordered flag probes,
  required dependency/shared sources, compiler/language conflict and required compile
  failure. Known objects must disappear. Two real standalone Stage1/Stage2 driver
  invocations overlap with distinct flag profiles; each must run42 and own disjoint
  private provider paths, compiling/linking each canonical source exactly once.
- I separately retain the actual final native command with two runtime inventory
  paths resolving to one canonical file. Its observer deliberately refuses the
  final compiler operation after real provider compilation; I require one canonical
  input, cleanup and an unchanged output sentinel. This proves actual final-command
  identity, not runnable behavior of substituted runtime implementations. The
  successful concurrent standalone controls provide executable behavior separately.

## My frozen gate inventory

I first prepare a fresh complete ABI/compiler closure on Linux and puck, with
actual C-seed, Stage1 and Stage2 products and their source/tool/provider hashes.
I do not copy old compiler binaries across this changed native-driver source.
I run all paired methods and actual publisher/provider controls with each selected
compiler identity. My seven C configurations are Linux GCC ordinary, GCC sanitizer,
Clang ordinary and Clang sanitizer; puck Apple ordinary, Homebrew ordinary and
Homebrew sanitizer. Supported sanitizers are ASan/UBSan with leaks required where
supported; Apple ordinary is not mislabeled leak acceptance. Fresh selected TUs are
instrumented, while inherited ordinary compiler/runtime objects are explicitly
listed and hashed before/after. I never describe a full ordinary link as full
provider instrumentation. The owning graph and plan receive their own fresh hooks
for graph prefix tests; inherited parser allocations retain their separate scope.

I retain unchanged source-plan, strict binding/publisher, parser/token/schema,
module/wrapper, File opcode/public refusal and byte-grant neighbors with actual
selected closure. Existing seven-matrix evidence is historical, not acceptance of
this new graph/provider implementation. Input/output bytes, provider commands,
compiler selection, all first failures, per-phase before/after inventories and
final artifacts remain immutable. Puck capacity must be checked before staging;
source packaging may omit old evidence only with an explicit selected-input map.

Passing this checkpoint would qualify descriptive preparation and provider closure.
It cannot authorize File source shadows or service execution, complete nominal
propagation/lowering, or close full8bbc, cyclic/indirect/richer-borrow/mixed/source
and full5.1 acceptance obligations.
