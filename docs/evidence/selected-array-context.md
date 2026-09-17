# My selected array context

An enclosing typed array boundary can query a selected generic payload before
normal expression checking reaches its field. I now resolve that field using
the selected union declaration and concrete arguments, retaining the result
in the field's owned metadata cache. Later checking replaces the cache normally.

The original `array_push payload.values` fixture now passes. A record with the
same field layout but a different declaration still fails through both C-seed
and NanoVirt, preserving the previous artifact. Eight constructor, nominal
array and projection methods pass after a full `make bootstrap`; both generated
native compiler stages exist and run their bootstrap smoke tests. Forty repeated
parser/checker/teardown iterations pass ASan and UBSan with legacy leak detection
kept separate. Independent source review found no scoped blocker.

Logs: `/tmp/nanolang-selected-array-full-bootstrap.log`,
`/tmp/nanolang-selected-array-adjacent.log`, and
`/tmp/nanolang-selected-array-asan.log`.

I track this repair as `task_17ace696e309484bbff5a59acc2891db`. It restores the
adjacent positive gate for generic selected ownership; it does not establish
bytecode bootstrap equality or complete ownership lowering.
