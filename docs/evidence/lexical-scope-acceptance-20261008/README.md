# My lexical binding acceptance

I replace three legacy emitter source-text tests with three executable methods.
Both my C-seed NanoISA CLI and installed Stage2 compiler emit each positive
fixture. I verify and execute each module in NanoVM, translate it with nvm2c,
then compile strict C11 with ASan and UBSan and execute it natively.

I check inner names, different types and mutability, function aliases,
function-valued locals hiding named functions with a different return type,
global/local restoration, match-arm bindings, range loops, scalar array loops,
and record-array loops. Six invalid programs through two producers must exit 1
and preserve the prior module: escaped block/range/array names, writes to an
immutable inner or outer binding, and a local integer used as a function.

My initial run exposes a real C checker failure: a local integer does not hide
a same-named function in call lookup. The baseline publishes a module; the
self-hosted producer refuses it. I preserve a separate source reproducer,
command/tool hashes, output and status. My correction extends existing lexical
lookup ahead of ordinary declarations and builtin handling, retaining the
existing explicit binary64 intrinsic policy. A bound callee must be a function
and uses its own full signature. I remove the now-redundant later lookup.

All three corrected methods pass, including the stricter status-1 rejection
checks in final.log. This is four emitted positive products with both VM and
native execution, plus twelve refused products. I add this suite to both fresh
compiler-product routes in test_one_ir_compiler.py. That integration still
requires its full gate; standalone acceptance is not proof of those routes.

Commands: make -j2 nano_virt CC=/opt/homebrew/opt/llvm/bin/clang;
python3 -m unittest -v tests.test_genenv_scope. The broader make -j2
test-typechecker test-env-scoping test-callee-snapshots gate is running and
includes a fresh compiler bootstrap. Its source inputs remain unchanged;
this development tree has pending documentation and test edits during the run.
Canonical integration, Linux and final release qualification remain open.

My direct C unit gate passes the typechecker and environment suites plus all
ten Python lexical-boundary methods (19.380 seconds for the latter). Command:
`make -j2 -o stage1 test-typechecker test-env-scoping
CC=/opt/homebrew/opt/llvm/bin/clang`. I deliberately omit the stage1 prerequisite
for this separate C-object check while the original full bootstrap remains
live. I do not count this as bootstrap completion. The source hashes match
my committed correction and I retain the rebuilt typechecker object hash.
