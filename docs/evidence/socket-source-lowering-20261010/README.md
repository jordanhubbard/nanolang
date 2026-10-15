# Paired TCP source lowering

I extend my C and independently implemented Nano source lowerers to my TCP
catalog. I retain nine nominal layouts, exact service import input arities,
aligned ownership flags and the 128-byte TCP nominal map. My C serializer
selects the matching checked indirect flow analysis for exact stack bounds;
my Nano serializer constructs its own sections and checksum.

I evaluate Endpoint fields in source order into scalar staging slots, then
load them in catalog field order for record construction. I include fields in
terminal-operand analysis, so a return inside a field drains pending operands.
I retain ordinary C output-preservation and allocation-refusal checks.

My tests compare C and Nano bytes for the same source path and selection.
They execute real IPv4/IPv6 programs through explicit grants and generated C,
including owned and borrowed indirect calls, reordered fields, invalid Endpoint
and a selected shadow. The wire comparison includes all unselected bodies.

I preserve initial failures: a constructor edit in the wrong dispatcher,
a stale archive after its object list changed, an import arity that counted
the catalog return descriptor, and a fixture parameter using a reserved token.
My archive recipes now depend on their object manifest and pass only object
members to the archiver.

This batch covers one nominal service declaration per source graph. I still
require mixed File/TCP wire and execution, CLI network policy and supervised
network shadows, DNS/WebSocket, fresh bootstrap and full platform/publication
qualification. I do not mark #990 or 5.1 complete from this evidence.

My expanded TCP source suite passes in 73.486 seconds with the C lowerer under
LLVM ASan/UBSan. I also compile and execute the Nano lowerer as both VM bytecode
and generated native C. For six source/selection cases, I compare both Nano
outputs to the C-produced wire, then execute granted VM and generated native
programs. Cases cover IPv4 direct calls, IPv6 indirect owner/borrow calls, two
selected address shadows, invalid port and a terminal Endpoint field. The C
fixture checks failed allocations and unchanged output storage for every case.
The retained archive includes exact source, wire and generated C; encoded ports
belong to that run, and the checked-in test starts fresh listeners.

My C File lowering corpus passes all 13 tests in 121.462 seconds. The independent
File wire tests pass before the first generalized-import arity failure. I retain
the original failed log alongside the corrected C corpus log.

I run these checks with LLVM on PATH and
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`:

```sh
python3 -m unittest -f -v tests.test_service_lowering
NANO_SERVICE_LOWERING_RETAIN=1 python3 -m unittest -f -v tests.test_service_lowering_nano
NANO_SERVICE_LOWERING_RETAIN=1 NANO_SERVICE_LOWERING_RUNNER="$PWD/obj/test_service_lowering_sanitize" python3 -m unittest -f -v tests.test_service_lowering_nano.ServiceLoweringNano.test_tcp_source_bytes_and_granted_execution
```

I build the runners with `make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang`
and targets `obj/test_service_lowering` and `obj/test_service_lowering_sanitize`.
These local runs compile the Nano implementation with the current C seed; they
do not establish a fresh Stage 1/Stage 2 bootstrap fixed point.

My final combined Nano lowering corpus passes all 14 tests in 214.432 seconds.
It includes the inherited File reference-map mutation checks after I repaired
the override that had returned no report. I retain that initial harness failure
in `nl51-source-full-parity.log` and the full passing run in
`full-parity-final.log`. Both public archives contain the Socket flow provider
and only object members (plus the archiver symbol table).
