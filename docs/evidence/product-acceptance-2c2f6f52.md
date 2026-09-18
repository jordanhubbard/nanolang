# My product acceptance at 2c2f6f52

At immutable product source `2c2f6f5258734b301ae35478f061d8bd59fc7035`, my fresh
bootstrap passes both stages, hello and installed independence. All 34 product
and conditional-type methods pass in 15.467 seconds after rebuilding NanoVirt.
The first expanded test attempt used the old NanoVirt binary from before the
C diagnostic repair; two negative cases failed. I retain both logs rather than
label that setup error a new source regression.

My full `make -j8 test-quick` builds parser, typechecker and retained transpiler.
It then passes 7 of 17 core examples. Ten examples stop at checked unsupported
boundaries: abs/map, tuples, union shapes, float arrays, declared filesystem
bindings, and string-edge expression types. The roadmap records six concrete
coverage tasks; I do not remove these accepted programs from the gate.

The isolated Darwin run at the same exact revision exits 2 in 228.29 seconds
at the bootstrap dependency fixture, before full acceptance. Report task
`task_e65fdbcb58664fca93f4fa9f8d7818b1` is complete as a failed report; correction
`task_fedcf8e93847494ab2e0887c089b960f` remains open. This is not Darwin product
acceptance.

My separate VM self-compilation gate passes. Initial bytecode and both complete
VM-produced compiler generations have the same 379844 bytes and SHA-256
`0c83512aeb663f557058ef7b0525da753ee2c9ab8ba169de767c0eeab114aa66`.
Both generations verify; the final generation compiles verified hello, which
prints the expected line. Declared host-library hashes match across generations
and remain unchanged, as do source, assembler helper and translator. The
[manifest and final integrity record](product-vm-fixedpoint-2c2f6f52.json) retain
commands, times, resource limits and hashes. I compare raw bytes without
normalization. Native self-compilation is running separately and is not claimed
by this VM result.

I preserve `/tmp/nanolang-product-conditional-bootstrap.log`,
`/tmp/nanolang-product-conditional-acceptance.log`,
`/tmp/nanolang-product-conditional-acceptance-rebuilt.log`,
`/tmp/nanolang-product-conditional-test-quick.log`, and
`/tmp/nanolang-product-vm-fixedpoint-2c2f6f52/` on the Linux host. The Darwin
peer retains its exact failed report log. PR522 and full publication remain held.
