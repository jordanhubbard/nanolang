# I retain service types through parsed annotations

I implement this batch under #989 and #982, based on `c1d97baa6209a74777bbda06c1a960ad7566ac5b`.

I keep declaration/module/ordinal/category identity in C TypeInfo and independent Nano facts. My catalog queries return exact scalar fields, Result payloads (including void), method signatures and exclusive/consume modes. Aliases retain their original target. Type copies, callback signature comparison and generated metadata retain those identities.

My actual C loader binds the parsed annotation graph after collecting the namespace. My Nano driver binds annotations before merged function names change. Parameters, returns, locals, array elements, callback fields and generic union payloads have focused controls. A generic formal named Handle remains a formal. I retain re-exported parameter paths and callback field signatures that my C parser previously truncated or discarded.

## My checks

- `NANO_SERVICE_NAMESPACE_DRIVER_MODULE=/private/tmp/nanolang-service-types-driver.nvm python3 -m unittest tests.test_service_namespace`: two methods pass in 115.035 seconds. This covers 17 graph cases through C seed, NanoVirt and the component Nano driver, plus independent Nano identity checks produced by both compilers and executed in VM and sanitized native code.
- `make test-checker-metadata-ownership test-union-metadata-ownership`: pass. I also compile the checker ownership harness with LLVM ASan/UBSan and execute it successfully with ordinary linked objects; this is not a fully instrumented compiler build.
- I compile and run the full `tests/test_module_metadata.c` harness. All tests pass, including a generated C metadata round trip retaining service identity, nested signatures and the existing intentional graph cycle.
- `make test-parser test-typechecker test-parameter-nominal-metadata`: pass, including all three parameter metadata methods.
- `make CC=/opt/homebrew/opt/llvm/bin/clang test-service-namespace-sanitize`: pass with ASan/UBSan and leak checking for the namespace harness and included namespace implementation. Other common objects remain ordinary objects.
- After retaining matching union payload tags, I repeat the graph method with the component Nano driver and sanitized C harness. All 18 cases pass; `final-graph.log` retains the terminal.

I retain the original failures. LLVM symbolization maps the metadata ASan read to `checker_borrowed_control` at its post-environment-teardown array read, its free to `env_reclaim_static_arrays`, and its allocation to `create_array`. Commit `603c4c44e` established environment-owned static array reclamation; the old test still assumed caller ownership. I now assert retained data before teardown, add an alias and verify borrowed callback metadata after teardown. No production ownership rule is weakened.

## My remaining work

This is annotation/type transport, not executable File support. Both compiler routes still refuse service execution before output publication. Actual body checking, full tuple/open-record type retention, affine flow, independent lowering, grants and supervised publication remain required. I keep the full 5.1 scope open.

The component driver is a development artifact, not a fresh Stage1/Stage2 qualification. An accidental `make stage1` dependency started a bootstrap during the build; I deliberately interrupted that run after recognizing it, rather than use an intermediate annotation batch as a release gate. I claim no bootstrap result from that interrupted run.

My user's untracked guide file remains unchanged (SHA256 `c739aeb158c5b3e94c15d8de1232e4e1415a3f20b2e80fcf96fba39f1dedb976`).

## My CI wrapper integration correction

CI run 37978631193 at c1d97baa6 advances past the repaired evaluator and VM FFI gates, then fails two of five wrapper generation tests. The wrapper's manually retained link list omits service namespace and immutable compiler-input objects. I reproduce the same undefined-symbol failure locally at fd46d3acb, add the compiler's complete immutable-input object closure, and retain the corrected `make test-wrapper-gen` result: all five C tests and nine Python publication tests pass. The positive C test links and executes a real wrapper, checking exit status 7. Linux sanitizer/platform qualification remains pending; the local correction does not establish a green full CI run.
