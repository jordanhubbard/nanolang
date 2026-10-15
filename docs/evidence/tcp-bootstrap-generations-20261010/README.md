# My TCP compiler bootstrap and generation checks

I qualify compiler sources at `fb1ee0745` under #990/#982. My fresh Darwin
bootstrap passes all seventeen recorded steps, raw Stage1/Stage2 equality,
native smoke tests and installed operation with the C seed removed. I use
LLVM clang, the ordinary O3 VM build and O2 native compiler products, with
the unchanged 1,800-second per-step and 30-second bootstrap shadow budgets.

I run:

```sh
env PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_CFLAGS=-O2 \
  make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  NANO_BUILD_CACHE=/private/tmp/nl51-tcp-fb1 bootstrap3
```

My receipt pins 1,090 source inputs, five tools and six immutable host libraries.
Both raw modules have 641,256 bytes and SHA256
`470a5016fb2fce87ae4d15e235d9730460e4547791c95329f5f6570e838186aa`.
I retain the receipt, bootstrap logs and the explicit input comparison before
changing my Make gate. The later Makefile change runs both generations; it
is not part of this bootstrap source hash. No compiler/runtime source changes
follow this receipt in this batch.

For each generation I select `bin/nanoc_stageN.nvm` and `bin/nanoc_stageN`
with `NANO_TCP_DRIVER_MODULE` and `NANO_TCP_DRIVER_NATIVE`, set
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang` and
`NANO_SERVICE_DRIVER_RETAIN=1`, and run
`python3 -m unittest -v tests.test_socket_service_drivers.SocketServiceDrivers`.
Each run includes the C seed and C bytecode driver as independent byte-parity
controls. Stage1 passes three methods in 49.565 seconds; Stage2 passes the same
three in 50.233 seconds. I retain all 154 CLI reports per generation, sources,
wire modules and generated C. These cover real IPv4/IPv6 exchange, dependency
and root-only shadows, denied grants, fuel limits, publication failure,
source/output aliases and native invocation authority.

I add the TCP corpus beside File in both generations of my Linux/Darwin CI
step and make the regular `test-nano-service-driver` target run both generations.
All six workflow configuration checks pass. I execute both generated shell
loops with recording stand-ins to check their selection and quoting; those
shell checks do not establish hosted test success. My real local suites are
recorded separately.

I retain the bootstrap host cache at `/private/tmp/nl51-tcp-fb1` because the
installed compiler modules reference its immutable libraries. I preserve the
user's untracked guide fixture with its original SHA256.

This is Darwin qualification of single-catalog source programs. Mixed File/TCP
execution, DNS/WebSocket, exact-candidate Linux and broader release gates remain
open. The earlier root-only supervisor SYSTEM incident remains unexplained;
these passing generations do not establish its cause. I have not released 5.1.

My complete File regression uses `NANO_FILE_DRIVER_MODULE` and
`NANO_FILE_DRIVER_NATIVE` with the same generation artifacts, compiler and
retention setting, running `tests.test_nano_service_driver.NanoServiceDriver`.
Stage1: Ran 16 tests in 462.131s; all 165 retained CLI commands match their expected status.
Stage2: Ran 16 tests in 460.076s; all 165 retained CLI commands match their expected status.
I retain both logs and command/source/wire evidence in this directory.
