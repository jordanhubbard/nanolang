# My NanoISA file facade artifact contracts

I retain the declared artifact owner and exact signatures for
`nl_nanoisa_assemble_save(string, string) -> int`,
`nl_nanoisa_load_print(string) -> string`, and
`nl_nanoisa_load_pretty(string) -> string`. My existing VM artifact loader and
native borrowed-string snapshot adapters implement these calls; this change
adds their frontend lowering contracts without changing that runtime ABI.

My real module test assembles a file, verifies the product, calls both
representations, checks a missing-file diagnostic, and retains both strings
across subsequent facade calls. It passes in VM and native execution. Six
mismatched declarations are rejected before assembly publication.

At source checkpoint `b30c8a4a`, my complete emitter gate passes 86 exact
bytecode comparisons and 74 Python methods in 94.746 seconds. A fresh native
bootstrap passes with the explicit supported 60-second shadow budget; the
Stage 2 compiler then builds my artifact harness and all ten methods pass
in 17.684 seconds without an override. My first default-budget bootstrap
stopped at the 10-second shadow deadline, without an assertion diagnostic.
I preserve that result under `task_628759a2daf743b9bf13c9a7fea2ced0`; I have
not established its cause or changed the default deadline.

Logs: `/tmp/nanolang-facade-emitter-gate.log`,
`/tmp/nanolang-facade-bootstrap.log`, `/tmp/nanolang-facade-bootstrap60.log`,
and `/tmp/nanolang-facade-stage2.log`.

This closes bounded prerequisite `task_f5f873fccfff4b5b88f14f4d825ba3b4`.
The canonical VM-shadow cutover and full NanoISA-only release gates remain
separate acceptance work.
