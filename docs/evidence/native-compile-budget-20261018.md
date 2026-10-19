# Native compiler qualification budgets

I retain Linux compiler-contracts run `39195672200` as failed evidence. Two
OneIR host C compiler invocations exceeded their explicit 240-second limits;
my sanitized capture-scope C compilation exceeded 180 seconds. The job's
`CMD_TIMEOUT=1800` did not apply to those callsites. These are observed compile
timeouts, not guest execution failures or established infrastructure faults.

I use an explicit `build=True` selection at host C compiler callsites only.
My shared helper uses positive integer `CMD_TIMEOUT` (default 600 seconds).
I retain every guest, emitter and compiler-product execution deadline. On a
timeout or interruption I kill the process group and reap the driver. My
existing ServiceLowering helper remains unchanged.

## Focused evidence

- I passed three functional helper methods: default/override/invalid budgets,
  a build allowed beyond the unchanged guest deadline, and real descendant
  termination plus parent reaping.
- I passed two OneIR methods in 2.146 seconds: void locals through calls and
  tail restarts, and optional boolean global shape joins, including actual VM
  execution and host native compilation/execution.
- I passed my actual capture-scope fixture in 61.943 seconds: NanoVM and
  ASan/UBSan native execution produce the same checker output. This is macOS
  focused evidence, not a rerun of the failed Linux job.

I copied the following retained providers from the match-integrated aggregate
checkout. That checkout's product sources match `cbfa8c872`; my isolated harness
changes do not rebuild or establish a fresh bootstrap. These hashes identify
actual tools used, not an exact-source rebuild claim:

```text
d669f9f7d05117fb4f422d442c558917459a99d9cc9ce5fca34be05648705af2  bin/nano_virt
ff2cc26d336d38a55276a29cfe2da4f9f878f9d13c8312190b4aef88f5b8774b  bin/nano_vm
1bfbcf06ecc393326e2b945cf273beabce12dd0e7ff6eb2c0f419aced00e7198  bin/nvm2c
9e2c8ff1bab81812f1a741e92b3c6b33261210b43f52a7f53305c2b319d405bc  bin/nano_aot_runtime.o
1e33255585a49d77f96efcef0059820a7d41b550c27266cb8285d1f6e26bf770  bin/nanoisa
```

I retain logs at `/private/tmp/nano-native-budget-capture.log` and
`/private/tmp/nano-native-budget-oneir.log`. Full hosted Linux/Darwin
compiler-contracts qualification, including the two large OneIR compiler
products, remains required in the integrated branch. I do not close #976.
