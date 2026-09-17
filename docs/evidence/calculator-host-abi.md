# My scalar calculator host contracts

I add only the existing empty builtin namespace contracts `strlen(string) -> int` and `atan(float) -> float`. My canonical emitter validates their declaration signatures. Native translation recognizes their complete namespace/signature identity and transports float host arguments/results using float storage, with checks when inputs retain runtime tags. A same-named user function keeps its body; another library does not acquire the builtin adapter. `strlen` reports UTF-8 byte length, matching its C contract.

Three focused methods pass through both C-seed and self-hosted bytecode producers, verified VM execution, and generated native execution under ASan/UBSan. I ran them with GCC and Clang, including empty/UTF-8 strings, direct/called/tagged float values and both signs. The local Clang initially refused its ambiguous GCC installation choice under strict warnings; I selected the installed GCC 13 toolchain explicitly, without suppressing the warning. All three methods then passed.

My full native gate passes 2,412 translator checks and 1,092 shape checks. Seven raw-emitter driver methods pass. Two old negative fixtures assumed scalar float and Boolean-array results were unsupported; task `task_64f967c7f1d14b26a2b8134fb94a706e` replaces them with the still-unsupported float-array boundary and adds positive float/Boolean-array verification. Previous output and exact diagnostics remain checked.

An initial extra float-record fixture exposed existing aggregate limitations. Task `task_93574cf9d200459aa16e959baf68201d` retains those refusals and the original fixture/log; this scalar ABI change does not lift them. Full unchanged-calculator acceptance awaits integration with the separate par frontend work, and broader foreign-call contracts remain open.

Task `task_9493ea33bbb54404acc47c256644a05a` owns this bounded ABI change. Logs use `/tmp/nanolang-calculator-abi-`: `scalar-final.log`, `clang-final.log`, `native-gate.log`, `emitter-final.log`, plus initial `tests.log` and `emitter-gate.log`. The original aggregate fixture is `/tmp/nanolang-calculator-abi-initial-tests.py`.
