# My first LLVM translator boundary

I translate a verified v2 module directly to LLVM IR. My initial contract covers integer/bool/void scalar storage, integer/bool results, locals, direct calls, branches and assertions. I retain value tags through calls and joins and check typed operations at runtime when verification cannot establish their inputs. My integer arithmetic wraps at 64 bits; division by zero returns zero, and INT64_MIN divided by -1 returns INT64_MIN. I use explicit control before division to avoid LLVM poison.

I reject imports, linked modules, heap values, nominal layouts, ownership/passive metadata, named module initializers, implicit exits and unsupported instructions. Non-entry functions must end with an explicit return or branch; I retain dynamic declared-result tag checks on every return. I do not embed NanoVM or resurrect AST target paths. I publish named output only after verification, profile checks and complete emission succeed, using an exclusive temporary beside the destination. Existing output and source survive failures.

My first same-module gate runs VM, C AOT and LLVM on calls, branch effects, loops, arithmetic boundaries and boolean tags. A passing scalar gate does not complete my required full-language LLVM or Wasm release scope. Wasm remains unimplemented.

## My bounded evidence

`make nvm2llvm nanoisa_dump` builds only the translator and module tools. `make test-nvm2llvm` requires LLVM's `llvm-as`, `lli`, `opt`, `llc` and a native C linker; missing tools fail this explicit execution gate. It does not run implicitly as an optional pass in the ordinary suite.

Eleven focused methods pass on Linux ARM64 with installed LLVM 23.0.0git. The same modules run in NanoVM, native C, unoptimized LLVM, optimized LLVM and linked LLVM machine code. I test wraparound, zero/overflow division and remainder, recursion, loops, stack joins, argument order, bool tags, runtime type rejection and atomic failure preservation. The negative bool-to-int call traps in VM/LLVM; my C translator rejects it earlier, which I record as refusal rather than claiming a native execution. At that foundation checkpoint a development PUSH_VOID fixture exceeded the C subset; the later scalar truthiness continuation below closes that prerequisite.

The final eleven methods pass in 6.098 seconds, and again in 5.292 seconds with the new host translator sources instrumented by ASan/UBSan/LSan (linked existing ISA objects are not fully instrumented). Logs remain `/tmp/nanolang-llvm-result-tag-tests.log` and `/tmp/nanolang-llvm-final-sanitizer-tests.log`. I retain initial assembly-fixture syntax failures and the new translator's corrected relative-branch offset failure in the earlier numbered logs. VM/C comparison tools come from the already tested host-ownership checkout; no full compiler build was run for this slice. Darwin and other LLVM versions are unrun evidence, not claimed coverage.

I do not yet lower generic arithmetic/comparison, floating point, globals, tail calls, strings, aggregate allocation, imports, linked modules, effects or resource/passive contracts here. Those remain explicit refusals or signature/profile exclusions. This foundation does not close the full LLVM release requirement, and I have not implemented Wasm.

My initial implicit-return fixture exposed a separate VM caller-resumption defect: a nested implicit return ended execution before the caller resumed. Task `task_4b3800f46af143fbb171f9565f92b8e0` retains `/tmp/nanolang-llvm-implicit-tests.log`. Entry implicit return executes in VM but C AOT refuses it; task `task_8a18a76c86884299ac3f7880ea617978` retains `/tmp/nanolang-llvm-profile-tests.log`. I refuse all such exits in this foundation rather than claim common coverage. My tests verify both refusals and a VM/LLVM runtime-unknown declared-result tag rejection. Named initializers are also explicitly refused because VM executes `__init__` before entry and this profile has no initializer contract.

My default executable entry is `main`. `--entry-name nano_NAME` selects an
ASCII identifier in my reserved target-entry namespace; other names are
refused before publication. My Wasm translator uses this API to avoid the
wasm32 C-main startup convention. The underlying scalar result remains i32
at the host entry boundary.

## My bounded float continuation

Task `task_4b6401a64a3b4accaae1d087c4c1aea2` adds typed F64 values to the existing tagged scalar ABI. I preserve exact constant bits and tags through locals, branches, calls and checked returns. I use ordered float comparisons except `F64_NE`, whose unordered predicate keeps NaN unequal. Division by either zero returns positive zero, and float truthiness treats both signed zeros as false and NaN as true.

For admitted int/bool/float/void scalars, CAST_FLOAT follows VM conversion; CAST_BOOL was refused at this F64 checkpoint; the scalar truthiness continuation below adds its separate common-profile contract. CAST_INT guards the ordered interval [-2^63,2^63) before fptosi, rejecting NaN, infinity and out-of-range values rather than producing LLVM poison. I do not enable fast-math flags, generic cross-type comparisons, heap values, imports or initializer/implicit-exit support here. The executable entry remains integer/bool; float helper results are supported.

Scalar CAST_BOOL continuation is `task_59c773b34bec49f4b46d5a0b4a8f2de7`; I do not treat branch truthiness support as opcode coverage.

My float gate passes 17 methods in 24.787 seconds, and again in 23.198 seconds with the new translator sources instrumented by ASan/UBSan/LSan (existing linked ISA objects remain normally compiled). Logs are `/tmp/nanolang-llvm-floats-zero-bits.log` and `/tmp/nanolang-llvm-floats-sanitizer-tests.log`. The unchanged modules execute through VM, C AOT, LLVM JIT, optimized LLVM and linked LLVM native code. Five NaN/infinity/range cases stop before invalid integer conversion. An additional LLVM harness checks returned signed-zero bits, including positive zero from a negative-zero divisor. This initial run did not establish Wasm coverage; the integrated gate below adds that evidence without claiming general heap/import coverage.

After the named-entry/Wasm foundation merge `2fdaeab7`, I pass all 28 LLVM and Wasm methods in 42.767 seconds (`/tmp/nanolang-llvm-wasm-floats-final.log`). The shared F64 fixtures run through Wasmtime and import-free Node as well as the existing VM/C/LLVM paths. They exercise typed arithmetic/comparisons, NaN, calls, locals, joins, zero truthiness and checked conversion; Wasmtime also rejects the five invalid float-to-int inputs. I retain the initial invalid string-table fixture syntax in `/tmp/nanolang-llvm-wasm-floats-integrated.log`; the corrected refusal uses an assembled string constant.

## My scalar truthiness contract

Under `task_59c773b34bec49f4b46d5a0b4a8f2de7`, I admit CAST_BOOL and generic AND/OR/NOT for void, int, bool and float. Void, integer zero, false and both float zeros are false; other admitted values, including NaN, are true. These instructions consume operands that earlier instructions have already evaluated; I do not introduce source-level short circuiting. Every result retains TAG_BOOL across storage, joins and calls. I keep U8 and heap truthiness outside this bounded shared profile and do not change typed BOOL operand checks.
