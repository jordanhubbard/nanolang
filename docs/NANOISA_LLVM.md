# My LLVM translator boundary

I translate a verified v2 module directly to LLVM IR. My current bounded profile covers integer/bool/U8/float/void scalar storage, integer/bool/U8/float helper results, zero-result void helpers, locals, globals, first module initializer, direct calls, branches and assertions. I also admit module-owned literal strings through the separate closed-literal-string profile, including byte length/content comparison and exact string tags through calls and storage. Executable entries return one int/bool value. I retain value tags through calls and joins and check typed operations at runtime when verification cannot establish their inputs. My integer arithmetic wraps at 64 bits; division by zero returns zero, and INT64_MIN divided by -1 returns INT64_MIN. I use explicit control before division to avoid LLVM poison.

I reject imports, linked modules, computed strings and other heap values, nominal layouts, ownership/passive metadata and unsupported instructions. My bounded literal-string profile refuses ADD/CAST_INT/CAST_FLOAT throughout string-bearing modules until managed string operations are lowered; numeric-only modules retain their generic arithmetic. Explicit RET and verified code-end exits retain declared result-count and tag checks. I do not embed NanoVM or resurrect AST target paths. I publish named output only after verification, profile checks and complete emission succeed, using an exclusive temporary beside the destination. Existing output and source survive failures.

My first same-module gate runs VM, C AOT and LLVM on calls, branch effects, loops, arithmetic boundaries and boolean tags. A passing scalar gate does not complete my required full-language LLVM or Wasm release scope. My [Wasm translator](NANOISA_WASM.md) uses this shared lowering.

## My bounded evidence

`make nvm2llvm nanoisa_dump` builds the translator, its target runtime IR packages and module tools; Clang and LLVM opt are required at build time. `make test-nvm2llvm` requires LLVM's `llvm-as`, `lli`, `opt`, `llc` and a native C linker; missing tools fail this explicit execution gate. It does not run implicitly as an optional pass in the ordinary suite.

Eleven focused methods pass on Linux ARM64 with installed LLVM 23.0.0git. The same modules run in NanoVM, native C, unoptimized LLVM, optimized LLVM and linked LLVM machine code. I test wraparound, zero/overflow division and remainder, recursion, loops, stack joins, argument order, bool tags, runtime type rejection and atomic failure preservation. The negative bool-to-int call traps in VM/LLVM; my C translator rejects it earlier, which I record as refusal rather than claiming a native execution. At that foundation checkpoint a development PUSH_VOID fixture exceeded the C subset; the later scalar truthiness continuation below closes that prerequisite.

The final eleven methods pass in 6.098 seconds, and again in 5.292 seconds with the new host translator sources instrumented by ASan/UBSan/LSan (linked existing ISA objects are not fully instrumented). Logs remain `/tmp/nanolang-llvm-result-tag-tests.log` and `/tmp/nanolang-llvm-final-sanitizer-tests.log`. I retain initial assembly-fixture syntax failures and the new translator's corrected relative-branch offset failure in the earlier numbered logs. VM/C comparison tools come from the already tested host-ownership checkout; no full compiler build was run for this slice. Darwin and other LLVM versions are unrun evidence, not claimed coverage.

At this initial checkpoint I did not lower generic arithmetic, globals or strings. The later numeric/global contracts and literal-string continuation add those bounded cases. The later managed-string continuation adds computed concat, string ADD and checked substring. Tail calls, string conversions, aggregate allocation, imports, linked modules, effects and resource/passive contracts remain explicit refusals or signature/profile exclusions. This foundation does not close the full LLVM release requirement, and the shared scalar Wasm implementation likewise leaves its full-language scope open.

My initial implicit-return fixture exposed a separate VM caller-resumption defect: a nested implicit return ended execution before the caller resumed. Task `task_4b3800f46af143fbb171f9565f92b8e0` retains `/tmp/nanolang-llvm-implicit-tests.log`. At that foundation checkpoint, entry implicit return executed in VM but C AOT refused it; task `task_8a18a76c86884299ac3f7880ea617978` retains `/tmp/nanolang-llvm-profile-tests.log`. At that foundation checkpoint I refused those exits. The later return continuations below replace those refusals with common execution checks while preserving runtime declared-result tag rejection. At that checkpoint named initializers were refused. My later scalar-global contract implements the first `__init__` before every entry and persistent per-instance globals.

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

My shared truthiness gate passes 33 LLVM/Wasm methods in 109.463 seconds, including six new methods. The six focused methods pass again in 26.627 seconds with generated native C instrumented by ASan/UBSan/LSan, and in 15.575 seconds with strict Clang. I retain `/tmp/nanolang-scalar-truthiness-final.log`, `/tmp/nanolang-scalar-truthiness-sanitized-final.log` and `/tmp/nanolang-scalar-truthiness-clang-final.log`. The Clang run explicitly selects installed GCC 13 support files; its first run stopped on a GCC-installation selection warning, preserved in `/tmp/nanolang-scalar-truthiness-clang.log`. These are Linux ARM64 execution checks, not Darwin evidence. Linked existing LLVM machine code and translator binaries are not fully sanitizer-instrumented in this generated-C check.

The first shared fixture exposed native PUSH_VOID refusal, now repaired under `task_96e1b3c367da476e9fb4f4cf8df68932` with an explicit kind-zero value; my local-storage test distinguishes void from integer zero. The later void/int stack join remains explicitly refused by C under `task_ca11c365f9e742d090f09ab59c6de45c`. I retain that original failure in `/tmp/nanolang-scalar-truthiness-combined.log` and check refusal without replacing prior output. Common positive join coverage is bool-valued. I reject known heap operands before C publication and retain a runtime scalar-tag check for tagged values; I do not claim heap truthiness support. U8 and generic numeric/comparison contracts remain tasks `task_ad1c498801b34aa38d82f58a588f1d5a` and `task_01d3e7ca2b6a47b0bef9504f2fd04006`.

After integrating merged owned-runtime and path-normalization changes through `68bd6607`, the rebuilt VM/translators pass all 33 methods in 53.821 seconds (`/tmp/nanolang-scalar-truthiness-restack.log`). The targeted owned-runtime gate passes 32 allocation checks and 1051 runtime checks (`/tmp/nanolang-scalar-truthiness-owned-runtime.log`). Before the additive restack, the native structured-C gate passed 2414 checks and shape inference passed 1092 (`/tmp/nanolang-scalar-truthiness-native.log`). The two former unsupported-opcode assertions for CAST_BOOL/PUSH_VOID are replaced by executed common-profile positives and retained out-of-profile refusals.

The later VM/C repair now resumes ordinary implicit returns and admits native scalar/zero-result helper completion; [its contract and evidence](NANOISA.md#my-implicit-function-exits) retain the native integer executable-entry boundary. The shared continuation below adds the corresponding LLVM/Wasm admission under `task_4fd2bff257a44da0b0c4bb62b52b91b8`.

## My shared implicit-return admission contract

Task `task_4fd2bff257a44da0b0c4bb62b52b91b8` follows the merged VM/C ordinary return repair. I admit zero-result void helpers and one-result int/bool/float helpers; an executable entry still requires exactly one int/bool result and no arguments. Explicit RET and reaching code end share result-count/tag checks. A void call executes without placing any result on the NanoISA operand stack. I preserve caller operands and branch-to-end behavior, retain atomic output on refusal, and do not admit heap/multiple results, captures or initializers.

The completed continuation passes all 38 shared LLVM/Wasm methods in 73.715 seconds (`/tmp/nanolang-llvm-implicit-combined.log`). Twenty-one LLVM/float/implicit methods also pass in 42.344 seconds with the host translator sources instrumented by ASan/UBSan/LSan (`/tmp/nanolang-llvm-implicit-sanitizer.log`); linked existing ISA objects and generated LLVM machine code are not fully instrumented in that run. Five new methods cover empty and explicit void helpers, caller operand preservation, recursive void calls, nested scalar/bool/float fallthrough, conditional edges to code end, runtime tag failure and unsupported result-profile publication preservation. Positive modules execute unchanged in VM, C, LLVM interpreter, optimized/native LLVM, Wasmtime and import-free Node. The source base is merged VM/C repair `79968058`, with lowering checkpoint `9a8353ab`; no full compiler rebuild or frozen acceptance mutation was needed.

## My unsigned-byte continuation

I preserve TAG_U8 through scalar constants, locals, calls, results and joins.
CAST_INT/FLOAT produce the exact unsigned byte value; CAST_BOOL and eager
logical operations use zero/nonzero truthiness. Typed I64/F64/BOOL operations
retain their exact tag checks. My initial byte continuation compared values through explicit CAST_INT and
typed I64 comparison. The subsequent generic comparison continuation below
admits exact tagged comparisons as well. My [U8 evidence](evidence/scalar-u8-contract.md)
distinguishes these gates from the VM/C same-U8 generic comparison repair.

## My generic comparison continuation

I admit EQ/NE/LT/LE/GT/GE for my closed void/int/U8/bool/float profile. I use
separate equality and three-way-order helpers, preserving VM behavior rather
than deriving equality from an ordering result of zero. Generic NaN LE/GE
remain true while EQ is false; typed F64 LE/GE remain false. Mixed int/float
comparison retains binary64 integer rounding, including the 2^53 and int64
boundaries. Other mixed scalar tags order by tag number and compare unequal.

My [compatibility contract and evidence](evidence/generic-scalar-comparisons.md)
records same-module gates and exclusions. This is raw ISA compatibility, not
a new source-language numeric-promotion guarantee. Generic arithmetic, heap
and string profile admission remain separate.

## My current string boundary

I specify static byte descriptors, module lifetime and conservative operation refusals in [my literal-string contract](NANOISA_LLVM_LITERAL_STRINGS.md). Managed strings, aggregates and declared host linkage remain required work; this is not full-language acceptance.

My [managed-string continuation](NANOISA_LLVM_MANAGED_STRINGS.md) adds consuming concat/string ADD, frame cleanup and explicit status/disposal exports. Substring, conversions, other heap values and full-language coverage remain open.
