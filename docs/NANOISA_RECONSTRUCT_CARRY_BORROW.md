# My carry and borrow reconstruction contract

I record `task_cbdce24cc6b747f6b39ceb4c87f20676` before implementation under parent `task_4bd034f6029b7458201db74e2c3aeb32`. I admit `I64_ADD_CARRY` and `I64_SUB_BORROW` with exactly three `INT` operands. I normalize the input carry or borrow to its low bit, including negative carriers, and push the wrapped low result followed by an `INT` high result of zero or one, matching my VM.

I represent the two results as exact scalar expressions with distinct per-instruction low/high temporary names. Existing operand snapshots retain evaluation order and once-only direct calls. Pure loop-condition expressions remain side-effect-free. I do not introduce aggregate storage, tuple return types or a multiresult function ABI.

My C helpers use `uint64_t` arithmetic and representable signed bit conversion. My NanoLang helpers use my tested total add/subtract and unsigned comparison helpers, with explicit zero/one high results. I retain previous output on exact-tag, arity and unsupported-profile refusal. I require both result orderings and consumption, endpoint and noncanonical carry inputs, calls, snapshots, loops, canonical byte roundtrip and paired VM/reconstructed C/current compiler execution with GCC and Clang sanitizers. Source and tool hashes remain separate; full reconstruction remains open.

## My retained historical boundary

The original draft at `bcaa677a` first exceeded the existing 4,096-instruction admission cap and expected the assembler to accept a `BOOL` carry. Its bounded endpoint fixtures later reached a non-C-seed compiler exit `-11` without establishing which self-hosted stage failed. I retain that evidence under `task_1009525724234d9ca7df3f8284e75943`. I do not replay, minimize or diagnose the historical crashing binary.

The later verifier work made all three operands exact `INT` values, so wrong-tag and arity modules now refuse during assembly while preserving prior output. The iterative block-parser repair has since merged. I therefore port only the reviewed reconstruction semantics to current main and qualify them with freshly built current tools and newly generated inputs. Passing current-main acceptance may close the historical acceptance gap; it does not retroactively identify the old signal or its stage.

## My current-main port

The stale PR branch predates later float, generic arithmetic, comparison, stack and wide-multiply reconstruction. I add carry/borrow to those current tables without replacing any later route. My focused gate must preserve every current scalar reconstruction method, exact helper dependency closure and existing output-preserving refusals. I record exact compiler argv and fresh tool identities for any terminal result.

My first complete current suite at production `008fdb8f` passes 53 of 55 methods in 603.601s. Both failures are obsolete expectations that generic `ADD` refuses, although `3c699f84` deliberately admits it. I record `task_a74f7ac27d25ae6c8eebacc12179c9cb` before correcting tests. I replace the foundation control with already unsupported `CAST_FLOAT` and remove the duplicate `ADD` entry from the arithmetic refusal list, which already contains `CAST_FLOAT` and exact wrong-tag cases. I preserve prior-output assertions and require a fresh complete suite; I make no production change for this correction.
