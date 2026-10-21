# Consuming resource loop exits with FLOAT locals

This is the next finite reconstruction slice for #999, release scope ownership and structured-control rows. It preserves the unchanged `test_float_local_unsafe_control_flow` source in `tests/test_source_borrow_emission.py`, not a reduced example.

## Observed baseline

All three current producers verify and execute the original ordinary module (three functions, 2/3/8 locals). The independent retained C-hosted original-shadow driver produces a verified executable four-function module (2/3/8/2 locals). Facts succeed; both reconstruction targets refuse a non-crossing owned diamond and preserve prior output. Ten provider/source hashes remain unchanged. The evidence directory retains the initial harness-only missing ownership_version-key error and corrected commands. Consuming facts intentionally omit that borrowed metadata field.

## Proposed representation boundary

Use a separate consuming-exits flag, never the borrowed single-exclusive witness flag. Restrict this initial route to consuming modules with scalar INT results, INT/BOOL or owned parameters, scalar INT/BOOL resource fields, and existing nonparameter FLOAT locals. Keep existing function/local/instruction limits, exact nominal authority and direct acyclic consuming-call checks. No FLOAT field/signature, new facts format, runtime instruction or synthesized consistency witness.

Reuse the structured region parser for innermost break/continue and exact zero-trip joins. Preserve explicit owned-parameter IR at function entry. Initial scalar-only pretests retain the existing whitelist; FLOAT body instructions reuse the existing binary64 IR and finite Nano literal renderer. No arithmetic or call pretest expansion.

Before rendering, independently prove every original function's owned origins. Parameters introduce unique exact nominal origins; pack/move/store/unpack retain current transfer rules. A direct consuming call must match its actual callee signature, consume each owned operand exactly once, and return INT only. No escaping owned call result or nested call expression. Every successful return consumes all owners. Every break/backedge must retain exactly the incoming origin map, including owned parameters; replacing an owner with a same-layout pack must refuse. Branch joins retain exact origins; zero-trip paths cannot establish initialization. Calls and FLOAT expressions are not reordered or duplicated.

Renderer support emits only existing break/continue syntax under this distinct flag. Borrowed helper work/witness proofs and their planner/scratch enrollment remain unchanged. First measure unchanged Nano output against 128 locals before proposing any planner or scratch enrollment; stop on refusal. C and Nano retain all assertions, typed NaN predicates, copies, signed values and original unsafe lexical slot identities. Original selected-shadow input remains independent; no fabricated consuming witness.

## Required qualification

Unchanged ordinary inputs from all three producers and independent original-shadow input; original and regenerated verifier/VM plus sanitized C O0/O2 execution. Exercise zero-trip, entered, break and continue paths, incoming owner identity, per-edge local initialization, negative branch-hidden owner replacement/consumption and side-entry/outer transfer controls. Preserve later assertion failure with explicit causality controls and NDEBUG native checks. Check exact 128 acceptance/estimated129 atomic refusal if reachable without changing the accepted grammar. Prior consuming FLOAT, STRING, borrowed-exits and current reconstruction outputs remain byte-identical where the new discriminator is absent. Fresh aggregate and hosted platform evidence remain separate requirements.
