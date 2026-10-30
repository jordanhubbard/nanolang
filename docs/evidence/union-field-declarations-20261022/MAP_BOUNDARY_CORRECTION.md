# Stale public map declaration test boundary

The old test expected a valid described STRING→INT HM_NEW/POP/int0 module to fail public verifier/native/v2 conversion. Current public map admission already accepts it. The exact unchanged f53539b9c ownership provider with identical current object closure reproduces the original line142 failure; `baseline-map-command.json` and baseline-map build/run stderr retain it.

The linked boundary now positively checks verification/native translation and v2 roundtrip verification/native translation. The unknown-kind99 negative remains, including selected converter output preservation under its explicit staging wrapper. There is no VM execution claim in this NanoISA-only harness.

The instrumented retry exposed cross-TU allocation ownership, not a product leak: retained-layouts plan allocations reached the declaration-only wrapped free through public map admission. `trace-run.stderr` records the actual stack. Full positive public proof runs in linked mode; instrumented mode retains exact declaration prefix/byte/work controls and malformed public refusals. Unknown allocations remain forbidden by the tracked free. Final combined gate passes; details accompany the declaration checkpoint receipt. This test correction is separate from union declaration production.
