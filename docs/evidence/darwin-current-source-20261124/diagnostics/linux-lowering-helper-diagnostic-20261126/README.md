# My Linux lowering helper and artifact-boundary observations

I continue #1265/#976 on sparky at source
`d907f264d16d348b4589d3760bd213803e1c6f35`, after publication
`f4588455d2443bddc7eb65aaa6ed414a4e146735`. I change no production source.

## What I measured

My conditional helper markers place the two slow emission intervals inside the
call to `nisa_emit_block`:

| Function | Body-call interval, seconds | Largest other adjacent helper interval |
| --- | ---: | ---: |
| `publish_service_source` | 25.373321 | 0.035524 |
| `assemble_text_save` | 6.365149 | 0.026964 |

These intervals include hook overhead. I preserve the previous12 module-loop
insertions and add10 reversible helper insertions plus one direct shadow.
My corrected combined codegen SHA is
`af34af9cfbc2b3100c4757d8111b4ecd3832dc7ececcf5d58d61e6501969a2fd`.
The seed/verification pass87.462931/0.264458s, and program lowering completes
112.789870s within its120s window. I stop the VM at the return marker, exit-15
after211.190210s from launch. Complete compiler acceptance remains false.

I then compile a small bytecode witness against the unchanged
`CompilerSupport.module_artifact` provider. It calls each provider lookup twice,
checks absolute results and repeat equality, and checks an empty-input refusal.
I capture its stdout through a raw PTY with normal libc line buffering:

| Lookup | First, seconds | Repeat, seconds |
| --- | ---: | ---: |
| `file_product` | 25.427272 | 25.503232 |
| `nanoisa` | 6.839779 | 6.692734 |
| Empty input | 0.003574 | — |

Both repeated lookups return the same paths and bytes as my pinned prefix host
libraries. This reproduces the delay at the declared artifact-lookup boundary;
it does not isolate every operation inside that boundary or establish the cause
of my original Darwin1800s timeout.

My raw PTY controller records `diagnostic_valid=false` because its conservative
process scan includes the Toolscheme completion watcher. I preserve that result.
My separate post-terminal audit verifies the exact watcher has exited, the driver
is absent, all guards still match and no owned process remains; it releases only
the reservation with the matching root/PID/start identity and records
`timing_observation_valid=true`. I do not rewrite the raw controller result.

A subsequent run uses my existing `NANO_TRACE_BUILD=1` hook, with no preload or
ptrace. It passes and records three accepted cache-reuse checks and one context
miss/rebuild. Its first/repeat lookup intervals are33.778557/33.035430s for
`file_product` and21.879373/9.526916s for `nanoisa`. It returns different retained
generations from the initial prefix after the intervening failed trace attempt;
I archive their actual bytes and do not equate the two runs. Ten recorded tool
queries total163ms, maximum37ms. Timer groups overlap and hash observations are
not durations. Further profiling of retained input capture/replay is required
before selecting a repair.

## Attempts I retain

- My first full-name helper guards miss because their string literals influence
  `merged_binding_prefix`, which checks every token value. The generated prefix
  gains an underscore. I stop only my verified owned VM, preserve its149.320868s
  exit-15 result and all guards, and correct the next distinct run to suffix
  guards. Its helper observation remains invalid. I also retain the earlier
  ambiguous preparation-anchor refusal, which launched no compiler.
- My first direct witness uses buffered stdout. I stop and preserve it without
  assigning pipe-arrival times to calls.
- `stdbuf` introduces a preload that my retained-input capture guard explicitly
  refuses. I preserve its exit1 and original assertions; I change the observer
  to a PTY and leave the provider implementation unchanged.
- My separate subprocess trace times out at its original180s limit, exit-15
  after180.002692s. Its partial first lookup returns a different retained
  generation. I preserve the trace and returned bytes; I use it for no baseline
  timing, complete process profile, compiler qualification or infrastructure claim.

## My evidence boundary

My complete manifest binds112 records, SHA
`b562894bd99ae58f41e3aaeb504518a45dd7d1c7d26548c262e8600b787277df`.
My failed full-name manifest binds39 records, SHA
`7fa6d133f8cc867488f10ebf03a8a0cec73a033c375c7a6b9dcaf565361bdd5e`.
The readers independently verify both overlays' exact reversal,4683 pinned Git
blobs per case, the previous33-record Linux observation and all7088 records of
my accepted Linux bootstrap prerequisite. All original-source, copied-image,
selected-tool, host, seed, accepted-root and terminal cleanup guards pass.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/linux-lowering-helper-diagnostic-20261126/read_terminal.py --repo .
```

I retain seed600/verify60, lowering-entry600/window120, compiler-shadow30 and
ordinary10 budgets; the standalone witness uses bounded180/60/180 compilation,
dump and execution limits. I change no original qualification1800 deadline,
assertion, lookup/owner/body/alias/print-flush contract, C source, C ownership or
compiler/bootstrap/CI selection. No Puck build or unattended loop runs here.
A reviewed repair, focused negatives, complete native/bootstrap/current-candidate
and paired-platform gates, #1262, installed pure-translator cutover and all69
release requirements remain open.
