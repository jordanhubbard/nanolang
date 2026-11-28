# My bounded Linux lowering observation

I continue #1265/#976 on sparky under the updated fleet policy. I use an
isolated source snapshot at `d907f264d16d348b4589d3760bd213803e1c6f35`, after
publication `5988c579e2cb8a76a272ee6235bd9bb83bf63230`. My production source
is unchanged. I copy the qualified Linux tools from my published04cee bootstrap;
this observation is not a fresh cold build or a release qualification.

My prefix seed and verification pass94.693285/0.214482s, taking96.899009s in
total. My VM reaches program lowering98.288415s after launch. Within lowering:

| Marker or interval | Seconds |
| --- | ---: |
| Globals complete | 0.693590 |
| Queue32 begins | 4.191799 |
| Queue41 `publish_service_source` begins | 5.308275 |
| Queue42 begins | 31.207108 |
| Queue49 `assemble_text_save` begins | 31.766883 |
| Queue50 begins | 38.304113 |
| Queue64 begins | 38.894577 |
| Function loop complete | 111.479125 |
| Prelude complete | 112.024460 |
| Program lowering complete | 112.060742 |

Entry41→42 takes25.898833s and49→50 takes6.537230s. The complete32→64
interval takes34.702778s; every other adjacent measured interval within32→64
is under0.477s. These are intervals between observation hooks, including loop,
next-entry retrieval and recording overhead. They do not isolate one helper.
Entry0→32 covers32 entries and has no per-entry attribution.

I stop the VM after its program-lowering return marker, exit-15 at210.382037s
from launch. I do not run the complete compiler/shadow/publication path. This
Linux result does not establish the cause or repair of my original Darwin
1800s failure, and it does not replace required Darwin acceptance.

My33-record manifest is
`3b8ee2a19660c6ca457fef7e2d9eee3ec5bbb369784728ab0ed3cae86317c8ef`.
The reader verifies all33 records,4683 pinned Git blobs,12 reversible NanoLang
observation insertions and one added shadow. It independently re-verifies all7088
records of my accepted Linux prerequisite archive and binds the copied images
and tools to that receipt. Nine copied images including the module-index helper,
22 selected tool/library bytes, six new host libraries, new seed, working sources
and original accepted-root sources remain unchanged. Cleanup leaves no owned
process and releases my reservation. This is a valid diagnostic; compiler and
whole-release acceptance remain false.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/linux-lowering-detail-diagnostic-20261126/read_terminal.py --repo .
```

I preserve seed600/verify60, lowering-entry600/window120, compiler-shadow30 and
ordinary10 diagnostic budgets. I change no original assertion, qualification1800
limit, lookup/empty-print contract, C source, C ownership or compiler selection.
Under #1265 I still need focused helper measurements before selecting a repair,
then original native/bootstrap/candidate acceptance. All69 release requirements,
#1262's repair/handoff boundary and installed pure-translator cutover remain.
