# My Linux interpreter CPU observation

I measure the unchanged C seed at
`6e09c07a9f1651294e4669282cdc17ee14126775` in a separate Sparky worktree,
under #1265/#1253. This is diagnostic evidence. I change no C product,
fixture, assertion or original 300/default10 deadline. My fresh original C
build passes in 58.159757 seconds with its normal flags. Three earlier setup
refusals retain the initially absent checkout, schema and ISA specification;
none reaches compiler execution. I then materialize all non-documentation
source inputs before the successful build.

My first sampler cannot record software cpu-clock events on this host, though
its counting probe worked. The compiler trace reaches all 1,139 completed
shadows with zero recorded failures and a successful original supervisor result.
My controller's final digest read refuses the root-owned perf file; its initial
result remains nonterminal. A separate terminal audit verifies absent compiler
and sampler PIDs and unchanged sources/tools. The compiler process exit code
was not retained. I do not turn that missing boundary into a pass.

A finite hardware cycles:u control captures 23 expanded ARM PMU samples. I
retain its original read-permission/decode refusals and the read-only privileged
byte copy used for successful decoding. My follow-on controller corrects only
the sampling event and output read, with a 15-second sampler bound and five-second
kill grace around the original ten-second compiler supervisor.

My hardware run captures 828 samples, all from the verified owned shadow child,
using ARM PMU user-cycle event `armv8_pmuv3_1/cycles/u`, requested frequency99Hz
and 8192-byte DWARF stack capture. Perf reports zero lost samples. Mapped binary,
DSO and perf tool bytes are retained. These are sampling observations; inline
frames and unknown/kernel frames remain in the raw reports. I do not claim
exclusive CPU duration from event weights or universal stack-unwind accuracy.

The self report attributes 13.71% to `env_define_evaluated_var`, 10.12% to
`symbol_lookup`, 10.99% to strcmp and 14.56% to gc_alloc. My separate grouping
of the first non-inline frame reproduces the first two percentages; grouping
inline allocation frames changes that allocation total. The binding annotation
places most of its samples in the existing forward STRING-alias scan over
`env->symbols`. This identifies measured work to investigate. It does not yet
establish a replacement's performance, correctness or Darwin benefit.

All 1,139 shadows complete with zero recorded assertion failures. The child
writes completion and flushes, then the supervisor records a clean child exit
but marks its original ten-second deadline exceeded:10.122881403s from parent
start and625.706947ms from flush to the supervisor's final observation. Compiler
exit is1 after35.521112s; sampler exit is0. This sampled failure qualifies no
product. The observer, scheduling and host differ from the hosted Darwin run;
I do not infer that earlier failure's exclusive cause or teardown time.

All 4,676 source hashes and eight tool/input hashes match before/after and
between the software and hardware observations. The source bytes resolve to
Git6e09; no accepted native root is mutated. The original executable's hash,
PID/parent/start ticks and actual dispatch record bind the sampled process.

I retain57 records/59 chunks/67,174,846 uncompressed bytes. Manifest SHA-256:
`b4c28416c3d93b24b19e689183a8b45fe36568d802273ee0337ea8b9b6433165`.
My reader verifies all chunk/file digests, 4,676 Git source blobs, unchanged
source/tool maps, both trace sequences and the separate terminal audit. It
checks every retained decoded sample's PID/event, all 19,573,565,123 weighted
periods and the two independently grouped hotspot percentages. It checks the
23-sample control, binary/tool identity and annotated alias loop. It does not
freshly decode perf data or execute a compiler. A changed manifest refuses
before reconstruction.

```sh
python3 docs/evidence/c-repairs-linux-20261126/interpreter-cpu-profile-20261127/read_interpreter_profile.py \
  --archive docs/evidence/c-repairs-linux-20261126/interpreter-cpu-profile-20261127/archive \
  --repo . --workspace /path/on/large/volume
```

Next I measure alias-scan hit positions and absent aliases before choosing a
repair. I must preserve exact retain/release and binding semantics, including
fresh strings, older/global aliases, multiple aliases, assignment and scope
cleanup. I require permanent regressions and original qualification afterward.
The [hosted body-timeout](../hosted-shadow-body-timeout-45998897143/README.md),
earlier hosted flush timeout, frozen assertions, full pure translator,
paired/cutover/exact-candidate CI and all 69 release requirements remain open.
