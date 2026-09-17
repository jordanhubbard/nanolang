# My bounded VM bootstrap observation

At source commit `223bf6091538c4ab993850e7b75a24662eca176d`, I ran:

```sh
NANO_AS_CAPTURE_HELPER="$PWD/bin/nano_as_capture.so" \
  timeout 1800 bin/nano_vm /tmp/nanolang-selfhost-stage1.nvm -- \
  src_nano/nanoc_v06.nano --emit-nvm \
  -o /tmp/nanolang-selfhost-stage2.nvm
```

The process reached the 1,800-second diagnostic limit (exit 124), emitted no
compiler diagnostic and published no output. It consumed roughly one CPU
core. Resident memory grew from about 104 MB to 112 MB in the sampled period.

Read-only debugger samples showed forward progress: `nisa_is_global_let`
advanced from candidate 3,819 through 4,441 and 5,564 of 7,076 declarations.
Later samples reached `compile_program -> transpile_parser_mode ->
is_let_in_function`, after NanoISA lowering and during my existing native
shadow C-generation path. Several samples were inside cycle-collector
`mark_gray` or `scan_black`, entered by releasing temporary aggregate values.

These observations identify work to measure. They do not establish a
correctness defect, a deadlock, or a fixed point. I must preserve declaration
ownership, initialization effects and shadow execution while reducing
repeated scans or collector cost. The independent full VM compilation gate
remains open even if native execution of the same bytecode converges.

The executable seed predated the pinned source revision; this was an
execution probe, not a valid equality comparison between matching stages.
A separate clean pinned bootstrap records its source, stage labels and
immutable host-library hashes before comparing successive outputs.

Evidence remains in `/tmp/nanolang-selfhost-stage2-timeout-evidence.json`,
`/tmp/nanolang-selfhost-stage2-build.log`, and
`/tmp/nanolang-selfhost-stage2-progress-frame*.log`.
I track this work as `task_36ceaa830d7d46ba8a5471326f525aac`.
