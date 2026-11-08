# My controlled C-seed comparison

Under #1065 I compare the baseline at
`9baeba459e6ea4c7ebd6c0694cb05fef4d36f401` with the current review head.
The baseline already includes the shutdown registry and STRING terminator
correctness fixes. The candidate adds aligned private pointer-entry allocation.
I keep those correctness fixes separate when judging that performance candidate.

My `cseed-source-comparison.yml` diagnostic workflow builds both actual provider
closures serially on one worker with the same selected compiler and SDK. I put
both detached checkouts under `~/.worktrees/nanolang/`, with equal-length names,
check worker headroom before provisioning/builds/samples, and retain the source,
provider and host-compiler hashes. I refuse different complete Nano source inputs
or production factory helpers before measuring.

I write this fixed plan before building: original, modified, modified, original,
original, modified. Every attempt invokes the unchanged production source-creation
factory in a fresh Python process, once. Its first-creation cache and failures
remain intact. I preserve the original default shadow selection, ten-second
supervisor and native build budget from `tests/native_process.py`. My diagnostic
job limit is not a replacement for any original release budget.

I retain every successful or failed attempt, native source compiler when
published, exact factory commands/inputs/output, before/after provider hashes and
ordered shadow traces. Failed traces must match a prefix of the same selection;
a shorter successful selection cannot replace the complete one. I copy actual
C-seed bytes and verify all copied evidence. The diagnostic still fails if any
planned creation fails. It does not retry until a sample passes or replace the
seventeen-method source/native gate, fresh stages, integrated platform checks,
NativeSDK custody or release acceptance.

Local validation covers changed/missing input refusal, interleaved/truncated
trace parsing and successful-subset refusal. I parse the workflow and check the
script without launching full provider builds on my disk-constrained local worker.
Hosted results remain required; I do not infer them from these driver tests.

GitHub initially refuses both file-name dispatch requests with HTTP 404 because
this workflow is absent from the default branch. I add a push trigger limited
to my review branch and its two comparison implementation files. That trigger
starts Linux/Apple diagnostic jobs; it does not change any release gate. Manual
input declarations remain in the workflow; the first successful run uses push.

## My first Apple setup refusal

Run 42533514074's Apple job stops before building a provider or running any
source sample. Its preflight has 28.75 GiB free; after two ordinary full
worktrees, 14.42 GiB remains. The bounded inspection measures 14,900,644 KiB in
the worktree directory. I retain the CRC-checked artifact, worker/plan snapshots
and inspection in `first-apple-headroom/`. This is a verified setup refusal,
not a compiler timing result. The Linux job remains on its original handle.

I keep the headroom policy and complete tracked trees. On macOS I prepare the
exact Git index without checkout, use native copy-on-write clones for unchanged
regular files, preserve links and executable modes, and materialize changed
entries from that exact index. Clones have distinct inodes and independent
writes. I do not omit evidence, source or fixtures to fit. Other platforms keep
ordinary full checkouts. The real Git/APFS fixture checks baseline differences,
links, modes, whole-tree equality and write isolation. All four local driver
controls pass. The corrective push starts only Apple; the already-running Linux
comparison is not restarted.
