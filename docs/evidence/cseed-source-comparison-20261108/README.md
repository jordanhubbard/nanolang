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
inputs remain available once GitHub registers the workflow.
