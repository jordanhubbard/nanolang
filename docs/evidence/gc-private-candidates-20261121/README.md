# My private collector candidates

I retain completed observations under #1161 while keeping full generation
and production source promotion separate.

My actual packed-array-only candidate fails full raw self-generation after
1800.119 seconds. Its last marker remains `typecheck-start`; no Stage2 module
exists. After failure, my original source/tool/host and compiler-artifact
guards pass. This candidate fixes the immediate packed-leaf reclamation
regressions but does not repair the bootstrap deadline.

My next private candidate forces child-walker inlining on GCC/Clang, with a
plain inline fallback for other compilers. I preserve every trial edge,
reference-count operation, root trigger, heap layout and original assertion.
All 299243 VM assertions pass. My targeted VM/heap/collector/value and test
ASan/UBSan run, with leak checking, also passes those assertions and the
unchanged allocation-failure driver. Unrelated providers remain their actual
prebuilt objects. The independent, raw-equal hello-controlled 30-second CPU
observation contains 2998 samples, still predominantly in collector callbacks
and header checks. I do not infer full-generation acceptance from that sample.

My further private candidate projects trial deletion onto cycle-capable
objects. My argument is that a vertex with no outgoing heap edges cannot
close a cycle. Removing those vertices leaves the incoming counts and cycles
of the remaining graph unchanged. I retain leaf reference counts during
trial deletion, then release each original leaf ownership edge once when
detaching a dead parent. Packed arrays and strings cannot be buffered roots
under the preceding repair; I keep boxed-cycle processing and the 4096-root
trigger. I count leaves actually reclaimed to preserve the existing collected
object report. Shared live payloads and field-name ownership remain required.
This is a design argument supported by the tests, not a machine-checked proof.

That private candidate also passes all 299243 original/current VM assertions,
including unchanged shared-payload, dead-leaf, live-field-name and boxed-cycle
checks. The same targeted ASan/UBSan/leak and original allocation-failure
controls pass. Both actual full generation runs remain separate. Their
running logs/products are deliberately outside this completed-observation
receipt; neither candidate has been promoted into production source.

My manifest retains 118 logical files, 179,517,122 raw bytes and 46,043,325
unique compressed bytes with checked raw/compressed SHA-256 identities. It
preserves actual candidate sources, images, test products, commands, CPU data,
original deadline failure and post-failure guards. In the experimental
manifests, checkout pins identify base checkout provenance; the explicit
private collector source hashes identify the changed code. They do not
assert that the private source delta was committed at that checkout pin.
Fresh authoritative bootstrap, raw equality and Darwin remain open.
