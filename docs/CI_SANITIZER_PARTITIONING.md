# I preserve my complete sanitizer suite when I schedule it

## My integrated PR522 schedule

I retain run `35690205396` as failed. Its `units-03` worker enters the scalar
reconstruction suite at 05:38:05 UTC and is terminated at 05:53:24 after many
passing methods. I give the complete scalar target a dedicated worker with a
35-minute test bound and a 45-minute job bound. The other workers retain their
20-minute test and 30-minute job bounds. My integrated schedule has eighteen
workers: Forth, source emission, scalar reconstruction, fourteen remaining unit
partitions, and the negative suite. Every original target still occurs exactly
once. I require a completed hosted run; these bounds do not imply acceptance.

I accept GNU Make 3.81's empty tab-only recipe line while requiring the same
sole nonempty tail invocation. My actual-inventory regression covers Darwin,
and my malformed-recipe controls still refuse additional commands.

The following sections retain my earlier schedule and its evidence.


## Measured failure and scope

I retain PR947 head81cb6eb50 and the earlier PR946 whole-step timeout. My actual947 sanitizer step begins at16:19:11, reaches the Forth session at16:21:27, then the next target at16:32:44. That target interval is11m17s, including its compile and execution; I do not attribute all of it to one function. I reach nvm2c at16:33:18, build the instrumented NanoISA emitter at16:34:59 and enter its comparison at16:36:09. Sixteen Python methods then pass before Actions enforces the unchanged20-minute step limit. The first terminal remains a failure. A timeout increase alone does not establish full-suite acceptance.

I start from actual main e0c7eb76d. GNU Make's read-only query database resolves282 distinct test-units prerequisites. The target also owns a trailing compile/run/remove transpiler recipe: an inventory of prerequisites alone would silently omit it. My separate negative-test step is required too. All are retained.

## Implementation contract before code

I factor the existing trailing recipe mechanically into a named target and keep ordinary test-units depending on all282 existing prerequisites plus that target. Its original instrumentation detection, compile/link flags, executable assertions and cleanup remain unchanged. I retain a byte comparison of the moved recipe. I do not replace original targets with copied test commands.

A small scheduler reads the resolved GNU Make query database, rejects parse errors/duplicates/unknown forms, and publishes the complete ordered inventory and exact partition manifest before any test. It recognizes only test-units' resolved normal prerequisites; order-only or unexpected non-test entries are errors requiring review. Tests compare partition union and multiplicity against that live inventory, including the trailing recipe. Additions to test-units must enter the manifest automatically or fail validation, never disappear.

My initial schedule has16 disjoint workers. Forth session and NanoISA source-emitter acceptance each own a dedicated worker because the retained timings identify them as substantial phases. I distribute the remaining281 entries deterministically across14 workers in resolved order. These are scheduling partitions, not smaller acceptance profiles. I retain the exact target arguments per worker and prove every original target plus the moved tail occurs exactly once in the requested union. Make may execute shared prerequisite tests again across independent workers; I report that duplication instead of claiming unique transitive execution.

Each worker checks out the same exact commit, independently builds the same sanitizer compiler/runtime and completes the same three-stage bootstrap. There is no unproved archive/provider reuse. I keep the existing CFLAGS, LDFLAGS and detect_leaks=0 policy. I select the inventoried distribution `clang` for generated native products while retaining `-O1`, ASan, UBSan and frame pointers; this targets the measured Stage 2 native compile without weakening its instrumentation contract. The source-emitter worker prepares its existing named compiler/emitter providers in an explicit build-preparation step with those same flags before its test step; moving preparation does not remove any shadow or product control. Other targets continue to own their actual Make prerequisites.

Each test worker retains the original20-minute step limit and30-minute job limit. Independent workers run with fail-fast disabled, at most four concurrently; an actual failure remains a failure and does not cancel unrelated evidence. No worker reruns a known failed target merely to obtain green. Sixteen jobs imply an explicit480-minute worst-case job allocation, not measured duration or a passing claim. Existing inner deadlines and test bodies are unchanged. If a partition exceeds its original bound, I preserve that measured terminal and revise scheduling only with exact complete-coverage evidence. I do not increase the worker bound to hide it.

The existing negative suite runs in a separately bounded sanitizer worker with the same original selected compiler and ASan policy. Its acceptance is required alongside every partition. I retain per-worker source/tool/provider identity, actual command/exit/timing, raw test logs, and first-failure diagnostics with always-run artifact publication. Outer deadline handling must leave Actions' descendant cleanup evidence intact; local qualification uses bounded process-group cleanup. A final aggregate check requires every expected worker and negative lane to succeed, with exact matching head/inventory digests. Missing, cancelled or failed workers cannot yield aggregate success.

## Qualification and limits

Before hosted execution I review the entire workflow/planner/Make delta, run scheduler coverage/refusal controls, validate workflow YAML and inspect actual dry-run command/flag correspondence. Then I exercise the unchanged instrumented Forth and source-emitter worker selections on fresh providers, preserve their actual terminals, and use hosted matrix results for the complete platform-specific acceptance. Local supported mixed instrumentation from PR947 proves only its link boundary and cannot stand in for this all-provider sanitizer matrix.

Known byte-source/list corpus refusals and any additional assertion, sanitizer, or deadline failure remain required repairs. This scheduling work does not close those tasks or claim the release ready. PR945 generated-backend integration onto current main remains separate; its older qualified source is not relabeled current.

## Source checkpoint

My first implementation resolves283 targets after factoring the original863-byte trailing recipe without changing its bytes. Eight local scheduler tests pass, including live read-only Make inventory, exact multiplicity, future-target inclusion, manifest tampering, missing/duplicate/failed/wrong-head aggregate refusal and workflow-limit controls. These are scheduler tests, not product acceptance.

I retain worker reports outside the checkout in RUNNER_TEMP because the original make sanitize invokes clean and deliberately removes .test_output. Workers publish their actual surviving bin/obj/obj-runtime providers separately from reports, so the aggregate downloads only small result/report artifacts. Existing test recipes can delete inner temporary executables; I do not claim those deleted bytes are retained. Source/tool maps are checked before test execution and after the terminal; provider maps describe each measured endpoint without claiming unchanged products across legitimate builds. The negative worker alone receives the original NANOLANG_COMPILER selection, and its test command clears the shadow override that previously applied only to build/unit steps. Sixteen unit jobs plus one negative job retain an explicit510-minute maximum job allocation; inventory/aggregate each have5-minute coordinator limits. No hosted execution has occurred at this checkpoint.

## I preserve the ordinary parallel-Make tail ordering

Independent review of787ffceb1 found a real ordering regression before hosted execution: making the old trailing recipe another prerequisite permits it to run concurrently with earlier tests under make -j, including a test that writes the same executable path. The original target recipe waited for every prerequisite. I retain787 local observations but do not publish that graph as accepted.

Before correction I specify one ordinary test-units recipe: recursive $(MAKE) test-units-tail after all original prerequisites complete. The isolated tail target keeps the original863 recipe bytes and its real object prerequisites. The planner must validate that exact sole tail invocation in the resolved Make database and add its isolated obligation exactly once to the original282 prerequisites. An absent, extra or changed recipe is a refusal, not silently ignored work. This retains ordinary parallel ordering and the complete283-worker inventory. I add an actual make -j control with multiple prerequisite completions and an observed tail-order assertion, plus missing/extra/changed-tail planner refusals. This graph/planner correction does not change a product test or justify replaying the known120-second source-worker failure.

My corrected checkpoint passes nine scheduler controls, including actual parallel Make ordering and refusal of missing, extra, or changed tail recipes. The isolated tail remains byte-identical to the original863-byte recipe (SHA256 a5a50e58d910ace9e089c90854367ac1cc5fb1b502783e178ed511cb90f72b06). I retain the original787 local product terminals separately: the source worker completed89 of90 methods before its unchanged120-second compiler-child deadline failed; the Forth worker reached its unchanged1200-second test-step deadline. Neither terminal establishes acceptance, and neither is replayed for this graph-only correction.
