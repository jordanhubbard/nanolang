# My checker-only record parameter boundary

I pin source `1a75e05b48ad4a937dc87e220e8593de21617119`. My reader verifies
2,215 records, 1,438 exact Git inputs and 43 distinct native ELF images.

Under #1149 I replace module-checker empty runtime records with type-only
parameter slots, matching whole-program checking. I retain declared nominal
names and direct/nested/array field validation, wrong-field/type/nominal
refusals and all runtime value ownership. I also return the registered record
field identity from the nested-field resolver, matching its borrowed query
contract. No other production source changes in this checkpoint.

My complete native fixture repeats six source cases through both checker
entries three times: 36 controls. The original code passes its semantic
assertions and fails LSan at 615 bytes/51 allocations. The placeholder-only
candidate still fails at 36 bytes/six nested-field name copies. I preserve
both complete failures, source and images. The unchanged fixture then passes
normally, with ASan/UBSan+LSan and with original production O3/Werror flags.
The real new Make target passes in all three profiles.

The original 88 checker methods and original 18 plus 16 session REPL methods
pass in all three profiles. The original full checker sanitizer fixture keeps
its explicit disabled leak boundary; the focused fixture and REPL34 keep leak
detection enabled. All seven admission methods pass normal/production and broad
ASan/UBSan. The complete strict admission fixture still fails three methods at
168/172/172 bytes in unchanged runtime record-call/local copies. I preserve the
complete failures and keep #1116/#1149/#1205/#1227/#1228 and release acceptance
open. Checker-only cleanup does not establish general aggregate ownership.

My source-built changed-checker C seed passes ten unchanged lexical methods and
seven record/list payload methods. Its full compiler-source command exits 1
after 140.893 seconds: all 1,306 selected shadows report start/completion with
zero assertion failures, then the supervisor reports its original ten-second
limit and refuses publication. No seed exists. I retain the original 600-second
outer limit and do not retry unchanged, assign a cause, or claim source/bootstrap
acceptance. #1229 tracks terminal-phase/process diagnosis and required full
source qualification. Per-case completion does not establish process success.

I retain two test setup failures too. My first direct unittest command omitted
the fault-observer library path; the complete 34-method result retains that
single missing-file failure. Supplying the built observer restores the original
fault controls. My first private C compiler location selects `.test_output` as
its source root and native includes fail. I retain those streams/products, then
invoke the exact same image bytes in the existing repository `bin/` layout.
I make no source-root-independent installation claim from this setup correction.

I compile only the changed checker object in three original profiles, link to
hashed retained shared providers, and preserve the earlier REPL component.
All unchanged shared object/archive bytes match my preceding
[REPL journal receipt](../repl-session-save-linux-20261123/README.md).
I retain five unchanged public tool guards and nine host-library identities.
I perform no fresh full shared-provider build, Darwin job or raw qualification.
Frozen roots, audits/stashes, Puck source/capture/regex/CI and NAS/TCC/migration
pauses remain unchanged.

Run `python3 docs/evidence/checker-record-parameters-linux-20261123/read_archive.py`.
My reader checks exact committed source, the two precise production hunks,
compressed/raw artifact identities, retained failures, actual compile/link/Make
commands, native fixture products, provider continuity, full shadow selections
and terminal refusal, and explicit incomplete acceptance boundaries. Reused
objects resolve through hash-verified relative archive paths.
