# Coverage checkpoint restore

I retain the successful coverage compiler generation and packaging in run
39276986919, job118149401524, artifact12204047789. The dependent coverage job
118186023879 then refuses continuation with `I require continuation without a
timestamp-induced completed-generation rebuild`. This is a restore dependency
failure after environment/checkpoint validation, not successful coverage testing.

I identify two reproducible dependency problems:

- `bootstrap_input_tree` intentionally includes visible source directories, so
  additions and removals invalidate completed generations. The checkpoint retained
  files and links but no directories. Creating restored directories changes these
  real prerequisites. I retain their original mode and nanosecond modification
  time and restore those records after every child file/link is installed.
- The generated schema header has a stamp dependency and a comment-only recipe.
  When the stamp is newer than the unchanged header, actual Make executes the
  no-op without rebuilding newer objects, but `make -n` propagates an imagined
  header update. I remove only the comment recipe, retaining the stamp dependency
  and actual generation rule. Real header/source changes must still rebuild.

The diagnostic manifest excerpt was extracted from a partial artifact download.
It records 87,319 file/link entries, no directories, and the schema stamp newer
than the header. It is explicitly not full archive integrity evidence. The
hosted failure log and small diagnostic artifact were downloaded independently.
The original workflow omitted the failed resume plan from its coverage upload;
I add that exact plan path for subsequent failures.

I preserve the exact source, environment, archive, receipt and no-rebuild guards.
I do not remove directory dependencies or assign invented old timestamps.
`aggregate-plan-*-isolated.log` files use explicit Make `-o` switches solely to
separate diagnosed prerequisites. They are not continuation success evidence and
those switches are not added to production. The current aggregate contains real
post-bootstrap facts-tool changes, which still correctly schedule generation.

All thirteen archive/identity/continuation tests pass in 0.487 seconds, including
fresh-tree directory restoration, real source addition/removal rebuilds, malformed
directory refusals and the actual Make schema rule. The final directory metadata
pass runs deepest-first. I record 91 visible source-directory prerequisites.
Exact hosted successful restore, coverage execution and the full release gates
remain open.
