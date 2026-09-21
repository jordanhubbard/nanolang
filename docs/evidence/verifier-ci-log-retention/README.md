# My verifier diagnostic retention

I select a dedicated verifier log root in hosted CI and upload that exact tree
on build, coverage and sanitizer job failures, with seven-day retention. My
local default remains a unique temporary directory. Successful corpus runs
remove only their own directory; failures retain diagnostics and their original
exit status.

My shell syntax and workflow YAML checks pass. A real compiler/VM smoke run
selects one valid and one invalid source: one verifies, one fails, status1, and
its compile log remains. A following valid-only run returns0, removes its own
logs, and preserves every prior failed-run log hash. These checks use the fresh
883dac9d5 compiler/VM tools in the retained qualification workspace. They do
not claim that a hosted upload has run; actual CI artifact retrieval remains
required before closing the task.
