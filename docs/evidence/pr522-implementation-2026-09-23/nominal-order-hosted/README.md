# My hosted native nominal-order failures

Run `35974675846`, macOS job `107552245349`, exposes ten subcase failures
in `tests.test_native_nominal_order` on both native bootstrap stages. I accept
record and mixed cycles where the tests require refusal and preservation of
prior output. I reject integer-list record fields at native translation and
union locals containing integer/string lists during canonical lowering.

The complete job log is retained. I track correction and fresh-stage owning
qualification in `task_2a1eaa9ecdc94d489c5f731d4daf416d`. These remain unresolved; local translator
representation checks do not qualify source-level nominal compatibility.
