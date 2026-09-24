# My exact scalar callback repair

I use my existing guarded exact-target native lowering for admitted scalar and
void results, as well as aggregates. A known boolean-returning target no longer
competes with unrelated integer-returning functions. I retain the runtime
function-identity guard, signature and argument checks, and conservative
handling of ambiguous targets and backward control flow.

I ran `make -j2 test-nvm2c`: all 2,582 translator assertions pass. The added
matrix checks int, bool, float, string, function and void results in both
function declaration orders, including selector/target evaluation once in
order. Existing aggregate and incompatible-argument refusals remain checked.
I updated two diagnostic assertions to the generalized argument mismatch;
I did not remove either refusal. My intermediate logs retain those stale
assertion failures and a corrected assembly fixture spelling (`PUSH_BOOL 1`).

The unchanged lexical signature/restoration method passes through the C seed
and both existing native stages. The full unchanged six-method declared-push
suite retains two failures: both native stages still reject a runtime function
parameter alias in the producer. I have not rebuilt those stages for this
translator-only change. Their hashes and the tested translator are recorded.
The earlier four-failure baseline remains in the record-projection-name-lifetime
evidence directory.

Fresh isolated ASan/UBSan qualification is running separately; this checkpoint
makes no sanitizer completion claim. This is partial progress on
`task_560bf9f1fca645d7aff9b71004ee3859`, not completion of #522.
