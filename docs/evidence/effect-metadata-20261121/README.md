# My evaluator effect-metadata ownership checkpoint

I resume #1149 with a fresh-provider GCC 13 ASan/UBSan run, leak checking
and every original evaluator assertion. It reproduces 29347 leaked bytes
in 578 allocations in 39.050 seconds; its last visible case is my original
terminal-match refusal control.

Under #1163 my new bounded effect entry finds a dormant basic declaration
test using obsolete operation syntax. My first grammar-only correction then
reaches its missing required main entry. I retain both setup refusals. I
restore current syntax, add the required entry and useful executed shadows,
retain the original noop result42 assertion, and enroll that method in the
complete original suite. My bounded entry invokes all twelve effect/handler
methods and makes its completion visible before sanitizer teardown.

Before changing ownership, that exact restored test set reports 25627 bytes
in446 allocations in the effect entry and 29765 bytes in583 allocations in
the complete driver. I do not compare its total directly with the earlier
test set: the restored basic declaration adds a method and AST nodes.

My only production behavior change frees the environment's registered copies:
effect/operation/result names, operation arrays, parameter arrays and copied
parameter names. My checker allocation list retains original spellings;
complete parameter/result annotations and declaration types stay borrowed
from the AST. I confirm that ownership against the sole registration producer
and the AST destructor, then run the same restored tests with fresh providers.

Those matching sanitizer runs now report 9214 bytes in247 allocations for
the effect entry and13352 bytes in384 allocations for the complete driver.
Both remove16413 bytes in199 allocations. Neither remaining report contains
an `env_define_effect` allocation stack. I preserve both full failures and
all residual direct-allocation groups. Remaining callable, AST, record,
array and binding owners stay open under #1149/#1140/#1116. I use no leak
suppression and change no fatal-child or refusal assertion.

With fresh ordinary providers and a matching native FFI fixture, the complete
evaluator executes134 named methods and passes; all twelve effect/handler
controls also pass. The original Make recipe removes its successful complete
test executable. I retain that terminal and separately derive, execute and
retain a driver with the same source/providers/flags and an explicit output
path. Its command, hash and complete passing terminal identify that actual
executed product. The bounded driver and failing sanitizer images are also
retained. Provider hashes are observed after these runs.

My manifest preserves72 logical inputs,301432780 raw bytes and56908283 unique
compressed bytes with checked raw/compressed SHA-256 identities. Complete
sanitizer success, C-producer/module/native qualification, Darwin and canonical
integration remain required. This checkpoint does not close #1149.
