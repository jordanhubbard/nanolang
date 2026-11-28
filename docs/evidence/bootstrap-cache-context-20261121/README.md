# My retained module-cache context

Under #1164 I reproduce three assertions failing in two real-receipt tests:
my standalone resumed process loses the cache context used by Stage1. Make
exports an absolute obj/module_cache default; my earlier private controller
did not retain that setting. Both shared-library builder routes select the
cache root from NANO_BUILD_CACHE. I do not infer a builder defect from their
different entry points.

My version2 receipt records the selected value, including absent and empty
settings. I restore it when the resumed process environment omits it and
refuse an explicit conflict, malformed context or legacy receipt. I preserve
source, tool, host, deadline and native-code guards. All36 bootstrap boundary,
message, source-dependency and checkpoint checks pass after correction.

My first actual allocation-policy compiler generation completed in742.020
seconds and verified, but failed exact raw and host-path equality. Its six
native imports have identical library hashes and different generation paths.
An in-memory dump comparison identifies those path differences; it grants no
raw equality and changes no module. I retain that full generation separately.

Restoring the original Make cache in a small import probe then fails after
6.064 seconds: my original native guard rejects a shared-link request against
a different cached generation. I preserve its marker and cache-probe log.
The subsequent full generation never starts. I leave that original receipt
failed; I do not reset its marker or claim that its Stage1 can resume. A fresh
complete bootstrap with recorded context, exact equality and unchanged1800/30
budgets remains required. This checkpoint does not close #1164 or #1161.

My manifest retains15 inputs,584822 raw bytes and checked raw/compressed
SHA-256 identities, including baseline/corrected tests and the refused probe.
