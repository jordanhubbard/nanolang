# Mixed effect scalar carriers

I transport ordinary scalar effect arguments and resumed results as tagged
values. A monomorphic arm projects its declared native carrier with a runtime
tag check; a mixed arm retains the tagged value. Aggregate and callable carriers
keep their existing exact transport. I use the same result publication for
explicit resumption and implicit arm completion.

I join observed scalar parameter and result kinds with retained tag sets. My
shape graph receives directed conversions into finite scalar storage instead of
an equality between unrelated integer and Boolean producers. Unknown and
aggregate payloads do not gain scalar authority from this rule.

My boundary suite has thirteen methods and twenty-three fixtures. New controls
round-trip integer, Boolean, float, allocated string and void in both producer
orders, asserting each returned tag and value. Distinct handlers for the same
operation also resume integer and Boolean results without losing their tags.
Existing aggregate/callable transport, lexical exits, implicit completion and
ordinary HALT controls remain in the suite.

I leave byte-bearing mixtures open: the retained byte module verifies and exits
zero in NanoVM but native translation refuses its optional/int shape join.
This checkpoint does not complete all mixed-scalar acceptance or #976. Resource
ownership, full effect identity, foreign callback reentry and exact release
platform qualification remain open too.

I pass all twenty-three fixtures with the translator, callable and shape objects
instrumented, as well as generated C executables, under ASan/UBSan with leak
detection. Other linked translator objects are not instrumented by this focused
build. I pass the 66-case source suite through the C frontend, the earlier fresh
seed and the audited regenerated compiler, including eighteen VM/native
execution programs per producer. All seven tool hashes stay unchanged during
that paired run. My corrected translator reproduces byte-identical compiler C
from the previously audited module.

My Git coordination claim is published at `4140cfc14`. The GitHub issue comment
failed to connect; I do not claim that it posted.

My complete native regression target passes 2,443 core checks, 464 callable
constraints, 3,113 shape constraints and adjacent suites on these sources.
