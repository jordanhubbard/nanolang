# My tagged map globals

I reproduce the missing native path with a string map initialized in `__init__`,
loaded through `LOAD_GLOBAL`, and consumed by `HM_SET` and `HM_GET`. NanoVM runs
this program; my previous standalone translator refuses the tagged receiver.

I retain the real hashmap tag in global and optional local storage. I unbox it
with a checked map conversion at operations and returns, and box direct maps at
mixed call boundaries. My shape graph keeps the tagged producer separate from
the checked operation's map shape. I do not infer that every assignment to a
global has the same type. My existing map runtime still checks inserted values.

My collector follows maps in tagged globals, locals, operand slots and record
fields. My tests remove the global reference before collecting when they test
another root. Each positive native fixture uses AddressSanitizer,
UndefinedBehaviorSanitizer and leak detection, and asserts at most 32 live-owner
allocations at the peak. My collection fixtures run a helper that creates
20,000 temporary maps. I also
check initializer function returns, aliases, reassignment, mixed direct/tagged
callers, tail calls, missing keys, identity, same-tag ordering, truthiness and invalid receivers.

I keep two boundaries open:

- Whole-record globals remain on MAC `task_95796f5f49564ed4a911fd05a1aac5b4`.
  This map slice is `task_af839ea3c3d14ebfa3191a0322f08298`.
- NanoVM accepts a string value inserted into a raw map declared with integer
  values; my existing native map runtime rejects it. I preserve both observed
  outcomes in the negative fixture and track the raw contract reconciliation
  as `task_b19f8bf0527d4a33911be26706629616`. I do not relax native checks.
