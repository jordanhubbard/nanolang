# My helper-owned source locals

MAC `task_c2464342430c4ed0a51401aec227da67`, dependent on merged runtime prerequisite
`task_74170cc4b1784e68970017da2a3ec64a` (PR696).

I admit exact finite resource-record locals inside my single borrowed helper.
I reuse construction in declared layout order, whole-owner moves, consumed-local
reassignment, full destructive patterns and existing exact ownership joins.
Every reaching return and lexical exit must consume source owners explicitly;
only already-proved disposal holders may be drained by terminal cleanup.

I preserve one through eight borrowed-only formals and their exact modes.
A local-owner scalar observation uses a temporary reference descriptor at
`arity` in the helper, after every inherited formal descriptor. The observed
owner itself guarantees a local slot beyond those formals. Entry observations
continue to use descriptor zero. The temporary region ends immediately after
the read; inherited descriptors remain in their enclosing call region. I
retain caller/local frame provenance from the qualified runtime.

I keep deeper calls, owned parameters/results, hidden owner expressions,
partial moves and local-owner field writes refused. Scalar writes through an
exclusive borrowed formal remain supported. I do not expand runtime authority.

I require both source producers and Stage1/Stage2 to agree on layouts,
ownership, names and canonical assembly, then execute VM/native results.
My positives cover multiple formals, local reads followed by inherited-formal
reads/writes, nested local fields, repeated calls, lexical shadowing and full
consumption across admitted control flow. Every selected shadow must lower and
execute. My refusal controls preserve old output and require semantic reasons;
missing consumption, wrong nominal identity and parameter authority remain
checked. Full normative ownership parents remain open.
