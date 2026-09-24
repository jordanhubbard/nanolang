# Owned indirect calls

I admit noncapturing callbacks in my bounded owned value-call graph. I retain
at most eight same-module functions, an acyclic call graph and the existing
owned signature and local/stack limits. A function value has its own tag;
it does not become an integer, an owned field or a borrowed reference.

Before admission, I compute conservative callback target sets through locals,
branches, loops, arguments and returned functions. Every possible target must
have the instruction's arity and result count. I then check every positional
argument's tag, consuming ownership and nominal resource/union identity, and
require the same exact result contract across all targets. Each callee must
resolve its owners on every admitted path. Unknown targets, call cycles,
captures and mismatched contracts refuse before execution.

My VM checks the actual function against the planned set before transferring
arguments. It checks globals and parameter contracts, reserves the frame and
only then publishes the new ownership activation. I currently recompute the
bounded target plan at each owned indirect invocation; this checkpoint makes
no performance claim. My native emitter builds a switch containing only the
planned targets and reuses the checked consuming helper calls. Wrong tags or
targets take the cleanup path. Results, pending values and arguments retain
the existing success, trap and allocation-failure cleanup rules.

My source emitter records exact function-signature spelling for named
non-entry functions, parameters, local aliases and returned callbacks. It
snapshots the callee before evaluating arguments and emits `CALL_INDIRECT`.
I share the function-type spelling parser with the C emitter. Source shadows
exercise fixed resource parameters/results, returned aliases and void
consumers. Generic resource-bearing callback signatures retain their existing
refusals; this does not complete the separately tracked generic-union feature.

Mixed ownership profiles, owner-array callbacks, function-valued globals and
borrowed callback slots remain outside this source path. The scalar/aggregate
ordinary callback backend keeps its separate contract.

[Qualification and retained failures](evidence/pr522-implementation-2026-09-23/owned-indirect-calls/README.md)
record the tested scope. These are executable tests, not a formal ownership proof.
