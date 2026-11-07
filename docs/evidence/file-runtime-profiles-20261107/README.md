# File plan and private-VM compile profiles

I share the actual File hosted-plan/private-VM flag, provider and fixture selections between my two original native harnesses and the optional diagnostic planner. I preserve linked/allocation-instrumented modes, exact-source inclusion instead of duplicate providers, the instrumented value-source substitution, and loader/fork sentinel hooks. I change no compiler/runtime C and enable no public route.

My [historical recipe comparison](recipe-equivalence.json) matches exact command vectors and flag/provider order against code-only `ea5e28e89af9df817d74d2ba9cf5087227f7ee36`, for both allocation modes and both sanitizer selections. Every original native test method and assertion runner has identical Python AST. This is configuration comparison: I record constructed argv and execute no compiler, native test or lifecycle action. I do not count it as native acceptance.

Python sources, embedded workflow programs and YAML parse; `git diff --check` passes. Local disk remains below the two-percent free-space threshold despite 26 GiB free, so I keep native build fan-out on hosted workers. My branch-scoped workflow checks actual worker resources, freshly builds the original Make prerequisites with selected GCC or Clang, runs both original ordinary and sanitizer targets, then compiles all selected strict ordinary/ASan/UBSan diagnostic profiles. It retains exact compiler identity, original native outputs/status and diagnostic argv/results.

Actual hosted results are pending. I keep this roadmap row, broader #1046 inventory/budget, main integration and native adoption/release acceptance open. The existing broad profile audit and PR #1068 native gates remain live and untouched.
