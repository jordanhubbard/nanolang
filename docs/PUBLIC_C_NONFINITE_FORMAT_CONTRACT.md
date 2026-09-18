# My public C nonfinite formatting contract

I execute `task_132ebac62ed842c38240522badf4ffb4` on main dd99573e after merged PR748.

I implement the public C target checkpoint under signed nonfinite policy task_e92a45b66a104e9ba3854cd5f994df8b, after merged scalar PR748. I consume unchanged NL_BINARY64_FORMAT_SOURCE with the backend private namespace. Exact FLOAT builtin float_to_string and print/println evaluate their operand once; declarations, lexical bindings, func_expr and checked call signatures cannot be rewritten as builtins by spelling. I preserve finite %g and explicit nan/-nan/inf/-inf, input bits and publication refusal.
Static lifetime audit finds the old GNU float_to_string block returns one mutable static buffer per callsite, so repeated calls can overwrite earlier aliases. I replace this path with a portable private helper returning a stable owned snapshot, tracked to generated-process exit with checked allocation/atexit registration and exactly-once cleanup. I claim process-lifetime retention, not tracing reclamation or an RSS bound. Ordinary aliases across repeated helper calls, loops, globals and returns plus ASan/UBSan/LSan and isolated post-repair failure injection qualify this ownership boundary. Existing int_to_string GNU blocks, concat allocation and broader6ade remain independent explicitly recorded requirements, not full-target conformance.
I require frozen GCC/Clang C99/C11 O0/O2 exact output and bit-observer controls including both NaN signs/kinds, infinities, zero signs, finite examples, once evaluation and private-name collisions; adjacent arithmetic/API/source suites remain. Linux results do not establish Darwin acceptance; e92 platform parent remains open. Contract/roadmap precede code, production checkpoint precedes fresh execution, historical artifacts stay untouched.

My first frozen controls passed both generated cleanup/failure and scoped API
methods, but the two source fixtures used an invalid equals sign after set. Both
GCC/Clang logs retain those parser refusals; no rejected source output ran. I
correct fixture syntax, preserve all 11 verified identities and freeze again.
