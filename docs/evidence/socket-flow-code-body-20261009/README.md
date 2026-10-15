# My TCP flow, CODE and acyclic-body checkpoint

I connect the checked TCP nominal map to one shared bounded ownership-flow,
CODE decoding and acyclic body-analysis engine. File and TCP retain separate
opaque APIs and catalog configurations. The normalized source comparison in
`shared-engine-review.diff` records the semantic changes apart from names:
exact copy-argument consumption at owner acquisition, a pending Endpoint domain
check, and an equivalent supported-opcode helper call. File's zero-input temp
operation retains its existing behavior.

I consume one exact Endpoint value for begin-connect and create an affine
ConnectResult. I require exclusive Conn references for send/completion/receive,
and consume Conn for close. I retain owner identity through Result refinement,
refuse overlapping loans and moved owners, preserve transactional failures and
record cleanup/callee/service obligations. Endpoint domains remain pending under
`NVM_SOCKET_FLOW_CHECK_ENDPOINT`; neither typing nor CODE analysis discharges it.

My decoder checks actual instruction boundaries, nominal constructor operands,
reference maps, all function bodies and acyclic control/call order. My body
analysis checks a synthetic TCP lifecycle with both Result arms, helper calls,
borrowed operations, close and cleanup. It checks uncalled functions as well.
The existing OP_FILE_* instruction encodings are interpreted only within this
separate private TCP profile; no File runtime is authorized to consume them.

| Corpus | Linked checks | Allocation-instrumented checks |
|---|---:|---:|
| TCP flow | 4,188 | 4,369 |
| TCP CODE | 2,112 | 2,739 |
| TCP acyclic bodies | 4,598 | 12,006 |
| File flow | 3,332 | 3,477 |
| File CODE | 1,817 | 2,409 |
| File acyclic bodies | 3,662 | 9,507 |

I pass all six paired LLVM ASan/UBSan/leak-detection methods per catalog.
After the GCC fixture correction I rerun all six TCP methods. I also compile
and execute all six TCP modes with Apple Clang and GCC16 using strict warnings.
The changed shared engines and TCP catalog are sanitizer-instrumented; ordinary
linked dependency objects are not all instrumented. Commands, logs, source
hashes and the original failures accompany this checkpoint.

I retain two fixture failures. Adding Endpoint local12 made the old File test's
out-of-range index12 valid in TCP; I derive refusal indices from the declared
local count. GCC then rejected three unbraced loops followed by same-line
assertions; I add braces and keep strict warnings. Neither fix changes an
acceptance assertion into a weaker success condition.

This is metadata and bytecode analysis, not TCP execution. Runtime stack ABI,
cyclic/indirect TCP analysis, matched dispatch, paired source lowering, grants,
network shadows, mixed File/TCP execution, public connect/DNS/WebSocket and
full candidate Linux/Darwin/release qualification remain open. Existing generic
and File execution consumers still refuse these synthetic modules. I do not
claim a fresh full compiler bootstrap or close #990/#976 here.
