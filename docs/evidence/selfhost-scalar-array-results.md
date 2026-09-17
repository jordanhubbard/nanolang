# My scalar array results and empty literals

I lower integer and string array results through the existing array result tag.
I carry declared scalar element types into literals in locals, assignments,
function arguments and returns. An empty string array now emits `ARR_LITERAL
5 0`, matching the C seed; its integer counterpart keeps tag 1. Literal elements
must match their declared scalar type. I retain refusal of nested and boolean
array results in this emitter slice.

On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline comparisons
and 17 focused Python cases. Fourteen new bytecode/function checks match the C
seed for empty and populated results, direct and tail calls, literals containing
string expressions, assignments and empty arguments. Both emitted modules
verify and execute in NanoVM and strict native C11, including string pushes and
reads after a reset. Seven unsupported shapes or mismatched elements fail
without output. The separate native mixed-array helper fix preserves strict
compilation of the fixture; its full translator gate passes 1,769 checks and
1,076 shape checks.

Real `src_nano/nanoc_v06.nano` emission passes the array-result boundary and
first refuses strings needing assembly escaping. I recorded continuation
`task_98ec870924bf4b82b9ba5e2591576488`. Tasks
`task_4ea96b68ae4b43f7a0cfc16cd7c19649` and
`task_0443ff4ed6224f5683301055555f4209` track these array fixes. Full compiler
emission and bootstrap equality remain open.

On Darwin ARM64, six focused driver cases also prove that I report the
lowerer's precise refusal while retaining the generic fallback when no
diagnostic exists. Refusal does not publish a new output or replace prior
output.
