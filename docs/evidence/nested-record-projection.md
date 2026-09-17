# I retain record identity through array projection

My checked array accesses now retain their nominal element name on the call
AST while the local type environment is available. My bytecode generator
already consumes that annotation. Without it, direct field access could fall
back to the first record containing the field name and emit the wrong field
index. The observed shadow failure was not evidence of destroyed nested state.

The original typed-local fixture and direct `at` and `array_get` variants pass
shadow execution and standalone VM execution, including repeated field names,
nested arrays, nested records, fields of returned array elements and append
aliases. Existing native record-array construction and rejecting nominal
controls remain in the same focused gate.

Native compilation of the full original fixture is separately blocked by
record-array field append storage (task_12805797d36043cd875792788f330520).
I do not claim native parity for that unrepaired operation.

Evidence: /tmp/record-projection-current.log (before),
/tmp/record-projection-fixed.log, /tmp/nanolang-projection-tests-final.log.

Fresh three-stage bootstrap, the typechecker suite and three nominal-array
contract methods pass. The focused record-array gate passes all three methods.
