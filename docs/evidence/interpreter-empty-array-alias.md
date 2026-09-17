# I preserve empty-array aliases in my interpreter

I initialize an empty static array's element storage in its existing `Array`
object. Previously, my first append returned a separate dynamic array (or a
separate nested static array), leaving aliases empty. My record-field probe
failed its shadow at line 14 before this repair.

I retain the representation and identity of nested arrays, including dynamic
inner arrays. Record and string elements use my existing copying setters. I
also support pop and removal on static arrays: preserving their representation
must not break a sequence that previously converted to dynamic storage.
Pop copies owned payloads before removal; removal releases owned string/record
slots, shifts the remaining slots, and clears the vacated slot. Nested arrays
remain shared.

I checked these boundaries:

- `make test-eval`: the full interpreter suite passes. My new test repeats
  primitive, string, record-field and nested aliases, growth, ordered receiver
  and value calls, pop/refill, removal, and record/string payload lifetime.
  A direct runtime probe also checks a dynamic inner array's tag and identity.
- The same full interpreter suite passes with AddressSanitizer and
  UndefinedBehaviorSanitizer at `-O3`, with invalid accesses set to halt.
- `python3 -m unittest -v tests.test_cseed_record_array_literals
  tests.test_cseed_nested_array_literals`: all six methods pass. The append
  probe now checks both empty and nonempty record fields through C-seed
  shadows, native execution, and VM execution.

I retain the baseline failure and test logs under
`/tmp/nanolang-interpreter-alias-{baseline,eval,asan,parity}.log` in the working
session. These local logs are supplementary; the committed tests reproduce
my checks.

I do not claim interpreter leak freedom. My existing environment teardown
ignores static array containers and their elements, including nonempty
literals. I used `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` for invalid-access
checks and recorded alias-safe reclamation separately as
`task_5eba51e216e343549c8ca7846713b6a3`. This repair does not change generic
resource-collection acceptance or native/VM array policy.
