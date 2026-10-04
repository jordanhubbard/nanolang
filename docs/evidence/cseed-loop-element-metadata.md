# My C-seed loop element metadata

My `AST_FOR` checker previously read array element metadata only from a named
receiver. An inline literal or a computed array therefore bound an `int` loop
variable and rejected valid string or bool bodies. The same shape was silently
wrong one stage later: my C transpiler recognized only an `AST_IDENTIFIER`
receiver and emitted `/* unsupported for-in pattern */`, dropping the body.

I now derive the loop element type from the iterable expression itself through
my existing inference helper, covering inline literals, producer calls,
`map`/`filter`/`array_slice`, and field access, and I retain the record identity
for record arrays. My transpiler accepts any iterable whose checked type is
`array<T>`, materializes it once, and selects the matching `dyn_array_get_*`
form.

`tests/nl_control_array_elements.nano` is my expanded probe. It covers inline
`array<int>`, `array<string>`, and `array<bool>` literals; an array returned
from a call; `filter`, `array_slice`, `map`, and a record field access; and a
filtered `array<string>`. Before the repair the string and bool forms failed
with `E001 TYPE MISMATCH` and inline integer iteration produced an empty loop.
After the repair the fixture compiles and runs to exit 0 under
`bin/nanoc_c`:

```
make -j8 bin/nanoc_c
./bin/nanoc_c tests/nl_control_array_elements.nano -o /tmp/nl_control_array_elements
/tmp/nl_control_array_elements
```

This is C-seed native evidence. My NanoISA emitter already iterates any array
expression through `ARR_LEN`/`ARR_GET`, so this repair does not change that
lowering.
