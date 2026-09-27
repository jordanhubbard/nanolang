# My signed string length carrier

I declare str_length(string) -> int. My evaluator returns an INT through
nl_cstr_length; my NanoISA C backend explicitly casts strlen to int64_t.
My native C-seed registry and Nano emitter instead name bare strlen, whose
size_t result can change arithmetic before any destination conversion.

My retained802 native trace enters the unchanged transp_str_index_of shadow
and reaches the ten-second deadline there. Source-only Stage1 emission gives:

```c
while ((nl_expected < 0) &&
       (nl_position <= (strlen(nl_text) - nl_width))) {
```

A needle longer than the text makes this unsigned subtraction wrap. I retain
all eighty differential controls; I do not cast their operands or narrow their
vectors to conceal this result-carrier defect. Inspection emitted C only and
did not compile or execute that output.

I map the C-seed registry to its existing int64_t nl_cstr_length runtime API.
Both ordinary C emission paths consume the iterative expression emitter and
its unified registry; their generated headers already include nl_string.h.
My separate NanoISA C backend already converts the result and needs no change.
The evaluator already uses the same runtime API. The Nano native emitter maps
its existing str_length/str_len spellings to a local int64_t nl_str_length
helper, with the same NULL-to-zero and explicit size_t-to-int64_t conversion
as nl_cstr_length. I add no external provider or new accepted source spelling.
Actual source strings remain NUL-terminated byte strings. I do not claim a new
contract for strings exceeding INT64_MAX or change the existing runtime cast.

I preserve call/argument evaluation and declaration selection. An explicit C
foreign strlen declaration is a distinct name and keeps its existing mapping;
I do not rewrite arbitrary C calls. Existing runtime bodies and all original
18+8 tests stay unchanged. I add primitive negative arithmetic/comparison,
empty/multibyte length, nested-call once-only evaluation and return controls
across all three native producers at O0/O2 through the retained supervisor.
Full original bootstrap and source/native matrices remain required after review.

MAC: task_716077fe2eb14ec89ead2ce9f2bb3cc2.

My source inspection uses the actual map_builtin_func_name mapping in the Nano
emitter; its existing declaration-selection callers remain unchanged.
