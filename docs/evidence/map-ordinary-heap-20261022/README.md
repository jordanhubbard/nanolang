# Ordinary map adapter checkpoint

I implemented only checkpoint1 of contract7abaf9f7f. Five private ordinary heap
adapters accept validated ordinary collector headers, reject running GC/private
DAG heaps, and preserve counted STRING→INT topology and retain ordering. Existing
private adapter source is byte-identical. No core/view/root activation, dispatch,
public selector, native or publisher admission changed.

The first strict O1 direct gate passed146 assertions in0.804s; it was not an
instrumented claim. Expanded O1 ASan/UBSan providers and harness passed3474 checks
in1.287s. The old private adapter suite independently passed10318 checks with the
same sanitizers (heap6312B). I retain the expanded harness's first compile failure:
VM_GC_GREY was a test spelling error; the actual enum is VM_GC_GRAY. Only that
spelling changed before the final run.

Controls cover real buffered PURPLE maps, unbuffered PURPLE after suspect-buffer
allocation failure, counted-NUL keys, independently allocated equal-content key
replacement at maximum references preserving the stored key, insertion overflow
before allocation, missing GET/HAS, invalid header restoration,20 colliding keys,
both NEW allocation positions and growth failure under transient/persistent denial,
retry, ordinary cycle cleanup and independent allocator balance. No arbitrary
forged-pointer safety or heap identity registry is claimed; managed ownership,
validated topology and intern integrity remain explicit C preconditions.

Eight GCC16/Clang syntax cases passed (ordinary/private/combined heap and new
harness). The existing six-budget executable was reused, not rebuilt, and matched
all six prior rows exactly (Analysis39840). No analysis/layout source changed;
this is retained budget evidence, distinct from the freshly built heap tests.

Commands:

```
python3 -m unittest tests.test_record_map_ordinary_heap
RECORD_ARRAY_VM_CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' python3 -m unittest tests.test_record_map_ordinary_heap tests.test_record_map_heap
RECORD_ARRAY_VM_CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' python3 -m unittest tests.test_record_map_ordinary_heap
make -n -f Makefile.gnu test-record-map-ordinary-heap
```

The combined command had one compile failure and one passing old suite; it is not
reported as fully green. Required target `test-record-map-ordinary-heap` is wired
into test-units, inherits compiler/instrumentation flags and needs no frontend or
provider rebuild. Raw test directories and exact commands/statuses are retained
in the compact files here; source hashes are in pins.json. Disk was33GiB free at
99%; no large builds or deletion occurred.
