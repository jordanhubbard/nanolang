# My scalar call-frame lifetime observations

I investigate #1139 and #1140 separately from retained record carriers under
#1116. I keep production code unchanged during this observation. My probe uses
the actual lexer, parser, checker and evaluator, with a fresh baseline env
object and fresh eval/io objects linked to hash-bound cached dependencies.
It is not a fresh full-provider or complete NativeSDK qualification.

I first validate Darwin allocator statistics with a real2MiB allocation/free:
live bytes rise from177008 to2274160 and return to177008. I use scalar
parameters and return values in every measured call. The global Item record
is created once outside the measured loops; these batches construct no record.

| Route,5000 calls per batch | First batch live-byte change | Second batch |
| --- | --- | --- |
| Clean named scalar call | 0 | 0 |
| Named scalar parameter inheriting same-name Item metadata | 80000 | 65568 |
| Clean first-class scalar call | 649536 | 643840 |
| First-class scalar call inheriting Item metadata | 725728 | 754816 |

Every result equals7 and every call restores its original symbol count. These
are pre-teardown live-byte observations, not exact allocation counts or proof
of post-teardown leaks. I do not attribute all first-class growth to metadata,
one signature clone, record carriers or the historical source deadline.

`env_define_var_impl` duplicates same-file `struct_type_name` metadata. Both
named and first-class call cleanup loops free binding names and STRING/FUNCTION
values, then discard slots without freeing that owned metadata field. Under
#1139 I require exact allocation/free tracking and ownership review before
correcting that omission. Under #1140 I separately trace the clean first-class
retention and require fresh-provider reproduction before assigning a cause.

My first probe build refuses undeclared Darwin malloc types because the inherited
POSIX feature request hides them. I retain that refusal and request the installed
Darwin API with `_DARWIN_C_SOURCE`; I add no replacement typedef or warning
suppression. The corrected observation is `initial-allocator-observation.json.gz`.
Its exact source, build command, output and selected source/object hashes remain
retained, including the cached-dependency limitation.

I validate the same actual Mach-O executable again without rebuilding or changing
production. Its before/after image and input hashes match; the allocator control
passes, clean named batches again add0 and the first metadata batch again adds80000
bytes. Exact repeated observations, driver hash at validation and streams are in
`validated-allocator-observation.json.gz`. I do not backdate that driver check to
the initial build. Actual probe images remain under the receipt's local output
paths. Fresh full-provider sanitizer and teardown checks remain required.
