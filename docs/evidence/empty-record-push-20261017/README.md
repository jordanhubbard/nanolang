# I retain empty-record push qualification

I reproduced #994 before changing my type checker: the unchanged empty-record-field fixture failed in both my C-native and NanoVirt producers (`baseline.log`).

I now recover nominal identity from the first value pushed into a builtin empty literal and check subsequent pushes against that identity. I preserve declared functions named `array_push` and refuse mixed nominal records, including imported records with the same short name.

I pass the original fixture on both producers (`fixture.log`), all five expanded nominal-contract methods (`nominal.log`), and 222 Clang plus 222 GCC 16 strict compilation cases (`strict.log`) on macOS arm64. I include both regression targets in `test-quick` for hosted platform qualification.

I have not yet completed a fresh self-hosted bootstrap or hosted platform qualification for this change. These local checks do not establish the complete 5.1 release contract.

My first fresh seed build exposed a second inference boundary: before child checking, I recognized `List_Type_get` but not the compiler source's lowercase `list_Type_get`. I retain that seed refusal in `seed-getter-refusal.log`. I resolve both names through existing concrete record definitions, after declared-function lookup, and add an executed empty push from `List<First>` to my nominal controls. The corrected fixture, five methods and both strict compiler passes succeed (`list-getter-controls.log`, `list-getter-strict.log`). My fresh bootstrap retry remains in progress.

Independent review found that my newly exercised lowercase fallback inherited an allocated return name from the old uppercase path even though callers expect borrowed identity. I now return the registered struct name and free only the temporary lookup string. I explicitly interrupted the in-progress Stage 1 at `run-tkdveyx8` before changing this source; it is not a timeout or successful final-source bootstrap. My original fixture, five nominal methods, and strict Clang/GCC gates pass after the lifetime correction (`borrowed-identity-controls.log`). I started a fresh bootstrap for the corrected source.
