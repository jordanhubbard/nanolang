# I retain empty-record push qualification

I reproduced #994 before changing my type checker: the unchanged empty-record-field fixture failed in both my C-native and NanoVirt producers (`baseline.log`).

I now recover nominal identity from the first value pushed into a builtin empty literal and check subsequent pushes against that identity. I preserve declared functions named `array_push` and refuse mixed nominal records, including imported records with the same short name.

I pass the original fixture on both producers (`fixture.log`), all five expanded nominal-contract methods (`nominal.log`), and 222 Clang plus 222 GCC 16 strict compilation cases (`strict.log`) on macOS arm64. I include both regression targets in `test-quick` for hosted platform qualification.

I have not yet completed a fresh self-hosted bootstrap or hosted platform qualification for this change. These local checks do not establish the complete 5.1 release contract.
