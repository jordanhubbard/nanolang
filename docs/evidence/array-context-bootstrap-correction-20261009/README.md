# My contextual string-array bootstrap correction

My first fresh installed gate at 66bad5d59 stops at Stage1 after 94.399 seconds.
The before/after source inventory is identical and the user guide file is
unchanged. The new contextual literal checker refuses the compiler's
`[(str_trim (str_substring spelling 0 opening))]`: my builtin table has no
`str_trim` result type. The retained reduction independently returns TYPECHECK.
I add its actual string result type and preserve the element-validation rules.
Corrected component and installed-product results must be recorded separately.

I also retain the terminal old-stage slice diagnostic: four methods run for
913.213 seconds, with two byte-literal checking failures and the original
900-second GNU leak-enabled runtime timeout. Other controls pass. That run used
old installed stages and a corrected native translator; it is not a clean
immutable-source qualification and does not validate the newer checker.

My rebuilt checked component passes all five methods in 3.747 seconds, including
the trim reduction, all original byte/nested/slice contexts and twenty refusals.
Positive cases pass verified VM and LLVM ASan/UBSan with detect_leaks=1. This
does not establish fresh installed-stage acceptance.
