# My canonical string trim acceptance

I retain the exact string type of undeclared `str_trim` calls and emit STR_TRIM.
My native lowering tracks each allocated result and copies the selected bytes
before any later collection. I trim only space, tab, newline and carriage return;
vertical tab and form feed remain ordinary bytes. Native C retains my existing
NUL-terminated representation boundary. I do not claim embedded-NUL parity here.
My VM releases the input and reports allocation refusal before publishing output.

At source `412b2810`, fresh bootstrap passes both stages and installed execution.
The string-edge program and two additional paired VM/native methods pass, as do
normal and GCC/Clang O1 ASan/UBSan/leak allocation-recovery controls. Native
checks pass 2422 and shape checks pass 1269. The first broad run retained three
obsolete trim-refusal assertions; replacing them with still-unsupported uppercase
conversion preserves refusal coverage. Logs remain in
`/tmp/nanolang-string-builtins-native-gates.log` and
`/tmp/nanolang-string-builtins-corrected-native-gates.log`.

At integrated source `d9e35ab3`, fresh bootstrap and all 14 string/filesystem/
canonical-publication methods pass in 12.039s. The audio example then exposes
the separate direct-return classifier's missing trim case. I preserve
`/tmp/nanolang-string-audio-integration.log` and record that path in the roadmap.
Source `890e682d` adds the inline-return classification and an explicit direct
return regression. Its final integrated acceptance is pending; task602 and
allocation task88c3 remain open until the final checks and canonical merge.

Additional logs: `/tmp/nanolang-string-builtins-final-bootstrap.log`,
`/tmp/nanolang-string-trim-gcc-sanitizer.log`,
`/tmp/nanolang-string-trim-clang-sanitizer.log`,
`/tmp/nanolang-string-integrated-bootstrap.log`, and
`/tmp/nanolang-string-integrated-focused.log`.
