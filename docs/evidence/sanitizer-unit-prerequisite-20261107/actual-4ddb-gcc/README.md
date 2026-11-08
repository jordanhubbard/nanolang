# My first actual instrumented GCC checkpoint

Run 42451010848 at 4ddb46fe1d6c18134c03bce797ea308e6f566bb8 passes explicit sanitizer flags to Make and refuses providers without actual ASan symbols. GCC passes all 27 focused call/return/STRING/canonical methods and the unchanged module fixture. The original complete unit selection then stops on a strict format-truncation diagnostic in tests/nanovm/test_vm.c, test_array_remove_releases_the_element, formatting its three labels with snprintf into eight bytes.

I retain complete original build/module/unit/negative/interpreter and focused logs, unchanged provider checks, and verified exact source-tree identity. This is not complete unit acceptance. I preserve the original ownership assertions and replace only the three fixture labels with exact constant strings under #1046/#1053. Fresh actual corrected compilation/execution remains required.
