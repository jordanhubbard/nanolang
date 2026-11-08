# My first binary-string source and runtime observations

At b65af0567650deb9c7d1374b3f410027e577915c, both Linux checker components complete all 693 selected shadows without assertion failures. Recorded body sums are 8.825130 seconds with GCC and 8.455578 with Clang. Their actual checker creation/execution commands return zero. These early metadata uploads contain hash observations, not the actual executable bytes.

Pinned Apple run 42505432100 retains a checker creation failure: 693 shadows start, 692 complete without assertion failure in 9.370052 body seconds, then the unchanged ten-second deadline stops the final main shadow. The new check_bstring_type_contracts shadow completes in 0.271415 seconds before main begins; final entry assertions are not completed and no actual checker product is published. I keep this performance failure separate from a semantic assertion failure and require a corrected exact platform run.

All three jobs retain the unchanged whole test_bstring fixture through actual C-native and VM execution, with identical successful output and zero exits. The separate production probes reveal additional contracts. `baseline-output-matrix.json` retains every exact output, diagnostic, command and observed source/product hash:

| Probe | C-native observation | VM observation |
| --- | --- | --- |
| Binary bytes A/NUL/B | Length 3; bytes 65/0/66; visible C string length 1 | Execution exits 1 after successful assembly/verification |
| Unicode before explicit validation | Byte length 6; UTF-8 length/characters report -1 | Byte length 6; length 3; characters 233/33590 |
| Invalid UTF-8 | false; length -1; character -1 | false; length 1; character -1 |
| Negative/beyond/clipped substring | 0 / 0 / lo | 0 / 0 / lo |
| Negative/beyond byte index | 0 / 0 | -1 / -1 |
| Valid concat/copy ownership after freeing inputs | Aé茶! and Aé茶!kept survive | Same observed values |

The whole fixture calls validation before querying Unicode length; the separate Unicode probe queries first. I retain that state-dependent observation rather than assuming validation is implicit. Boundary failures are measurements, not accepted restored semantics. I have not changed VM/native BSTRING representation at this checkpoint.

Each complete early ZIP passes CRC. Per-compiler archive records retain artifact IDs, sizes and SHA-256; deterministic gzip retains all uploaded text and file hashes. Product hashes and magic values come from the observed actual commands; these uploads exclude the product bytes and do not independently establish native product custody, TAR modes or native SDK admission. Full artifact/product checks and later source/runtime/fresh/platform qualification remain open under #1114/#1065/#1081.
