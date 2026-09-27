# Focused resource classification qualification

I qualified source `54e39da36` with a private reviewed evaluator selector that executes exactly five resource shadow bodies in their original imported module: original and additive `resource_classify`, original `resource_concrete_payload_fact`, `resource_type_is_declared`, and `resource_instantiated_classify`. I retain full dependency checking, all 20 module initialization passes, exact imported source context and the original 60-second aggregate shadow deadline. Unrelated shadow bodies do not execute; this is not complete preparation acceptance.

The actual focused compiler invocation passed in 76.734 seconds including checking/emission. All four preparation/test phases exited zero, with every owned group gone and no outer timeout. The selector compiler retains the original 159 provider objects plus one rebuilt evaluator object; its undefined symbols include ASan and UBSan. I retain the original preparation environment, including `ASAN_OPTIONS=detect_leaks=0`; this does not establish LSan acceptance.

| Selected shadow | Body seconds |
| --- | ---: |
| resource_concrete_payload_fact | 4.905639056 |
| original resource_classify | 1.188868553 |
| additive resource_classify | 0.314551836 |
| resource_type_is_declared | 3.419458948 |
| resource_instantiated_classify | 5.392239265 |

The earlier original full-selection run measured 4.920267571 seconds for concrete payload and 8.563604432 seconds for declaration classification. The latter improved by about 5.14 seconds in this focused configuration. I do not infer full-workload acceptance: that earlier run stopped before 164 remaining checker shadows, and this selector omits preceding shadow side effects.

My bundle retains 59 reports with exact bytes; 363 CAS objects retain providers and products, including all 1,584 generated native files and their modes. Complete source/tool maps match before and after each phase. The original failure remains sealed at c5084b874. Private selector and driver sources, actual commands, symbol output and exact selected source/item/name records are in this bundle. No unchanged failing full suite was replayed.

## Diagnostic object flag attribution

My private evaluator selector object used the recorded global CFLAGS and ASan/UBSan flags. The inherited main-object build closure omitted `eval.o` target-specific `-ffp-contract=off -fno-fast-math` from Makefile.gnu:5477. I retain that actual command and limit this evidence accordingly: it is not exact original evaluator Make-flag parity. No binary64 semantic defect is demonstrated by this omission; resource-classification measurements and selected assertions remain observed. My corrected full-component gate uses the untouched original Cseed/capture products, not this selector compiler. I do not replay the focused gate solely to erase this attribution limitation.
