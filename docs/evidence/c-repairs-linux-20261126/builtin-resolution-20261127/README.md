# My C interpreter declaration lookup repair

I preserve pre-argument declaration lookup and contextual typing. After evaluating arguments I resolve declarations only where my selected dispatch consumes them: unknown/bstring, Builder, higher-order arrays and array_push. I change no Symbol or GCHeader layout.

My unchanged-C and repaired public evaluator fixtures pass the same13 cases. Six ordinary builtin calls each go from two evaluator declaration lookups to one; the two-call argument-order case goes from four to two with the same two side effects. Declared functions, Builder, push and actual bstring provider keep their values and lookup counts. These counted calls establish removed duplicate work, not a whole-compiler speedup.

My broader evaluator/STRING/index/scope/reference/GC controls pass. All13 cases pass ASan/UBSan with leak detection enabled. Removing both declaration-priority guards makes the declared Builder case fail under UBSan. Removing only one leaves the other effective; I retain that nondistinguishing control. I retain initial shared sanitizer-archive link refusals and an initial mutation observer that expected an assertion rather than the actual earlier UBSan failure.

My changed C seed compiles the original tests/selfhost_shadow_emitter.nano at the unchanged default10-second shadow deadline and outer300-second bound (44.823 seconds total). This local pass does not complete the17-step bootstrap,36 native tests,12 emitter tests, full cache/worker suites, both platforms or release gates. Both C repair issues stay open.

I retain fixture setup failures, the separately filed unchanged checker crash #1267, exact source snapshots, source/tool guards, full selected command streams and product identity metadata. I restore the ordinary shared archive after the isolated sanitizer build. Run `python3 verify.py --workspace /path/on/large/volume` to verify the receipt without executing a product.
