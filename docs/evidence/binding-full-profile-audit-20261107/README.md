# Full Make/private/initial-binding profile audit

I pass [run 42403291622](https://github.com/jordanhubbard/nanolang/actions/runs/42403291622) at exact source `b42ea96e951d06919b30f86a798e1c7630dd720d`. GCC and Clang each pass 1,579 ordinary/instrumented compiles. Both original eight-test binding behavior jobs also pass. My [summary](summary.json), [hosted run](hosted-run.json), and compiler directories retain full compressed command/results, raw report hashes, actual selected tool identities, resource observations and logs.

I generate real managed-runtime IR and all three binding corpus headers privately, retain coverage instrumentation with writable outputs, and preserve Fortify level two with ordinary/ASan/UBSan diagnostic modes. My worker observations identify four CPUs, 16 GiB total RAM, about 79 GiB free disk and two compiler workers per compiler job.

GCC takes 1,236.749 seconds and Clang 872.076 seconds. GCC alone exceeds 1,200 seconds; their serialized sum is 2,108.825 seconds. I do not qualify my combined default runtime budget from this measurement. The exact report still lists 235 unreviewed test C files. The two subsequent binding neighbor fixtures have separate actual 114-profile/eight-native-test acceptance at `55b552dd2`, retained under `docs/evidence/binding-compile-profiles-20261107/`; they are not retroactively counted in this report.

I keep #1046 open for remaining Python-built profiles and runtime, plus main integration, native adoption and all remaining 5.1 requirements. This broad pass does not substitute for PR #1068's full native/qualification jobs.
