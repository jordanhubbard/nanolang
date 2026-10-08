# My compiler phase qualification

I preserve phase ordinals 0 through 4, rename ordinal 3 to NanoISA and append
backend at ordinal 5. C emission and host compiler failures use backend.
I regenerate the C and NanoLang schema and update the historical callers.

I built my C seed in `/private/tmp/nanolang-phase-tools` to avoid replacing
tools used by the outstanding source-snapshot suite. I ran:

```sh
PATH=/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:$PATH NANOLANG_PHASE_CSEED=/private/tmp/nanolang-phase-tools/bin/nanoc_c python3 -m unittest tests.test_compiler_phases tests.test_nanoisa_schema
```

All 36 methods pass. The phase controls compare every generated schema output,
execute diagnostic constructors in VM and strict native C, execute historical
formatter source with its adjacent shadows, and force an actual C compiler
failure to check backend JSON phase 5. My first fixture used the wrong schema
key; I retain that failure and its corrected rerun.

My larger existing diagnostic corpus passes C-seed bytecode and Stage2 VM,
but its self-hosted native translation fails at AGG_PACK field 1 in function 16.
I reproduce the same failure at unchanged 17347f8b4 and retain both modules and
logs. Task `task_f61d318c7c8640e58f47ffccd1abe4f4` keeps that separate native
record-flow defect open; these phase checks do not establish full diagnostic
backend equivalence or release readiness.
