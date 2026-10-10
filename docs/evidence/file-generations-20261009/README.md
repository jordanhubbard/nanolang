# I qualify both File compiler generations

I run a fresh bootstrap over the compiler sources atfea341eb0. All967 recorded
source inputs still match the manifest. Every bootstrap command succeeds,
including guarded VM self-compilation, module verification, native translation,
native compiler smoke checks and installed-compiler checks. The17 recorded
bootstrap steps take746.984 seconds in total.

Stage1 and Stage2 raw modules are identical:635,296 bytes, SHA256
`9f120d47fbe19bec124bf08c7418b2c1d8de19c31709ec15cdc81dddfff7ab96`.
All six host-library paths and byte hashes remain unchanged across generations.
I preserve the native-generation guard and complete source/tool manifest.
This is a local Darwin fixed point, not proof of compiler correctness or the
complete release-candidate Linux/Darwin gate.

I use the dedicated cache `/private/tmp/nl51-142704d0b`. These qualified local
compiler artifacts refer to that retained cache; I keep it available. This is
not a relocatable release installation. The prior shared-cache run failed exact
closure checks; I retain it in the preceding callable-source evidence. The
first longer dedicated cache failed combining file_product objects before seed
publication; I retain that terminal under `long-cache-failure`. I do not erase
those failures or relax closure comparison.

| Source qualification | Terminal result |
| --- | --- |
| Stage1 VM/native compiler forms, existing full corpus | 15 methods pass,349.765 seconds |
| Stage1 VM/native, new affine-result/multi-exclusive fixture | 1 method passes,33.715 seconds |
| Stage2 VM/native compiler forms, expanded full corpus | 16 methods pass,380.938 seconds |
| C seed/bytecode drivers, new owner fixture | 1 method passes,14.365 seconds |
| Workflow validation | 5 methods pass |

The prior [callable source batch](../file-callable-source-20261009/README.md)
retains the full C-driver and sanitizer suites. Here I add an affine OpenResult
callable parameter/result, two distinct exclusive borrows beside an owned File
argument, and a callable Boolean result. The complete nine-shadow owner fixture
publishes matching C/Nano bytes and executes in VM/native products with explicit
grants. The first fixture left owners live on ordinary error returns and was
correctly refused; I preserve that result and explicitly consume those owners
in the corrected fixture.

My platform CI now explicitly runs a real bootstrap, the full C-driver corpus,
and the full Nano-driver corpus for both generations. I keep a finite60-minute
step budget and preserve the prior suite's75-minute job budget by adding that
allowance. The File source tests use a10-second shadow deadline. Hosted success
of this new gate remains unverified; local evidence does not replace it.

`verification.json` records compiler and qualification-source hashes. Mixed
profiles, remaining File platform acceptance, public Socket/network/WebSocket,
other compiler/backend requirements and full5.1 publication remain open.
