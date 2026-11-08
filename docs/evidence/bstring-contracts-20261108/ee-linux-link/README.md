# My first Linux interpreter-harness link boundary

At `ee286d683f2834d7179b208d89ae0804beea2e0a`, both Linux compilers pass the original VM and raw C string sanitizer controls, then fail to link the standalone interpreted dispatcher harness. The inherited -rdynamic flag retains unused environment functions, which reference builtin/service providers outside this harness's deliberately small closure. The interpreted dispatcher does not execute on these attempts; later source-method steps are skipped. I retain exact early artifact text and CRC/SHA metadata from run [42512286440](https://github.com/jordanhubbard/nanolang/actions/runs/42512286440).

I filter the host-export flag only for the standalone harness, preserving compiler/VM exports and every runtime assertion. Corrected hosted execution remains required. I do not classify these link failures as infrastructure.
