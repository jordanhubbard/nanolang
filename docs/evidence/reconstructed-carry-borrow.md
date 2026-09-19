# My carry and borrow reconstruction evidence

I qualify the bounded current-main port of `task_cbdce24cc6b747f6b39ceb4c87f20676` and the fresh acceptance required by `task_1009525724234d9ca7df3f8284e75943`. I preserve the historical signal-11 evidence without executing its compiler or generated inputs.

## My source and tools

I started from canonical `41e955996422fe3ea353d95ac609c8b675a7abc1`. Production checkpoint `008fdb8f1c9e9f1b884dc44025460fa3596df4b6` adds the carry/borrow reconstruction and its three-method fixture. Test-only checkpoint `d80847e8d464e8f3e2e82c20d825ba9a89b962bc` corrects two obsolete generic-`ADD` refusal expectations. The checkout was clean before and after every passing gate.

My selected source hashes at the final test checkpoint were:

```text
e546caa28ac467c0e856bb5d0fd356ac3802852c76a47cbed6fc1f7ce3cbafb4  scripts/nanoisa_reconstruction.py
efb264bdf5c9a606c3e4c5f4c90a2bcb9aa94507d557aa87abfb7f710a09b601  tests/test_reconstructed_carry_borrow.py
2d30094412e1e9463c3a4a2396b878ac26544eb791da13fe780c62cafa1a1f87  tests/test_scalar_reconstruction.py
071c0efbe0015cfcd37f56406a7685ecf328e56ff517b92c49feb22df915f2a1  tests/test_reconstructed_integer_addition.py
```

My fresh build selected these artifacts:

```text
c2e67d240f1c0028957d982ac2329b8ee51714d91f63c587a8a29113a8717ea4  bin/nvm2hl
57b1056b41f758ba75dc12246f646b03d3af628fb709707816c9f0b5fd97b1ec  bin/nanoisa
fd682ddfc0038da6cfb85bc2697b9afdf4f9234c14f76afe01a62f897779bcdd  bin/nano_vm
d1de2b83134bf55a8576be9ab7715c2c558382f173e571e5bfae5a2ad03f1fb6  bin/nvm2c
88e417eabada6be36421ff919ff2d9f410ebbb6132ba110af92d07e3b6a9efd8  bin/nanoc_c
435b131b5197c0fedb81ebdaabe19983c836f2d69bcd907566f7951e44933560  bin/nanoc_stage1
744a12ad67a1779302332a22924db8ffd995a04265392d2e4652de7ea74ccbc3  bin/nanoc_stage2
744a12ad67a1779302332a22924db8ffd995a04265392d2e4652de7ea74ccbc3  bin/nanoc
```

I used macOS 26.6.2 on arm64. My default compiler was Apple Clang 21.0.0 at the Xcode-resolved path, SHA-256 `1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`. My focused compiler controls used Homebrew GCC 16.2.0, SHA-256 `3f088afcf8c0e60aacb6a8a11d1e2d987cf3eddd8bfcd7d7fa3ded93b1a7b49e`, and Homebrew Clang 23.1.1, SHA-256 `570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`. Python was 3.14.6 and GNU Make was 3.81.

## My results

I built `nvm2hl`, the assembler/disassembler, NanoVM, the native translator/runtime and a fresh three-stage compiler with:

```text
make -j8 nvm2hl nanoisa_dump nano_vm nvm2c nvm2c-runtime bootstrap
```

The command passed in 347.41s. Stage 1, Stage 2, installed hello and installed C-seed-independence smokes passed. The native Stage 1 and Stage 2 binaries differed, and the build correctly reported that this is not a fixed-point proof. I retained the terminal result in the execution session but did not create a separate raw bootstrap log; I do not describe that terminal as a sealed file artifact.

The first complete scalar run passed 53 of 55 methods in 603.601s. Its only failures were two legacy controls expecting generic `ADD` to refuse after generic arithmetic support had merged. I retained `/private/tmp/nanolang-carry-current-scalar.log`, SHA-256 `3dd2bd6b483808ac2f457c7b9b2ecfc1aa41dfd3f277f145a960b6b506902fd9`, and recorded `task_a74f7ac27d25ae6c8eebacc12179c9cb` before correction.

The two corrected refusal methods passed in 1.578s. The complete 55-method scalar reconstruction suite then passed in 638.424s. Its raw log is `/private/tmp/nanolang-carry-current-scalar-corrected.log`, SHA-256 `af82a62293ce1c41457426e5eaf8cec8972e57fe8ba4128ff3b6318e122016af`.

The three carry/borrow methods also passed with both additional compilers:

- Homebrew GCC 16.2.0: 3/3 in 270.405s; `/private/tmp/nanolang-carry-current-gcc16.log`, SHA-256 `982446ce9301dfb15d20c2772c2630cc15ea4fd86a532bc67f7571af03218a17`.
- Homebrew Clang 23.1.1 with `ASAN_OPTIONS=detect_leaks=1`: 3/3 in 102.069s; `/private/tmp/nanolang-carry-current-clang23.log`, SHA-256 `adb3f361865b62ded68d6ad1341637397a87cdf7d8d53d370713abc8ed7b7340`.

These gates cover endpoint and noncanonical carry bits, both result positions, pure loop use, call/local snapshots, exact tag/arity refusal, canonical assembly roundtrip, VM execution, strict sanitized reconstructed C and all three freshly built NanoLang compiler paths. They qualify this bounded reconstruction child. They do not qualify full product PR522, the active owner-ARRAY source lane, the full reconstruction parent or release publication.

## My canonical integration

While these gates ran, canonical main advanced from `41e955996422fe3ea353d95ac609c8b675a7abc1` to `a52d990d7a0883ec5692be0f03872e0aaa1d54d9`. I merged that exact main into this branch at `ea3a519d87b557fad3dc7b0b67e2e219081dd194`. The reconstruction production and three affected test files retained the hashes listed above. Current main changed compiler, VM, Makefile and roadmap inputs, so I removed all build products and performed a new integrated qualification instead of relabeling the earlier results.

The clean integrated build passed in 345.95s. Stage 1, Stage 2, installed hello and installed C-seed-independence smokes passed; native Stage 1 and Stage 2 remained intentionally non-identical and are not claimed as a fixed point. The complete raw log is `/private/tmp/nanolang-carry-integrated-bootstrap.log`, SHA-256 `b9fcd697abbaa6643c97dea95bb0401ded22494a1999c3f409bb6d3a5f046500`.

The integrated carry methods passed 3/3 in 68.229s. The raw log is `/private/tmp/nanolang-carry-integrated-focused.log`, SHA-256 `25244d38172eb4e674a726c0ab39913844a1e6f3f93f0c7d082ee468177ff761`. The complete integrated scalar reconstruction suite passed 55/55 in 598.708s. Its raw log is `/private/tmp/nanolang-carry-integrated-scalar.log`, SHA-256 `2d4643a394300bf741730e4741dc54af2839f4fdf538e70e60ff4d534384ac94`.

My integrated tool hashes were:

```text
c2e67d240f1c0028957d982ac2329b8ee51714d91f63c587a8a29113a8717ea4  bin/nvm2hl
bc101a657d6847706c0539f902d3bca80c443011d40ad56692af6b6e5e1bd005  bin/nanoisa
0d3ebf1353d43d81f48ee8cdf47df8fa374478b5be186ce595a7715cda8f9359  bin/nano_vm
e2a95f05c6a419be92ab10d7bdc19ae27d40c1a1c2251c569d152524bb6a8aca  bin/nvm2c
8b3b7f76e2e27e56f6c0f5f04a643c1d189156fb7b70746ee4435bf0dfaa44a0  bin/nanoc_c
57fce0bd00e680de223cf1ccd1064e134ae4f22181a1c22276a8383e37502372  bin/nanoc_stage1
6ee473316831478acf8fc0a16978a57c3b0e6592863a25aac256ed2fda4f4ee5  bin/nanoc_stage2
6ee473316831478acf8fc0a16978a57c3b0e6592863a25aac256ed2fda4f4ee5  bin/nanoc
```

This integrated rerun includes the now-merged owner-ARRAY runtime/source work, but it remains a scalar reconstruction qualification rather than a full product or release gate.
