# My native union padding evidence

I changed only the classifier's initial field vector for `AGG_VARIANT`:
absent payload slots start unknown instead of integer. Every supplied field
still receives its actual popped kind and shape. I retain conflicting present
field refusal and the existing runtime field checks.

Source checkpoint: `0c1c08a4067e18f82ceba58164e3d83f31494cb0`, based on main
`8182f95c`. MAC `task_a4b730306b84428da1f4e5683697353f`.

- `make -j8 test-nvm2c`: **2,422 passed, zero failed**; shape constraints
  **1,269 passed, zero failed**. Log `/tmp/nanolang-native-union-padding-native.log`.
- Two focused methods pass with four ordinary empty/data caller and return-join
  combinations plus unchanged present-field conflict refusal. Generated C uses
  strict warnings, ASan and UBSan. GCC run: **0.696 seconds**; Clang run:
  **0.903 seconds**, explicitly selecting installed GCC13 with
  `--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.
- The original unpinned Clang driver reports `-Wgcc-install-dir-libstdcxx` under
  `-Werror`. I retain `/tmp/nanolang-native-union-padding-clang.log`; I selected
  its toolchain explicitly without suppressing warnings. Pinned log:
  `/tmp/nanolang-native-union-padding-clang-pinned.log`.
- The retained ordinary source module with `Value.Data(int,string,bool,float)`
  and `Value.Empty` translates, links and runs under ASan/UBSan after this repair.
  Module SHA256: `98cc8a4aecd1d02a557bdcb7ce35e8665e21546df6892e40824cda8611d414d2`.
  Its earlier translation refusal is in `/tmp/nanolang-core-union-second-tests.log`.
- A checked resource-bearing union constructor also translates and executes
  normally under sanitizers. This does not admit source resource unions into
  the separate selfhost scalar union subset.

My initial new assembly harness used symbolic call/string operands where my
assembler requires indices. I corrected the harness before its passing runs;
that setup failure did not establish a product defect.

I do not infer types for absent values, admit conflicting present payloads,
or claim complete variant-sensitive native shape inference.
