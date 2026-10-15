# I restore complete Darwin example compilation

I export the GLUT and SDL_image API constants and qualify SDL_image references in four examples to avoid global-name collisions. I select module compilers in this order: NANO_CC, explicit module c_compiler, CC, cc. This preserves Bullet's C++ driver under CI's LLVM CC without disabling the explicit NANO_CC override.

`CC=/opt/homebrew/opt/llvm/bin/clang NANO_VM_EXAMPLE_SHADOW_TIMEOUT_SECONDS=60 bash tests/test_vm_examples_coverage.sh` exits 0: all 244 eligible examples compile, the same four excluded sources still fail both producer checks, and nesting/builtin execution controls pass. I change no exclusions. After trimming inherited trailing whitespace on changed lines, both affected examples compile and verify again. Compilation and selected shadows do not establish graphical rendering or interactive physics behavior.

Three integrated compiler-selection tests pass: C++ language/header invalidation under CC, NANO_CC precedence over explicit/default drivers, and changed compiler identity across NANO_CC/CC/PATH/same-path controls. The old C++ regression fails with a missing cstdint header because CC selects C mode; the corrected candidate passes. All eleven original failures also compile and verify through the isolated candidate under LLVM CC.

My first isolated copy reports an SDL_image artifact-publication failure; I retain it without attributing a root cause. A fresh isolated cache and the integrated full gate pass. I keep the retained failure separate from the constant/import/compiler-selection defects.

The prior introspection compiler batch is independently qualified at 119683b80 in ../introspection-bootstrap-119683b80. Linux, main CI, the full 5.1 scope and publication remain open.
