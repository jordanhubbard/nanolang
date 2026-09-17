# AOT sanitizer build isolation

I no longer rely on changing `CC` to invalidate an existing make object cache.
My sanitizer driver gives each invocation a fresh temporary object directory,
binary directory and test-harness path. It passes ASan/UBSan flags to the
compiler, disables sanitizer recovery, propagates build/test failures, and
checks undefined symbols in both translator and shape objects for ASan and
UBSan instrumentation before reporting success. I remove these privately owned
outputs when the driver finishes.

I retain the normal output paths for ordinary builds. The generated-C test
harness inherits the instrumented compiler command, so its generated test
programs also use these flags. Leak detection defaults off because generated
helpers retain some allocations until process exit; this is not a leak-freedom
claim.

## Verification

On macOS, with existing normal-build objects and `bin/nvm2c` present,
`make -j1 test-nvm2c-sanitizers` exits successfully: 1,048 AOT checks and 952
shape checks pass, followed by successful object-symbol verification. The
SHA-256 digests and modification times of the normal translator binary,
translator object and shape object are identical before and after this run:

| Artifact | SHA-256 | Modification time (epoch seconds) |
| --- | --- | --- |
| `bin/nvm2c` | `416e81904821effc6891cb6545832148237d866f4c49a0b38eef5ce29fc86886` | 1789517329 |
| `obj/nanoisa/nvm2c.o` | `626db8f0fe538173f9f60aa0e2905db790ed4d84726ff134db1666bffe3e9171` | 1789517177 |
| `obj/nanoisa/nvm2c_shape.o` | `b159455d897d4cf977afa2a909c8ca560437971ddf2e8e9a71068adcb92088c9` | 1789516796 |

My three Python driver tests check fresh output roots across repeated runs,
cleanup, build-failure propagation, and rejection of missing sanitizer symbols.
These are orchestration tests; the real warm-cache run above supplies actual
compiler and runtime evidence. This change does not complete nested aggregate
emission or full compiler acceptance.

A subsequent ordinary `make -j1 test-nvm2c` also passes 1,048 AOT checks and
952 shape checks. `git diff --check` passes.

MAC task: `task_f3df199b025042e0b1d83484cd104ed3`. Claiming currently fails with
`agent_status_unavailable`; I record repository evidence without inventing a
successful ownership transition.
