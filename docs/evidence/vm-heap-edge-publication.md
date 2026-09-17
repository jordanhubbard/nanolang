# Heap edge publication before release

I replace a heap edge before releasing its former value. Release can synchronously invoke cycle collection, so the collector must observe the updated graph. I preserve the existing ownership transfer at VM stores, array and record updates, and captured bindings. Map replacement retains the incoming value before publishing it; map deletion detaches both slots before either release. Array removal shifts and clears the vacated slot before releasing the removed element.

I checked this change against base `ae4aa585` with `make test-nanovm`: 272579 checks passed, followed by passing field-allocation and stack-reserve failure/recovery checks. The retained log is `/tmp/nanolang-vm-mutation-unit.log`.

I also checked the existing complete compiler shadow module. Its assertion still fails after 3.22 seconds; that result is retained in `/tmp/nanolang-full-vm-shadows-mutation-fixed.log`. I do not claim this repair resolves that separate blocker or completes the NanoISA-only compiler cutover.

I track this bounded repair as `task_493552bb4a7144188299474314bde470` and the compiler-shadow assertion as `task_fdf43892a1104b1facddc2553af390af`.
