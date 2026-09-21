# My trapped callable teardown

Direct/core invocation can return a trap while its frames still own callable
values. My vm_destroy previously released the operand stack and globals but
omitted these frame-owned edges. I now detach each surviving frame owner and
borrowed closure pointer, release its callable, and clear the handler count after
callback shutdown succeeds. Failed callback shutdown still returns before any
teardown mutation.

My regression uses an actual indirect call whose closure captures a managed
tuple. It checks the assertion trap, two surviving frames, the owned closure and
both heap objects, then verifies destruction clears frame/handler counts and
balances allocated/freed bytes with no live heap objects. I audited and repaired
the defect before running this control; I did not execute an unfixed variant.

The independent source checkpoint efc99628e applies the reviewed correction to
main81454fdc5 without the unfinished shared-capture work. Fresh make test-nanovm
passes on Linux GCC and Darwin Apple Clang:274642 VM assertions per host, zero
failures, plus existing substring, callback, heap and stack allocation controls.
Both runs preserve all981 source/driver input hashes and modes. Reports retain
compiler identities, exact commands, exit statuses, deadlines and process-group
cleanup. Raw Darwin reports are copied locally promptly; the remote tree lives
under /Users/jkh/nanolang-qualification/nanolang-vm-teardown-efc99628e.

This directory contains logs/manifests, not an archive of all compiled products.
Earlier qualification in the capture development branch retained a missing-test
package refusal and corrected passes; this independent current-main package
passes on its first execution on each host. The repair does not complete shared
capture integration or authorize my5.1 release.
