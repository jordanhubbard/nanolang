# My remaining sanitizer shadow timeout

CI run 37964483577 at 80a472c85 moves past the earlier Docs/Concurrency and strict
example failures, but Memory Sanitizers job 113935278506 reaches its finite
300-second limit during nanoisa_emitter shadows. My uploaded component log
retains 651 completed shadows, zero reported assertion failures and 291.508265
seconds inside completed shadows. Shadow 652, nisa_emit_function, starts and is
interrupted. I do not count this as a passed sanitizer gate.

The complete component log and timing summary are adjacent to this file. The
slowest completed parser block shadows take 28.733772 and 28.582326 seconds.
Other build and coverage jobs were still running when I collected this result.
The measured ordinary 120-second / instrumented 300-second CI adjustment has not
completed qualification; #982 remains open. I neither skip these shadows nor
increase the deadline again without a measured repair decision.
