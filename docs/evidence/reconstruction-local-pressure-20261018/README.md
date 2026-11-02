# My reconstruction local-pressure correction

I preserve the native product's 1024-local bound and every large carry/division fixture. I avoid materializing immutable INT/BOOL/binary64 literals and fold only adjacent temporary-to-local stores whose temporary has exactly one use across all structured regions. Shared snapshots and computed operand ordering remain explicit. U8 literals retain their typed bindings because a bare Nano numeral is INT.

I retain the original complete-corpus failure and the intermediate 87-method failure. The latter includes two source-shape assertions tied to redundant temporaries, two already tracked computed-U8 refusal subcases (#1004), and the genuine U8 literal typing regression found during optimization. I correct the assertions to require one actual call and the high-word operation, preserve exact byte typing, and pass all three focused regression methods (8.325 seconds).

My final scoped run excludes exactly `tests.test_reconstructed_u8.U8Reconstruction.test_contextual_source_literal_refusals_preserve_output`, whose separate implementation and fixture repair belong to #1004. I do not remove or skip that method in the repository, and I do not call this the complete release gate. The full gate must run after #1004 integration.

My local providers are recorded in `tested-inputs.json`; the installed stages are retained from the earlier 55da02bf9 bootstrap, not freshly bootstrapped from this revision. Fresh integrated compiler and hosted platform qualification remain required.

My final scoped regression passes all 86 methods in 412.667 seconds, including the unchanged large carry/division cases and all binary64 cases. The excluded #1004 method and full integrated gate remain open.
