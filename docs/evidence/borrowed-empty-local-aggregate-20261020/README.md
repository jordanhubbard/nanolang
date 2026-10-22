# I qualify the aggregate borrowed-empty local slice

I integrate cb094e7e2 and its design/baseline ancestry at b160fde5a. All six aggregate methods pass in 38.645s through seed and both installed stages, including original and regenerated native execution, strict ISO C, separate original/generated shadows, swapped layout identity, exact128/estimated129 and atomic refusal. Generated shadow counts remain18/12.

I relink only the facts helper from current aggregate source and the recorded compatible objects; I retain exact before/after hashes and commands. All eight provider hashes remain unchanged throughout the test run (the planned facts binary replacement is separately recorded before the build). I retain every command/status/stdout/stderr JSON byte-for-byte in commands.json with hashes; other artifacts have hash/size records and remain in the named temporary directory.

This completes the mapped borrowed-plus-empty local acceptance locally. Full #999 closure and exact hosted Linux/macOS/sanitizer/coverage gates remain separate and open.
