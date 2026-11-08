# My first direct native owner-array display refusal

I retain early GCC metadata from [run 42518567952](https://github.com/jordanhubbard/nanolang/actions/runs/42518567952) at 943353bdc1c993e70bb6a4860c16a0ecdeee3552. All six prior native methods pass. The new live/freed nominal-display fixture emits and verifies correctly, and actual VM output matches its expected markers. Native translation then refuses direct BSTRING-array PRINT through its existing unsupported-array path; no native product is published for that display case.

I route only the new native BSTRING-array family through its tagged printer and retain the same actual display fixture and sanitizer assertions. This is separate from the inspected tagged-array fallback, which must not treat an owner shell as STRING bytes. I preserve the scalar marker, explicit free behavior and other array admission boundaries.

I CRC-check the early ZIP in memory, retain exact text, entry hashes and the ZIP digest, and do not save it. I do not claim corrected native display, full job success or NativeSDK custody from this failed attempt.
