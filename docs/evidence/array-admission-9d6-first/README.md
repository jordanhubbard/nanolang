# My first corrected-fixture terminal

I retain the first Darwin `9d6118d87b876a969468d31f58a690e6685ad5c7` attempt. My ordinary checker, evaluator and selected codegen refusal controls passed. My sanitizer checker completed assertions but reported 2,138 leaked bytes in 22 allocations: a malformed new lexical fixture exposed missing token disposal on parser failure. My sanitizer evaluator/codegen and Linux configurations did not run on this pin.

I corrected the test helper cleanup and required successful parsing independently of semantic refusal in `0eb14852e`; the original production admission changes stayed unchanged. This seal preserves 47 report members, 661 local content-addressed objects and the first terminal. It does not turn that terminal into a pass.
