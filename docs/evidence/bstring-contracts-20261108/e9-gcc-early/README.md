# My corrected GCC runtime component and remaining C boundaries

At `e9eb29d62606ea4939ddc61e2224d9367b4bb4a2`, run [42509577199](https://github.com/jordanhubbard/nanolang/actions/runs/42509577199) passes my sanitizer binary-string ownership executable, complete original VM suite (274654 assertions), original allocation-failure controls and three standalone private-heap controls (10318, 543 and 4445 assertions). The five focused VM methods pass; six numeric methods are also selected because the first harness imports their TestCase class. Those numeric methods remain separately selected in the unchanged full operand suite when I correct that harness import.

`actual-vm-products.json` matches actual hosted module bytes against their before/after hashes and successful build/verify/run commands. I retain all early text and file hashes after CRC-checking the full early artifact in memory. The full ZIP is not saved locally. I record and correct my first receipt attempt's mistaken assumption that every entry was UTF-8 text; no complete receipt came from that failed recorder.

The additional useful-function-shadow probe succeeds through NanoVM but fails C-seed creation on undefined bstr_new/bstr_length/bstr_free calls. The C seed correctly refuses publication after those shadow failures. The generated C extreme-substring native product builds successfully, then UBSan reports signed overflow for 1 + INT64_MAX and exits 1; the same VM probe verifies and exits 0. These are measured product boundaries, not infrastructure failures.

This early receipt is not a terminal full-job, actual native-product, fresh-stage, integration or release claim. The complete exact C/reference and VM baseline stdout/stderr/records remain in the compressed text evidence.
