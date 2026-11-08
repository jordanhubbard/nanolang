# My terminal a98 original-route checkpoint

Run [42506595409](https://github.com/jordanhubbard/nanolang/actions/runs/42506595409) at a98c6e23bd876d4bfe3da85d4fffefc87da3c1c8 is terminal with failure in the complete original corpus step. I retain every step conclusion and the exact four route XML files.

| Route | Selected | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: |
| seed native | 237 | 237 | 0 | 0 |
| seed VM | 237 | 237 | 0 | 0 |
| fresh Stage 1 | 237 | 213 | 24 | 0 |
| fresh Stage 2 | 237 | 213 | 24 | 0 |

I verify every ordered name/class pair against all four 9c selections. Relative to 9c's 207/237 per fresh stage, both stages clear the same six failures: test_all_imports, test_bytes_conversion, test_env, test_setenv_shadow, test_std_modules_simple and test_unsafe_basic. No new failure appears. I retain all twenty-four remaining failure diagnostics in route-results.json; binary strings still fail with unsupported local type bstring.

I CRC-check the complete 57354162-byte hosted artifact 13631010627 in memory and record its SHA-256 and file count. I retain exact XML only, without another local full ZIP. This receipt does not independently verify executable bytes, other artifact entries or embedded TAR/SDK custody. It does not qualify the complete original corpus, original budgets, later sources, platform integration or release.
