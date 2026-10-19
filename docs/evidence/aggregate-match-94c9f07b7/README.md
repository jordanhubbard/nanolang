# My match-integrated Darwin qualification

I qualify source commit `94c9f07b7` after merging the match-coverage component into the current 5.1 aggregate. My fresh bootstrap `run-_z6iugw8` passed all seed, Stage 1 and Stage 2 generation, verification, translation, native compilation and hello checks. Raw Stage 1 and Stage 2 modules are byte-identical; I retain the full source/tool manifest and provider hashes here. My installed compiler also compiled and executed hello with the C seed removed.

I ran all 15 match-coverage, inferred-effect and declared-purity methods through the C producer and both newly generated stages: all passed in 15.719 seconds, including the suite's VM and sanitized native controls. I explicitly supplied all three absolute producer paths via the respective metadata compiler environment variables.

This is Darwin evidence for the integrated metadata change. I retain the preceding complete 87-method scalar/binary64 and 19-method authority/effect/U8 receipt under `../aggregate-5f2f7c0b3`. Hosted Linux/Darwin qualification is still required on the final candidate. I have not completed the broader affine facts, source authority or reconstruction obligations, and this receipt does not authorize a release claim.
