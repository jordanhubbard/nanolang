# Original-pin assembler diagnostic

I reran the original `xassembler` helper across all placements, external
selections and shared choices at clean `f0f6a0c62b9a3fb53251f987a492f3de9f7ab514`.
It passed in 217.945 seconds with unchanged source and probe hash. This includes
the earlier failing combination; it neither identifies its cause nor replaces
complete source-snapshot acceptance. I retain the original failure and keep
GitHub issue #980 open while the corrected full corpus runs.
