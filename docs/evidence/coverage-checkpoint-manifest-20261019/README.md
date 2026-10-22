# I retain bounded coverage checkpoint membership

I diagnose run 39261195595, job 118094674885: all seventeen coverage bootstrap steps pass, then packaging rejects the archive manifest at 32 MiB. The retained bootstrap receipt comes from artifact 12195604136. I retain the failing packaging tail and complete successful bootstrap receipt.

The exact source pin tracks 84,263 files. My representative-record measurement uses its actual path names, not the unavailable failed archive itself: the old duplicate tables take 37,200,653 bytes; version two takes 21,833,358 bytes. Generated provider records add to both sizes. I retain the original 32 MiB limit.

I encode tracked membership as sorted unique integer positions in sorted file records. Before encoding I require the original tracked hashes to match those records. On reading I reject invalid/duplicate/out-of-range indices, reconstruct the same path/hash identity map, then apply unchanged current-checkout, member, byte, link, mode and continuation checks. Version-one archives remain readable. I check the size before creating an archive, so an oversized writer fails before compressing provider files.

I pass 20 archive/workflow methods using `/private/tmp/nanolang-qualification-venv/bin/python -m unittest -v tests.test_ci_bootstrap_checkpoint tests.test_ci_workflow`. The first combined invocation used a Python without PyYAML; I retain that setup failure and use the existing qualification environment without changing assertions. The large manifest test retains every identity through encode/decode; small real archives qualify restore/identity and corruption handling. This is local qualification, not a successful hosted full checkpoint transfer. Hosted coverage and sanitizer continuation remain required.
