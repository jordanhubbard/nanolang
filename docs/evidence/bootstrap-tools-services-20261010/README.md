# My service archive dependency fixture correction

Hosted coverage [job 114192223190](https://github.com/jordanhubbard/nanolang/actions/runs/38044859073/job/114192223190)
at `f5696fdc6` passes build and actual three-stage bootstrap, then fails
`test_real_make_links_only_changed_or_missing_tool`. I retain its cleaned log
and reproduce the same missing `file_indirect_public.h` refusal locally.

My synthetic tree intentionally contains timestamped object/tool stubs rather
than compiler sources. It omitted the newly linked Socket and mixed-service
archives, so Make traversed their real production prerequisites. I add those
archives to the existing timestamped/assumed-old fixture providers. I preserve
the no-op, changed shared object and each missing-tool assertions; I neither
remove production dependencies nor fabricate source headers.

The test-only correction leaves all 1,139 frozen mixed-bootstrap source inputs
unchanged. Full local dependency and later hosted acceptance are recorded below;
a local pass does not establish a green coverage job.

My focused fixture passes in 0.789 seconds. The complete
`make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-bootstrap-dependencies`
gate passes: five compiler-component methods in 48.647 seconds and all 21
dependency/message/bootstrap/tool methods in 20.325 seconds. I retain both
terminals and the original failing baseline. Hosted coverage must run again
on the corrected revision; I leave #982 open.
