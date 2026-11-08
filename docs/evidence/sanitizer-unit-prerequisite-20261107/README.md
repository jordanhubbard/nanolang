# My instrumented unit prerequisite failure

In job 128517156997 of run 42397278342, my canonical instrumented bootstrap continuation and both generation verifications succeed. My following unit step passes the original 3018-assertion NanoISA C suite, then enters the umbrella build prerequisite of test-nanoisa-module. That target itself compiles modules/nanoisa/mvp.nano with bin/nanoc_c; its recipe does not consume the legacy component binaries.

The umbrella build enters .stage2.built: parser and typecheck compile, then nanoisa_emitter reaches its unchanged 3600-second compile alarm. The unit corpus does not complete and negative checks are skipped. I retain the complete original job log, component logs, bootstrap logs and manifest. This is a prerequisite timeout; I do not infer a sanitizer violation or an infrastructure cause.

I replace only this module target's umbrella prerequisite with its actual schema, module-index and C-seed inputs. My module recipe, complete unit/negative corpus and instrumented provider settings remain. Legacy component/build acceptance remains a separate release obligation; this correction does not remove it from normal build. Full exact integration checks and blocking sanitizer acceptance remain open under #1053.
