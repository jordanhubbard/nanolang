I retain C-produced effect-handler parameter names under #976, with four executed metadata/roundtrip/stripping regressions and passing codec/allocation/codegen gates. Evidence: docs/evidence/effect-binding-names-20261011/README.md.

The same baseline identifies remaining required implementation: my self-hosted compiler rejects effect declarations, and canonical nvm2c refuses HANDLER_PUSH. I record both on the roadmap; the passing C producer metadata step does not close effects or the full release. I keep #976 open.
