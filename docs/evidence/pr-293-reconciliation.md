# PR #293 reconciliation

I reviewed both incoming commits through
`c751d2e80ca5fdc840bff13f07e117a9b1f10ec8`: 11 files covering the schema,
generated AST layouts, parser, checker, native emitter, NanoISA subset guards,
and positive/negative self-hosted fixtures. I preserve the head as a merge
parent on my integration branch. Its CI was still running during review;
the quota-limited review comment was not approval.

The AST now carries the callee's node kind alongside its index. I retain
newer integration behavior and repair a unary-parser call site absent from
the incoming branch. I guard computed print arguments from identifier-table
lookup, distinguish function arrows from generic closing brackets when
splitting signatures, and register nested function-pointer typedefs before
their users. I strengthen the incoming positive fixture's placeholder
shadows with observable assertions.

## Verification on Darwin, 2026-09-15

`make test-selfhost-returned-calls` rebuilds both compiler stages and passes
their smoke/no-C-seed checks, then passes three explicit Python methods:

- the incoming positive program executes nested calls and prints exactly
  `callee\nargument\n`; wrong-type and wrong-arity fixtures return failure,
  report the function-expression diagnostic, and publish no artifact;
- a returned function accepting another function compiles, runs its shadows,
  and prints its integer result as `42\n`;
- a two-argument computed call prints exactly `callee\nfirst\nsecond\n`,
  checking evaluation order and once-only evaluation.

Log: `/tmp/nanolang-pr293-final.log`, three methods passed in 8.327 seconds.
`scripts/check_compiler_schema.sh` also passes.

The first serial fixture run exposed the nested native typedef failure;
`/tmp/nanolang-pr293-fixtures.log` records it. Two earlier overlapping
bootstrap processes were terminated after I detected my orchestration
mistake; neither counts as verification. I moved my stale stage-two/three
build stamps to Trash and rebuilt serially. They are recoverable build state,
not source changes.

The original `make test-selfhost-map-results` now gets past returned-call
parsing. It still fails: int-to-int passes, while the other 15 scalar pairs
fail native compilation because generated `nl_map` takes
`int64_t (*fn)(int64_t)`. I retain the entire matrix and the open map task.
Log: `/tmp/nanolang-pr293-map.log`, 15 failing subcases in 25.803 seconds.

Self-hosted NanoISA computed calls remain explicitly unsupported. These
native fixtures do not establish all closure, aggregate, resource or backend
semantics. The general shell suite's negative tests also need stricter
failure classification (`task_4f84d7b8485a467da3909f79e2417233`); I do not use
that runner as evidence for these rejections.

While I published this integration, PR #293 was separately merged into main
as `ce0e8e9555f98604636cd62954d1c63ca2a9521c` at 18:53:23 UTC. That merge
contains the worker branch, not my additional integration fixes above. I must
still reconcile the advanced main branch before the release. The source task is
`task_256337a9977f43b2baee7b26ebd66bc7`; scalar map parity remains
`task_75b340982b6cf797f29b38c1a188aab3`. This is not release acceptance.
