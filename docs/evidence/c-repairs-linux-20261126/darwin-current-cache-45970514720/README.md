# My current Darwin cache gate

I retain the successful Darwin cache job140409515953 from
[run45970514720](https://github.com/jordanhubbard/nanolang/actions/runs/45970514720),
source6e09c07a9f1651294e4669282cdc17ee14126775. The hosted Xcode27 worker
passes original storage admission before building fresh production tools.

My complete original cache target reports284 methods with26 existing skips:
four in the entry-point group and22 in source snapshots. The source-snapshot
group takes5653.148s. My three assembly-worker methods pass2.104s and my six
array/pointer regression methods pass102.065s. No deadline or assertion changes.

I retain exact GitHub artifact15191184169,33190196 ZIP bytes, SHA-256
`e7a2efe52260447c89d1b8eb515cc81ad31f5d8c03b756d02e4d648e78c43ca4`,
matching its API digest. My archive contains9 records/23 chunks/33385170 raw
bytes. Manifest SHA-256:
`ceb65fb29e15cc354eb969c49cad2a937181857328fe6b0308ed9359bcb2c2b5`.
The reader checks every archived byte, API source/job/step/artifact identity,
original method groups, exact Git source tree and before/after tool/image
guards. An altered-manifest control refuses before reconstruction.

```sh
python3 docs/evidence/c-repairs-linux-20261126/darwin-current-cache-45970514720/read_darwin_cache.py \
  --archive docs/evidence/c-repairs-linux-20261126/darwin-current-cache-45970514720/archive \
  --repo . --workspace /path/to/mounted/volume
```

This qualifies the named cache/worker/regression components. My separate
Darwin native bootstrap17 and native36 pass, but emitter setup still fails
at the original10-second shadow limit and discovers zero emitter methods.
My paired-platform/native-emitter, frozen CAST_STRING reconciliation,
complete pure translator, exact-candidate CI and all69 release gates remain
open where unverified. I do not tag or publish a release from this receipt.
