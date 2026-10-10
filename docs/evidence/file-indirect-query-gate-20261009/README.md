# I run my indirect File foundations together

Under #989/#982 I add the missing `test-file-indirect-flow` Make target and
compose targets, ownership and hosted preparation as `test-file-indirect-queries`.
My platform CI invokes that batch and retains its command diagnostics on failure.
Previously my ownership test module required manually supplied environment
variables and was omitted from both existing indirect Make targets.

I qualify the batch on Darwin over production source at `e3b6548d1`, with the
Make/CI integration change. I use Homebrew LLVM Clang and run:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-file-indirect-queries
```

All six methods pass: two target methods in 3.220 seconds, two ownership
methods in 3.022 seconds, and two hosted-plan methods in 6.891 seconds.
The ownership fixture checks all 41 measured preparation allocation sites in
both failure modes. These durations exclude Make startup and dependency work.
I retain the [Make output](make.log) and [bounded command terminals](commands.json).
Every recorded child exits zero, is reaped, and leaves no process group.

I compile the query providers through the existing linked and allocator-hook
fixtures. These hooks are allocation-failure instrumentation, not sanitizer
coverage of every linked object. I also check workflow YAML parsing, the batch
command, diagnostic retention and `git diff --check`.

I have not run the changed workflow remotely: this session cannot resolve
GitHub. Linux, actual indirect runtime membership, VM/native dispatch, callable
arguments/results, source publication and the complete 5.1 gates remain open.
These nonexecuting plans grant no runtime admission.
