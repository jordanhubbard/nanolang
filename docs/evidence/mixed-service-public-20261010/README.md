# My mixed public authority checkpoint

I extend `eec32b4c6` under #990 with explicit ordered per-instance grants,
public VM/generated-C execution and an installed C99 runtime package. My
[authority contract](../../SERVICE_MULTI_TRANSPORT.md#public-mixed-authority-and-installed-runtime)
distinguishes catalog-position policy from module-byte or endpoint policy.
Paired source lowering, source shadows, DNS/WebSocket and full 5.1 remain open.

## Qualification

I run the following on Darwin with Homebrew LLVM Clang:

```sh
PATH=/opt/homebrew/opt/llvm/bin:$PATH make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-services-public test-socket-public test-file-indirect-public-sanitize test-services-dispatch
```

I rerun only `test-services-public` after changing grant identity inspection to
copy the pointer representation before accessing the opaque grant fields.
I retain both terminal logs, command/status/output records and generated C/wire
artifacts. My source and installed-package hash manifests identify the inputs.
GCC 16 also accepts the public grant, ABI, emitter, VM and fixture with C11,
`-Wall -Wextra -Werror -fsyntax-only`.

My public matrix executes simultaneous File/TCP/File owners over IPv4 and IPv6,
direct and indirect helpers, permuted mappings, successful execution, assertion
failure and fuel limits zero and 45. I compare VM/native status, acquisition,
cleanup, scalar and host-resource counts. Success publishes 42 after cleanup;
failure preserves 999. Denied, missing, mismatched and individually or wholly
revoked grants report no acquisition, steps, files or sockets. Reentrant host
hooks require BUSY before invalid argument access. Close-error hooks close the
real resources before returning EIO and require cleanup failure without scalar
publication.

I also check copied policy storage, 64-entry creation, invalid policy/count,
wrong ABI/profile/runtime grant identity, repeated revocation/destruction,
allocation failure and truncated-input refusal. Native products exclude the VM
and emitter entrypoint symbols. Installed native consumers run the policy
matrix, while installed VM/emitter consumers execute the allowed program.
These consumers compile as C99 outside the checkout against installed headers
and archive only. The instrumented corpus uses ASan/UBSan and leak detection
for the service adapters, cores, dispatcher/emitter and generated code;
ordinary compiler/transport support objects are not fully instrumented.

## Retained development failures

My first package-edit script failed to parse, leaving no Make target. I fixed
the editing script before the package build. My first native fixture link
included an emitter-only BUSY probe and failed with an undefined emitter symbol.
I keep that probe in the VM/emitter fixture; generated consumers retain their
independent execution boundary. I retain the failed link log and the initial
passing public matrix separately from final qualification.

GitHub DNS resolution failed in this session for both API and SSH, and also
for an unrelated hostname. That initially prevented issue publication and remote
synchronization; it did not establish a GitHub service outage. A later API retry
succeeded and published [my coordination update](https://github.com/jordanhubbard/nanolang/issues/990#issuecomment-6096254517).

## Terminal results

My final public target passes in 59.131 seconds with 108 recorded commands.
The combined gate exits zero: initial public 56.410 seconds, TCP 51.271 seconds,
and both File methods 214.640 seconds; its final mixed dispatch result appears
in [the complete terminal](adjacent-final.log). I retain
[final public commands and generated artifacts](public-artifacts.tar.gz),
[adjacent artifacts](adjacent-artifacts.tar.gz), [source hashes](sources.sha256)
and [installed hashes](installed.sha256). My [public terminal](public-final.log)
records the final identity-check implementation.
