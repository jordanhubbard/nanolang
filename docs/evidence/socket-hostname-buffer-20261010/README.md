# My hostname and buffer Socket implementation

I add `nl_socket_resolve_tcp`, `nl_socket_send` and `nl_socket_receive` to my
private owned Socket adapter under #990. I retain numeric connect, generation,
rights, transfer and exactly-once close behavior. My byte APIs now share the
buffer implementation.

I accept counted canonical IPv4/IPv6 literals without resolver permission.
Hostnames require an explicit trusted-host `allow_lookup` argument. I copy and
deduplicate at most 16 endpoints, preserve IPv6 scope returned by the resolver,
refuse overflow without publishing a partial list, and separate `EAI_*` errors
from `errno`. I retain no resolver-owned memory or caller string. My resolver
call is synchronous; this interface does not establish a deadline or a public
DNS capability. The POSIX error and allocation contract is documented at
<https://pubs.opengroup.org/onlinepubs/9799919799/functions/getaddrinfo.html>.

Each buffer operation validates the live owner and required rights, admits at
most 65,536 bytes, and performs at most one nonblocking host I/O call. I report
partial progress, distinguish zero-length operations from EOF, preserve the
byte API's zero-on-EOF behavior, and refuse token/buffer overlap.

My initial Darwin tests found that `inet_pton` accepts leading-zero IPv4 and
scope suffixes outside this interface's contract. I now validate canonical
IPv4 and permitted IPv6 characters before conversion. Those inputs remain in
my regression suite.

## Executed checks

- `clang-network.log`: final included-source and separately linked tests pass:
  counted inputs, lookup refusal, numeric bypass, IPv4/IPv6 copied results,
  duplicate suppression, malformed/oversized lists, resolver errors and release,
  output preservation, actual localhost resolution and TCP traffic, partial
  buffer transfers, NULs, EOF, rights and stale tokens.
- `sanitizer-network.log`: the same final tests pass with ASan, UBSan and leak
  detection enabled, including the later scoped-IPv6 resolver controls.
- `clang-adjacency.log` and `sanitizer-adjacency.log`: existing socket and value
  suites pass too: 9,878 adapter checks, 3,823 instrumented value checks, and 320
  linked value checks, with real IPv4/IPv6 connections and cleanup. These logs
  precede the final additional scoped-IPv6 test cases; production code is the
  same.
- `gcc-network.log`: GCC 16 passes both linkage modes before the final additional
  scoped-IPv6 test cases; production code is the same.

My Make target is `make -f Makefile.gnu test-nsi-socket-network`; it is included
in `test-units`. The logs preserve exact compiler flags. `source-sha256.json`
identifies final source and tests. These checks ran on Darwin, not Linux.

## Remaining implementation

I still need supervised hostname resolution, public DNS authority, string-bearing
service metadata and paired lowering, and the WebSocket protocol and owned
service integration. I do not treat a trusted C boolean as a verified public
capability. The legacy WebSocket implementation remains unsafe and must be
replaced. I have not rerun a full bootstrap for this adapter-only batch.

My attempts to update #990 and fetch origin failed from this environment.
`issue-update-pending.md` retains the exact proposed ownership update, not an
alternative task ledger. I must publish an evidence update when access works.
