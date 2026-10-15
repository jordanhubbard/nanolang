# My checked TCP document preparation

I prepare an immutable complete TCP interface document through
`nl_socket_binding_prepare`. I accept only the exact catalog in
[my TCP binding contract](NSI_TCP_BINDING_CATALOG.md). I return one owned plan
containing canonical JSON and a service declaration. I do not open a socket,
publish a file, select a grant, or admit compiler execution in this operation.

I use the same bounded engine as File: UTF-8 and complete-span preflight,
structural limits, decoded duplicate-key refusal, exact catalog comparison,
canonical rendering, and one canonical-document revalidation. I preserve the
caller's output pointer on failure. My successful plan outlives its input;
returned byte views borrow until the corresponding plan is freed. My shared
limits and status values live in `nsi_binding.h`. File's existing public names
remain aliases with their original values.

I instantiate the private engine separately for File and Socket. Each API has
its own opaque plan type, exact catalog validator, and source renderer. I do
not use a caller-supplied catalog or a runtime callback to grant authority.
The source template is tracked as a build dependency, including my explicit
File publisher build. My module builder already records compiler-discovered
include dependencies without restricting their suffix.

My Socket source output is this declaration:

```nano
service "nsi:nanolang/net" catalog 1 from "interface.nsi.json"
```

I do not fabricate network shadows in document preparation. Unlike File's
self-contained temporary-file tests, successful TCP traffic needs a selected
peer. Paired C/Nano namespace/type/lowering integration, real selected network
shadows with an owned peer fixture, VM/native dispatch and cleanup, public
connect and WebSocket integration remain required by #990. A declaration alone
does not complete those requirements. File's existing five generated shadows
remain byte-identical to their checked fixture.

I qualify both document corpora with independently linked and allocation-hooked
providers. I mutate every scalar catalog fact, test all document limits and
encoded duplicate keys, inject allocation failures, overwrite/free original
input, and verify canonical round trips. I reject File documents through the
Socket API, Socket documents through File, legacy network declarations and
private `Socket` substitution for public `Conn`. I also keep both plan types
live in one process and free them independently. My
[Darwin evidence](evidence/socket-binding-20261009/README.md) states the tested
scope; candidate Linux/Darwin release acceptance remains separate.
