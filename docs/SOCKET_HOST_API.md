# I grant outbound TCP explicitly

I accept one checked `nsi:nanolang/net` service declaration per source graph.
My public source type is `Conn`; my private Socket token is a separate identity.
I require a validated companion document and checked nominal, body and ownership
facts before lowering. Numeric IPv4/IPv6 endpoints are supported; DNS and
WebSocket integration remain required release work.

I compile source and run its selected shadows with an explicit connection grant:

```sh
bin/nanoc_c client.nano --allow-tcp-connections --emit-nvm -o client.nvm
bin/nano_virt client.nano --allow-tcp-connections -o client
```

My Nano compiler accepts the same source option. I run dependency shadows before
root shadows by default; `--root-shadows-only` changes selection, not source body
checking. Each selected shadow receives a fresh grant in the supervised child.
One default ten-second deadline covers the entire suite. The bounded
`NANO_SHADOW_TIMEOUT_SECONDS` override applies to that suite. I remove descendants
and require cleanup before publishing. This supervisor is not a security sandbox.

I preserve prior output on missing authority, failed shadows, timeout, compiler
failure or malformed bytes. Source and companion aliases cannot be outputs.
With no selected shadows, translation needs no network grant and performs no
network execution. A compile-time grant never persists into the output.

Each invocation needs its own authority:

```sh
bin/nano_vm --allow-tcp-connections --socket-instruction-limit 1000000 client.nvm
./client --allow-tcp-connections
bin/nano_virt client.nano --allow-tcp-connections --run
```

The VM requires a decimal instruction limit from zero through 1,000,000, and
rejects combinations with other execution modes. Zero means zero fuel. The
source driver and generated launcher use the bounded maximum. The grant permits
outbound TCP acquisition without restricting destination addresses; it does not
authorize listeners or arbitrary foreign calls.

My nonexecuting C emitter validates exact bytes without a connection grant:

```sh
bin/nvm2c --socket-tcp --entry-name client client.nvm -o client.c
make -f Makefile.gnu PREFIX=/desired/prefix install-socket-public-runtime
```

The generated `nvm_socket_indirect_program_client` accepts a `NvmSocketHostGrant`,
`NvmSocketIndirectOptions` and `NvmSocketScalar` output. I install C99 headers
beneath `include/nanolang/socket` and `lib/libnano_socket_runtime.a`. Public VM
execution uses `nvm_socket_execute_indirect_bytes`. I publish scalar output only
after successful cleanup. Missing, revoked or busy grants preserve it.

File and TCP public calls share one gate. Reentrant or concurrent calls refuse
BUSY before inspecting arguments. Grant identity, ABI and catalog are checked
separately; pointer lifetime follows ordinary C rules. Mixed File/TCP programs,
fresh bootstrap and the complete platform/release contract remain open.
