# Advisory module metadata transport

I retain ordered v2 METADATA key/value pairs through my in-memory module and
canonical assembly. This prerequisite does not implement local-name producers,
frontend proof facts, compute profiles, or high-level reconstruction.

I use existing string-pool indices and lengths, including empty strings and
embedded zero bytes. Duplicate keys remain ordered entries. I give the last
`nano.source_file` entry precedence for the legacy source-file view. My append
API updates that view; serialization refuses a contradictory independently
modified view. With no explicit source entry, I preserve the existing legacy
source-file behavior by synthesizing one pair. I reuse an existing key string.

I strip all source-file entries when explicitly stripping debug information,
while retaining unrelated advisory entries.

I preserve unknown keys without interpreting them. Advisory entries never grant
ownership authority, pure-call eligibility, capabilities, or execution support.
My typed ownership and passive verifiers remain authoritative.

I reject metadata-bearing v1 serialization instead of dropping entries. Ordinary
v1 modules retain their encoding. My canonical `.metadata` directive uses two
string-pool indices so every byte is carried by the existing escaped `.string`
representation. It appears outside functions, after its strings are declared.

Before acceptance I check v2 conversion and text roundtrips, source precedence,
duplicate order, exact bytes, existing ownership/passive facts, lifetime and
allocation cleanup, module-copy/link consumers, ordinary VM/native behavior,
and the genuine canonical compiler seed build. I keep every broader Phase20
acceptance row open.

## Shadow execution markers

Inside a function, `.shadow "target"` emits one `NOP` and retains the target
under the advisory key `nanolang.shadow.offset.<absolute-bytecode-offset>`.
I require a nonempty target without control bytes. My ordinary and owned
self-hosted shadow emitters place this marker immediately before each selected
shadow. With `NANO_SHADOW_TRACE` present, NanoVM reports
`I am testing shadow <target>` to stderr when execution reaches that NOP.
A failed shadow therefore does not report later unreachable shadows. With the
environment variable absent, the marker has ordinary NOP behavior.

These names are diagnostics, not verified source identity or permission to
execute. They neither grant capabilities nor change ownership admission.
Raw-bytecode consumers may author advisory metadata themselves. A bytecode
rewriter that relocates instructions must relocate or discard these offset
keys; copying them across a changed code layout cannot preserve trace meaning.
Native translation preserves execution semantics but does not currently emit
this optional diagnostic; my self-hosted compiler executes shadows in NanoVM.
