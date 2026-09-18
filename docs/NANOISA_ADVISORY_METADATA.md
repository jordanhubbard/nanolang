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
