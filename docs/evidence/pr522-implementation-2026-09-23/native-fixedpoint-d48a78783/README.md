# My native fixed point at d48a78783

I completed both generations on the pinned source
`d48a78783482bd6bc201ea710b49f4f4d1173eff` in the isolated Linux ARM64 VM.
Both raw modules contain 526,188 bytes and share SHA-256
`513a1e34ae87c714da2549dc78331d7f49bfe3e5185fab487962a489bb91a5c6`. I independently compared the copied raw modules again after the
runner completed. My manifest records source, providers, host closure,
commands, deadlines and successful terminal results. Compressed logs retain
the run output. My final compiler compiles a hello module that verifies and
executes. I preserve the default shadow deadline and 1,800-second stage limit.

This is a within-run fixed point, not a claim that VM and native manifests
produce identical files across different host closures. The changes from
this pin through `d5acbb09a` affect only tests and documentation. Later compiler
changes require new qualification. Complete hosted acceptance and the separate
C-seed-produced compiler module translation check remain open.
