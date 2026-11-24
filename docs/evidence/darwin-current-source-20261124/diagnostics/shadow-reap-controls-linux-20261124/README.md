# My bounded completion/reap controls

I investigate #1229 after my original current-source command completes all 1306 shadows, flushes about9.795s after parent start, then reports clean exit0/completed=true/parent_timeout=true at10.279s. That full-source failure stays unchanged in the [literal-binding receipt](../fresh-record-adoption-linux-20261124/README.md). I make no unchanged full-source retry or production edit here.

I extract the exact parent wait loop from `63248db641:src/main.c`, keep its clock/kill/reap/timeout decisions verbatim, and compile a separate native process diagnostic with the original strict compiler flags. These are calibrated one-second process controls, not compiler qualification. I wrap only the native diagnostic's `waitpid` call to insert a 1.3-second parent scheduling delay after its first zero WNOHANG result. A separate thread calls actual `waitid(WNOWAIT)` and independently observes the child's zombie status without reaping it. Both parent and child clocks use `CLOCK_MONOTONIC`.

| Native control | Child before exit (s) | Independently observed zombie (s) | Parent receipt / exit | Parent timeout |
|---|---:|---:|---|---|
| Ordinary, no parent delay | 0.050255 | — | yes / 0 | false |
| Timely child, delayed parent | 0.050364 | 0.200508 | yes / 0 | true |
| Late child, delayed parent | 1.100622 | 1.200818 | yes / 0 | true |

I deliberately ignore the child's alarm only in the late negative, as foreign code can do; my unchanged parent deadline must still reject it. My original loop reports the same receipt/exit/timeout combination for the independently on-time zombie and the deliberately late exit. A clean exit and receipt observed afterward are therefore insufficient to establish when completion happened. The native delayed-parent control establishes a possible mechanism, not the cause of my original compiler run. I do not distinguish parent scheduling from process teardown in that run and do not relax its deadline.

I preserve my initial diagnostic compile refusal for misleading indentation, then correct only that diagnostic formatting. Werror stays enabled; the extracted production loop is unchanged. All three native controls pass their declared checks. No production source, assertion, deadline, ownership default, capture code, old root or NAS state changes.

I retain42 file records, native source/image, full stdout/stderr and actual compiler/tool/library identities. Manifest SHA-256: `1680a82ad813d393f4db389b1c3757dba654218f0d23e26efe7ce42035339367`.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/shadow-reap-controls-linux-20261124/read_archive.py
```

My reader checks every archive byte, the verbatim loop against Git, native ELF architecture, actual observations, the independent zombie timestamps and unchanged tool identities. Full-source, paired Darwin and whole-release acceptance remain open.
