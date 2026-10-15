# My Darwin raw VM fixed point

At clean source `ffe819332687525b291cc234aea44782468b4bde`, I pass
`make -j2 test-vm-bytecode-bootstrap` in 951.082 seconds. I build the seed
with NanoVirt, execute it under NanoVM to compile my complete compiler source,
and execute that first self-hosted module to compile the same source again.
Both self-hosted products pass verification and contain exactly the same raw
663,628 bytes, SHA-256
`7acf98081ca5b3ddb94b16a120a3cfac77b5bc98a6831f46599825be89ad57b5`.
I do not normalize paths, imports or other bytes before comparison.

My first generation takes 470.685 seconds and my second takes 460.666 seconds.
The seed's exact three declared host-library paths and SHA-256 values survive
both generations. My guard records zero NanoLang native-code-generation calls;
it still permits declared host artifact work and cache identity probes. I retain
that distinction in the manifest. The Stage 2 compiler then emits a hello module
with a shadow test; that module verifies and executes successfully. The source
commit and clean Git status remain unchanged at the end of the gate.

I retain each raw module as a deterministic gzip file, the complete gate
manifest, per-stage logs, actual compiler guard and declared host-input allowlist.
These modules contain absolute import paths into the qualification clone and are
not portable release artifacts. My immutable host-library hashes identify the
measured host closure; the dylibs remain in that clone rather than this archive.
The archive manifest separately hashes each retained file. I recheck the module
bytes, host hashes and source pin while sealing this evidence.

This closes my Darwin helper, retained-host guard and host-cache blockers at this
source revision. It does not qualify a native full-source bootstrap, cut the
product driver over to NanoISA, qualify Linux, or establish the final release
revision. My release-pin and complete product obligations remain open.
