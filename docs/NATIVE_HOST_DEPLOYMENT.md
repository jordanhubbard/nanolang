# My native host-library deployment

I preserve my raw NanoISA import paths. When `nvm2c` emits a native artifact adapter, I record the selected library's SHA-256 and canonical source path in a dedicated ELF `.nanohost` or Mach-O `__nanohost` section. Ordinary strings in a program cannot declare deployment dependencies.

My adapter checks an executable-relative `../lib/nanolang/hosts/` directory first. When that directory exists, I require the entry named by the recorded SHA-256 and verify its bytes before loading. A different library with the same basename cannot substitute for it. When no deployment directory exists, development execution may load the original path after the same content check. I do not change the working directory or export source-root overrides.

An unavailable import remains translatable as a lazy refusal: I abort if it is called and refuse deployment of its unbound declaration. I do not load a library that appears later without a content binding.

My installer consumes declarations from the selected native image itself:

```sh
python3 scripts/install_host_artifacts.py \
    --binary path/to/program --prefix /chosen/prefix --name program
install -m 755 path/to/program /chosen/prefix/bin/program
```

I validate all declared source bytes before publishing libraries. `make install` performs this step for its selected compiler. My receipt under `share/nanolang/host-closures/` records executable and library hashes. Uninstall preserves blobs referenced by another receipt. Relocating the complete prefix preserves executable-relative discovery.

This deployment format currently reads ordinary 64-bit ELF and thin 64-bit Mach-O images. It does not package the host's system libraries, authorize new foreign signatures, prove native code safe, or protect against a concurrent writer replacing executable/library files. Raw-module verification, shadow tests and platform qualification remain separate requirements.
