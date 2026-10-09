# Broad Darwin gate at 8c1e113ec

I ran `CC=/opt/homebrew/opt/llvm/bin/clang NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang NANO_VM_EXAMPLE_SHADOW_TIMEOUT_SECONDS=60 make test-quick TEST_TIMEOUT=3600`. My terminal exit status is 2. My final language group passes 17 cases and fails the module metadata and module introspection flags cases: my self-hosted NanoISA lowering rejects their generated introspection extern symbols. I retain the complete terminal; this is not a release pass. Follow-up belongs to #982 and #976.
