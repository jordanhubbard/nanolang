# My installed native package qualification

I retain the fresh bootstrap and installed-package test logs here. My package
change installs the standalone native runtime object and resolves compiler
helpers beside the invoked executable, after explicit environment overrides.

My installed-package test uses the actual make install/uninstall targets,
a temporary prefix containing spaces and an apostrophe, an external source
manifest and an unrelated working directory. It checks absolute, relative,
PATH and symlink invocation, execution of each generated product and refusal
with prior-output preservation for an invalid explicit translator override.

My fresh bootstrap passes. The ordinary installed-package test passes in
335.494 seconds, including its install-triggered rebuild. All six original
linker/scalar source methods pass through Stage 1 (5.081 seconds) and Stage 2
(5.409 seconds). I retain the command logs and source hashes. The Stage 2
neighbor run preceded the install-triggered rebuild at the same compiler source.

The additional installed-package run with ASan/UBSan/leak/UAR flags remains
in progress at this checkpoint; I do not claim its result. Its temporary log
is `/tmp/pr522-native-install-sanitized.log`.
This directory does not establish Linux qualification or full hosted acceptance.
