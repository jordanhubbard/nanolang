# My updated development compiler and factory observation

I retain exact source `bf9fef87f2c965db3f8c9f6c813a542a15333675`, including the bodyless scope change in `ee78adba9c`. I emit the compiler with the actual retained 8b2 bootstrap Stage2 image, SHA256 `2eb02fd33d8cc8d8ca5f2684b4399d55540b648a1f4eb2570e6ead118e03a63d`. I preserve its original bootstrap receipt and host closure. This explicit native-producer development derivation is distinct from the original Make bootstrap: I establish no new raw fixed point and publish no Stage1/Stage2 alias, bootstrap receipt or sentinel.

I give compiler-source generation the documented compiler-building 1800-second parent limit and 30-second shadow limit. I retain every command, complete stdout/stderr, actual product and source/tool/image/host guard. Verification, C translation and native compilation use the original 600-second component budget. Native compilation uses the exact qualified compiler and flags recorded by the retained bootstrap receipt.

I emit the 1,059,248-byte NanoISA module in 1355.242 seconds (SHA256 `68ed64ddbcf2a2ecb90770195e61d24dfc0119cc2e12203eea58725dd79049b5`), verify/dump it, translate it in 57.994 seconds and build the native compiler in 175.092 seconds. The 19,732,544-byte native image SHA256 is `aa6472382c2130bb166742d7bee32f180c68341d77a2bebd9d63a0f81194844e`. My original native suite passes all 36 methods in 32.792 seconds, with zero failures/errors/skips. My factory observation still times out at its unchanged 300 seconds: exit -9 after 301.279 seconds including reaping, no product and stable selected guards. Markers record imports at 59.297 seconds, lexer at 66.319, parse at 95.594, checked emit at 99.906 and typecheck-start at 99.947; no typecheck completion is recorded. I retain that failure without an unchanged retry.

The consumer checks retain their original limits. My native suite runs the original `make -f Makefile.gnu test-native-shadows` through the explicit development producer selector with ordinary 10-second supervision. My single factory observation invokes the same unchanged `tests/selfhost_shadow_emitter.nano` under a 300-second parent limit/default10, records existing phase markers and retains temporary terminal products. Complete consumer stderr/stdout survives timeout. The factory observation is not execution of the complete 12-method emitter suite and cannot grant its acceptance.

I claim no historical timeout cause, whole-compiler speed gain, fresh bootstrap/own-Stage2, current paired platform, pure NanoLang translator/cutover or full release acceptance. #1255 remains open. I preserve every old root and change no assertion, deadline, CI compiler pin or peer-owned C/profile/capture source.

I independently verify 5996 records, manifest SHA256 `e6a8aa9cd29b1bcb7623c02e7bb2514e8b05b126bb2646458018f13e605499d8`, actual source/producer/native identities, all original command roles, outputs and final guards with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/bodyless-compiler-development-linux-20261125/read_archive.py
```
