# My original emitter fixture after the checker repair

I compile the original tests/selfhost_shadow_emitter.nano with the C seed from6c54346f0f4b62e2252776e1cd287651420c6672, integrated as e598013cd5363be1664a40a7fe7c615f7b2cdb2a. I retain the default10-second shadow policy, original300-second outer bound and both ordinary pipes. My command exits0 after43.270 seconds total; source and actual compiler hashes match before/after and the compiler matches my preceding checker component.

This single local fixture pass does not qualify all17 bootstrap steps,36 native cases,12 emitter cases, both platforms, cache/workers or the release. The complete unchanged original platform matrix runs separately at e598013cd5363be1664a40a7fe7c615f7b2cdb2a. Run `python3 verify.py --workspace /path/on/large/volume` to verify the retained receipt without executing my compiler.
