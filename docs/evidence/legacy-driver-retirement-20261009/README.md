# Legacy compiler driver retirement

I retire four obsolete compiler drivers under [#983](https://github.com/jordanhubbard/nanolang/issues/983): `compiler_modular.nano`, `nanoc_integrated.nano`, `driver.nano` and `transpiler_driver.nano`. None is selected by the current product or component Make rules. I preserve their historical source identities in `source-provenance.json`; Git retains the deleted implementations.

My phase test previously extracted identical formatter bodies and adjacent shadows from two of those compilers. I retain the exact shared body and shadow in `tests/nanoisa/fixtures/historical_phase_formatter.nano.txt`, with source provenance. I additionally execute all six named phases and two unknown ordinals. The schema ordinal, generated-output equality, diagnostic helper and real C-seed backend-failure assertions remain intact.

My corrected gate passes 23 phase, bootstrap membership, Make dependency, bootstrap diagnostic, tool-link and emitter-component methods in 52.657 seconds. The emitter control includes VM and sanitized native execution. I retain the first invocation: six executed methods passed, but I supplied a nonexistent `test_bootstrap_make_dependencies` module. Its loader error is not a product failure or a passing gate. I rerun with the actual source-dependency suite and adjacent bootstrap controls.

I update current component documentation and redirect the surviving historical wrapper messages to `nanoc_v06.nano`. I label the old integrated-compiler plan historical. This checkpoint does not retire `transpiler.nano`: its extern-declaration selection and Boolean ABI regression still requires a NanoISA replacement. Fresh final-candidate fixed point, installed product and Linux/Darwin acceptance remain in the full release scope.
