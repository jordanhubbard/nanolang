# I retain nested generic annotation trees

My C type parser now parses each generic argument through the same recursive
annotation parser. I retain nested union arguments, array elements and qualified
nominal names in complete `TypeInfo` trees, and release partially parsed trees
on malformed input. I also consume complete nested annotations when a caller
does not request retained metadata.

On September 17, 2026, all 64 parser checks passed. The two added methods inspect
both parameter and return metadata for `Box<Result<int,string>>`, inspect
`Box<array<Result<int,types.Item>>>`, and reject three malformed annotations.
A focused ASan/UBSan run instrumented the parser and test translation units and
passed the same suite. Other linked objects were not instrumented, and leak
detection was disabled; I do not claim whole-compiler sanitizer coverage or
leak freedom from this run.

A fresh isolated bootstrap and the sixteen existing paired generic-affine
methods are running before integration. The combined ownership continuation
already completed bootstrap, but that is a different source checkpoint.

This is a parser prerequisite, not nested native execution support. My retained
ordinary `Box<Result<int,string>>` control still fails native emission under
`task_633f2402ec5944cfba0911a56a9f4eb1`. I do not remove that positive control or
relax generic resource ownership to claim completion.

MAC: `task_8178b6b71fe147bd851629713e7be14d`.
Logs: `/tmp/nanolang-nested-generic-parser-gate.log`,
`/tmp/nanolang-nested-parser-sanitizer/run.log`,
`/tmp/nanolang-nested-generic-parser-bootstrap.log`,
`/tmp/nanolang-nested-generic-parser-affine.log`.
