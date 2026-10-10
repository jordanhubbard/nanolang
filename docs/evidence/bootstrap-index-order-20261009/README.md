# My generated-index bootstrap ordering

The standalone 8c1e113ec bootstrap run-tsitggza has no modules/index.json in its recorded source set. Subsequent make test-quick first executes make build, which generates that file, and then starts run-javkp8ib. The two manifests differ only at that added file in their source maps; tool hashes and recorded compiler/linker settings match. The index generator also contains a wall-clock generated_at field. I do not remove this file from frozen-input validation.

My prepared fix makes the generated module index an explicit prerequisite of bootstrap Stage1. This supplies the file before the first snapshot even when Make's wildcard source list initially cannot see it. The regression uses the real Make rules in an already-built fixture, removes the index while restoring directory mtime, and requires its regeneration before the bootstrap script. The old rules incorrectly return up-to-date. All ten dependency methods pass with the candidate.

My first candidate fixture run reports Make's missing fs.c prerequisite because an existing deletion test deliberately removes it. The fixture already holds native provider binaries built to isolate bootstrap invalidation; I include the index generator in that same held-tool list, keeping every source/membership assertion unchanged. I retain the initial terminal.

The patch is prepared, not integrated: the broader test-quick gate is still running against unchanged 8c1e113ec compiler inputs. I will apply and qualify the dependency change after that run ends under #982.
