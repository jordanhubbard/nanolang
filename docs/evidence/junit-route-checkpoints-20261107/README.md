# My complete producer-route checkpoints

I retain the complete original 237-case selection on all four routes, with zero skips. Both runs finish fresh bootstrap and collect every route despite failures.

| Source pin | Run | Seed native | Seed VM | Stage 1 | Stage 2 |
| --- | --- | --- | --- | --- | --- |
| eef81410cf0e34d042ce584ac5f79abf6f48d5e8 | 42413840812 | 236 pass / 1 fail | 233 / 4 | 173 / 64 | 173 / 64 |
| ebaefe24bca13fb4a94ba6018b098ea5182bcae6 | 42415103105 | 236 / 1 | 234 / 3 | 174 / 63 | 174 / 63 |

I verify each hosted source-tree.txt against `git ls-tree -r` of its recorded commit. I retain original XML, summary, every compile/run log, bootstrap manifests, provider hashes and before/after verification. Large text members use deterministic gzip. Archive inventories identify every member, including executable bytes retained in the complete private archives; I do not substitute rebuilt executables for them.

The source float correction removes the full f-string failure on the VM and both stages. The native route still fails test_resource_effect_checker; the VM additionally fails both generic-context fixtures. Stage failures remain recorded individually in the complete reports. These are failed complete reports, not release acceptance. The focused canonical suite also retains the original NanoVirt str_trim scalar-refusal failure; both installed stages pass its source formatting methods at these pins. Later STRING-checker corrections have separate evidence.
