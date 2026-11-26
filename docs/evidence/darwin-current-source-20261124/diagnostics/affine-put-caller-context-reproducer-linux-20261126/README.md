# My bounded caller-context reproducer for #1262

I retain three bounded reductions of the scalar-array write refusal in my legacy C translator. My original production `put` body and shadow come from `901caef8e6f995080ac7d3c48c4cf23c75c99315:src_nano/nvm2c/affine_analysis.nano`. I change its visibility only in the imported-setter variant.

| Caller context | Native / C-produced VM / genuine-produced VM | C-produced module → C | Genuine-produced module → C |
| --- | --- | --- | --- |
| Imported setter | All return `7` | Pass | Pass |
| Nested factory and context | All return `7` | Refuse | Refuse |
| Bounded mutual recursion | All return `7` | Refuse | Refuse |

I complete 24 commands, including nine observable probes and four translation refusals. I retain 600-second build/translation limits, 60-second probes and the ordinary ten-second shadow deadline. My selected inputs and tools match before and after execution. No command times out. The nonrecursive nested case needs three small source files; I do not claim global minimality or a sole cause.

My read-only debugger traces locate both nested-case refusals at `put`, byte 42 (`ARR_SET`): function 1 in the C-produced module and function 3 (`__nano_module_1_put`) in the genuine-produced module. The receiver has unknown kind and origin 0, the value has array kind, and the facts are nonfinal and noncanonical. My module and translator hashes match before and after debugging. These observations localize the refusal; they do not establish a repair.

I verify the compressed objects, original helper source, all command outcomes, guards, decoded instructions and traces with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-put-caller-context-reproducer-linux-20261126/read_reproducer.py
```

I can extract the nonrecursive case to a new directory. I rewrite only its absolute import prefix:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-put-caller-context-reproducer-linux-20261126/read_reproducer.py --extract /var/tmp/nanolang-put-repro-new
```

My archived `reduction/result.json` records the exact producer, VM and translator paths, arguments and hashes. With those recorded tools, the C producer compiles the extracted `main.nano` to a module, the VM prints `7`, and the translator refuses it:

```sh
nano_virt /var/tmp/nanolang-put-repro-new/main.nano -o /var/tmp/put-repro.nvm
nano_vm /var/tmp/put-repro.nvm
nvm2c /var/tmp/put-repro.nvm -o /var/tmp/put-repro.c
```

I retain the genuine producer route and its same refusal in the archive. Other tool builds need their own observations. The archive reader needs this repository's recorded source commit and the adjacent shared `read_chunks.py`; extraction does not execute a compiler.

My manifest SHA-256 is `af05221123efd8afe0e3aa237cf0aed5a16e26ed2b2139ca1099d9d57bfcbc9c`, with 100 content-addressed records. I preserve the earlier positive standalone-helper control separately. I stop after these three variants. This receipt accepts the reproducer matrix only: #1262 remains open, no C implementation changes, and no component, five-route, paired-platform or release acceptance follows.
