# I retain integrated capture and CI qualification

I qualify source `c0bede240d8974c156cc8d190ed43976902102c7`, integrating Puck's promoted `04791f56dba3d3df74dfdc3e57bd9816645247cd` inputs. My 42 capture/source/test/CI files match that peer exactly. My Make change only appends `tests.test_gnu_tool_queries` to the existing recipe. I preserve all 14 protected source/test inputs, including Builder, bootstrap, translator, regex and runtime work. Puck retains capture/CI implementation ownership.

My production capture source is SHA256 `971b242622730d2581bd5d576bec8dc7ce48a05c67e4917765d4f347f346bb09`; my source fixture is `7e80518ce26a983ec316fe161cbbf7c5a77a64724554ba3dc592f690ab7ee259`. I preserve all original assertion/skip ASTs: 728 checks across 128 source methods and 13 checks in the response method. I change no assertion, skip condition, deadline or compiler version for these gates.

| Linux ARM64 gate | Result |
| --- | --- |
| Profile, image, refusal and fixture controls | 38 pass, 4.145s |
| GNU query methods | 3 pass, 0.132s |
| Original source suite | 128 methods, 24 original conditional skips, zero failures/errors, 977.711s |
| Original response matrix | 600 equivalences pass, 35.543s |
| Actual cold compiler_support capture | Pass, 2.369s |
| Fresh ordinary guide | 40 snippets pass, 327.583s, ordinary default shadow 10 |

I retain prepared tool/source identities before and after. My local compiler is installed GCC13 with supported assembler2.42. My separate, clean, detached local qualification checkout starts without installed bootstrap state. Its original native selector preserves both tools and exports no override. The original Make guide flow builds the three compiler components, uses `nanoc_c`, passes all snippets and finishes Git-clean. I retain this checkout at `/tmp/nanolang-capture-ci-clean-20261122-my86wavm`; I preserve the earlier installed bootstrap and frozen roots.

My actual [hosted guide run45097562054](https://github.com/jordanhubbard/nanolang/actions/runs/45097562054) succeeds at merge `64a604f41409af72b2a9b2d88bee89a77d0f5e0d` (head c0bede, base `74644c29fb62b54fc3b1b519cfa91144cec0caaa`). I verify actual merge blobs for capture source, source tests and Make against my qualified head. The native selector refuses generic assembler2.46 and selects committed r1 image SHA256 `7e9edb4b7fc0c7ce97fbcdd4ecb63e3eb1c1663e2452cd32ce0249110b253bf5`, reusing completed producer45009129801 without rebuilding GAS. The original GCC15 image remains unchanged. All 40 guide snippets and HTML generation pass; PR deployment is skipped. The workflow retains its original shadow60; my separate local ordinary flow proves default10. I retain the official run/jobs/log and downloaded native artifact files. I do not claim verification of the downloaded zip container.

I import four complete peer provenance capsules (8,401 records): producer1,449, GCC1,476, Clang1,838 and Darwin3,638, including every referenced compressed object. My local manifest contains 1,514 records, with hash-bound immutable base archives. My hosted supplement contains 24 records. My reader checks compressed and raw hashes, duplicate descriptors, gate counts, image/source identities and actual hosted status. I preserve the two observer setup refusals (a misspelled preflight path and a missing output parent), both before native commands; neither requires repeating a native cohort.

```sh
/usr/bin/python3 docs/evidence/capture-ci-integration-linux-20261122/read_archive.py
```

My [retained verification](verification/linux.json) reports `INTEGRITY_PASS` and `hosted_completion: true`. My hosted manifest is [separately bound](verification/hosted-manifest.json).

My prior [09cab full Linux bootstrap](../bootstrap-current-linux-09cab-20261122/README.md) remains qualified only at that exact pin. I do not extend it to this later capture/Make source, current Darwin, the pure translator, the complete source census or all 69 release requirements. I preserve earlier hosted failures without assigning a retrospective cause. I hold no native worker reservation after these terminal gates. The migration/TCC/NFS operational pause remains unchanged.
