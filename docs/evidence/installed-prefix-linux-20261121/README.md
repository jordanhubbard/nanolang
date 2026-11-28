# My independent installed-prefix controls

I retain a partial repair under #1167/#1161/#976. My successful frozen bootstrap remains at `c908523c3`; these later compiler and installer changes do not establish a new Stage1/Stage2 fixed point.

My first actual Make installation succeeds, then my installed compiler refuses a shadow-bearing external module because it searches the client working directory for `bin/nano_vm`. My new regression reproduces that failure through both an absolute compiler path and PATH lookup. I discover the installed sibling tools before requiring checkout markers. My changed `compiler_tool_root` shadow also checks that my discovered root contains the VM and translator. My installer builds and copies the required `nano_aot_runtime.o`, with matching uninstall cleanup.

I build a development compiler from the changed source using the genuine C seed. Its source-module build passes in85.730s, verification in0.266s, C translation in67.221s and native compilation in233.941s. I retain the original30-second complete compiler-shadow budget and1800-second build-command budgets. My resulting native development image has SHA256 `d88f5dbd92ed3c905f6b6ccd0520704a437382f3155eebaf7930218f228e80d8`.

My first combined25-method development run passes the new independent-prefix regression but retains one setup error: this implementation checkout has no native Stage1 image. That attempt also inherits the compiler factory's30-second shadow environment, so I do not use it for ordinary deadline acceptance. I then run every unchanged method/assertion with the immutable frozen fixture/tool root, its actual distinct C seed and native Stage1, and my new selected development compiler. All25 methods pass in6.074s with the ordinary ten-second shadow deadline. I create no stage aliases and skip no producer.

My actual new `make install` consumes the explicit completed development image. I use Make's `--old-file` for that input because the generic compiler target otherwise owns a symlink recipe; I preserve and compare the actual ELF bytes. This independently installed compiler compiles, verifies and executes the passing-shadow module, then compiles and executes its native product from an external client directory. Both executions print `fresh-prefix-ok`. I unset source-root, module-path, tool and helper overrides; raw module emission has `NANO_CC=/bin/false`.

## My remaining installed-closure failure

My namespace setup succeeds with ordinary filesystem permissions. I then hide my known NanoLang source/build roots while preserving the installed prefix, external client, system libraries and a separate cache. The installed compiler exits134 in0.165s before module publication:

```
I stopped at a native invariant in nhost_source_inputs_11 at generated C line 1512.
```

The retained generated C at that line checks a failed `dlopen` of `libfile_source_inputs.so` through an absolute development build-cache path. My sibling-discovery repair does not relocate my host-library closure. I retain the exact failing installation and all six actual development host libraries; I do not add path aliases, rewrite raw modules or export a source-root workaround. Root-independent installation and paired-platform acceptance remain open in #1167. My pure translator, failed GCC13 public sanitizer gate, ownership and full release obligations also remain open.

I recheck the original frozen source/tool/host/artifact guards and my collected development host-library hashes after these probes; all match. My [manifest](manifest.json) retains1696 logical files,456115284 raw bytes and104159116 unique compressed bytes, with each raw and compressed SHA256 checked after writing. It includes exact original/corrected sources, controllers, terminals, emitted C/modules/native products, installations and host generations. My local failed setup evidence remains alongside the successful controls.
