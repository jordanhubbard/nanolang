# My default NanoISA product policy

I make the unqualified source invocation publish verified bytecode, as required by my NanoISA-only architecture. This draft follows product cutover PR522 and module-facts PR534; it does not claim their remaining native bootstrap gate has passed.

| Invocation | Product |
| --- | --- |
| `nanoc file.nano` | `file.nvm` |
| `nanoc file.nano --emit-nvm` | `file.nvm` |
| `nanoc file.nano --emit-nvm -o saved.nvm` | `saved.nvm` |
| `nanoc file.nano -o program` | native `program`, through NanoISA and nvm2c |
| `nanoc file.nano --target native` | native `a.out`, through NanoISA and nvm2c |
| `nanoc file.nano --target c` | translated `file.c` |

I derive sibling filenames from the last extension in the final path component, preserving leading-dot filenames. I reject a bytecode flag combined with either explicit target. My existing source/destination identity checks apply to derived names too. An explicit `-o` alone retains its existing native meaning; a `.nvm` suffix does not silently change that request.

My C-seed-built canonical driver passed four policy methods: bytecode defaults including hidden and extensionless names, native and C overrides, conflicting targets with previous-artifact preservation, and derived output refusing to replace its source. All 23 adjacent module-facts, VM-shadow, canonical output/artifact and product-route methods passed. Build and test logs are `/tmp/nanolang-default-product-{build,driver,tests,adjacent}.log`.

This closes the bounded CLI policy implementation under `task_d76ae44a12fd4d27a2b4aa84c30d7bc6`, not the whole product acceptance. I have not run the known failing native Stage2 bootstrap again. My separate raw VM fixed-point run remains pinned to PR534 source, before this CLI change.
