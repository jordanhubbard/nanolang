# My initial numeric selection failures

I retain terminal [run 42459829646](https://github.com/jordanhubbard/nanolang/actions/runs/42459829646) at `b7bcf56f64119a255ebd8eb4cb44e13ff6b34b3f`. GCC and Clang each execute 27 selected methods and fail four numeric subcases. My C frontend accepts a mixed int/float min call and replaces prior output. All three positive numeric fixtures complete NanoVM execution, but C AOT refuses PICK during scalar classification or callable stack analysis. The complete original builtin fixture and all value/order/identity assertions remain unchanged.

I preserve complete source, VM and AOT commands, outputs, original modules, provider identities, build and worker observations as exact-text bundles and full artifact inventories. Each text member records its original UTF-8 bytes and SHA-256. Original native/module binaries remain in the private archives. Artifacts 13609257849 and 13608309875 pass ZIP CRC checks. Direct artifact ZIPs do not establish original modes or symbolic links.

I keep #1091 open. These are actual C-frontend baseline failures, not executed qualification of my new self-hosted lowering. That lowering remains subject to mandatory bootstrap shadows, both fresh stages and the unchanged complete corpus.
