# My Apple runtime pass and checker deadline failure

Run [42509585495](https://github.com/jordanhubbard/nanolang/actions/runs/42509585495) at `e9eb29d62606ea4939ddc61e2224d9367b4bb4a2` passes the sanitizer binary-string ownership, complete original VM, private-heap and VM source-method step. The later checker component fails its unchanged ten-second shadow deadline: 693 starts, 692 completions, zero recorded assertion failures and 8.439380 summed body seconds before the final main shadow is stopped. I retain all events, exact early text and complete job conclusions.

The prior a98 Apple checker passed all 693 shadows; I do not infer infrastructure or a specific runtime-change regression from this timing difference. The checker deadline remains a release blocker under #1065. I do not weaken its deadline, selection or assertions. This receipt does not qualify the complete Apple job or actual native-product/SDK custody.
