# I qualify the corrected mixed-service live peer

I reproduced the old live-peer failure without a sandbox restriction: my generated client requested `/test`, while the shared peer required `/?mode=test` and disconnected. I retain that failure with all five other passing mixed-service methods and the passing five WebSocket driver methods plus numeric/DNS source lifecycle (`baseline-and-adjacent.log`).

I correct only the generated URL, preserving the exact request assertion, both service-catalog graphs, numeric and DNS hosts, all driver/VM/native routes, and the expected binary message count. The unchanged live-peer method now passes in 63.990 seconds (`corrected-live-peer.log`) on macOS arm64 with compiler products from commit `8054f17d3`.

I do not infer Linux or exact-release qualification from this local result. Paired source lowering passed in the adjacent full matrix, but this URL correction does not close every requirement under #990.
