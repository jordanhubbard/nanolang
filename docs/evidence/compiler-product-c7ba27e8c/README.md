# My clean compiler-product checkpoint at c7ba27e8c

I ran the complete `make -j2 test-one-ir-compiler` gate at
`c7ba27e8c29906aed9aa18cf0dc1bea3e30fcb3e` in a clean detached clone, with
explicit Homebrew LLVM and a clone-local native host cache. All 95 methods
passed in 805.224 seconds; Make completed in 806.803 seconds. The source stayed
clean and its HEAD unchanged. The manifest binds the source, command and log.

This separate 2026-10-08 attempt followed the preserved interrupted attempt.
I did not infer failure or success from that partial log. This result qualifies
the native named-function record/array repair and full compiler-product routes
at the stated pin. It predates my self-hosted source-container repair at
6a39d5e6c, and does not establish a current raw fixed point or final release.
