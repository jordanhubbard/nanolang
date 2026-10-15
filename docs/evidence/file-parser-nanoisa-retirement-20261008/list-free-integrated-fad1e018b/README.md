# My integrated list-release gate

At unchanged `fad1e018be60f825075994fd03293e084af76f3a`, `make -j2 test-nanoisa-list-free test-nanovirt CC=/opt/homebrew/opt/llvm/bin/clang` exits zero after 110.987 seconds. All ten paired methods and 90 NanoVirt checks pass. This rebuilds the component driver from integrated source. The source commit, tracked status and user-file hash remain unchanged. Fresh installed-stage and full parser qualification remain open under #978.
