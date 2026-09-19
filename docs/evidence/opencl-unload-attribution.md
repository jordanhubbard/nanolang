# I identify the module in my OpenCL leak report

My fresh diagnostic at `4b71cc2ab` builds successfully, passes the original 48 buffer, 24 kernel and 384 exact integer observations, then exits 1 with the same 75-byte, three-allocation LeakSanitizer report. I preserve that failure. I do not suppress leaks, retain the loader, change GPU assertions, or replay the original failed executable.

My diagnostic wrapper captures `/proc/self/maps` immediately before calling the real `dlclose` once. All seven unknown-frame occurrences in this new process map into `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1.0.0`, at file offsets `0xb24c`, `0xb0ac`, `0xa8cc`, `0xb50c`, and `0xc1b4`. The installed package is `ocl-icd-libopencl1:arm64` version `2.3.2-1build1`; its ELF build ID is `d2e11dd86dc02194317b8b27a72e8ccdd08abc57`. These are same-process map observations. I do not reuse historical process addresses.

This identifies allocation-stack frames in the OpenCL loader. It does not establish allocation ownership, the proper shutdown contract, an upstream defect, or a production fix. The installed library lacks usable source-line information: nearest exported names from `addr2line` are not exact function attribution. Task `315caf01b3764a3eb655d825c9203e24` and actual GPU sanitizer acceptance remain open.

My [manifest](opencl-unload-attribution/manifest.json) seals the original commands, logs, mapping analysis, source/tool/library maps, runner and module inspection. Source, tools, libraries and HEAD remain unchanged across the run. The fresh diagnostic binary remains at the path in `binary.json`; its digest is retained. My original uninstrumented fixture and OpenCL runtime hashes equal the earlier qualified sources. No production file changes in this diagnostic.

My independent audit rehashes all 7,278 sources, six tools, thirteen libraries and the fresh binary, and independently maps each unknown frame against the single captured executable mapping. It confirms the bounded attribution and retained failure without repeating execution.
