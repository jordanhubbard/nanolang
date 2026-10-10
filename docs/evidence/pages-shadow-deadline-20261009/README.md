# My Pages shadow deadline

Pages job113754851927 at18fd40099 fails while building the NanoISA emitter component: all selected shadows run, then hit the default 10-second deadline. My main CI already supplies NANO_SHADOW_TIMEOUT_SECONDS=60. I give Pages that same finite hosted deadline and retain make userguide-check unchanged.

My Linux ARM64 probe executes the exact component compiler command with a private module cache and the 60-second deadline: bin/nanoc_c src_nano/nanoisa_driver.nano -o /work/pages-emitter-probe. Compilation exits zero with all dependency shadows enabled. The resulting component executes successfully and prints its entry-assertion confirmation. Workflow YAML parses and both hosted deadlines agree. This focused component result is not a full hosted Pages qualification.
