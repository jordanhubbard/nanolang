# Older coverage checkpoint replacement terminal

I retained run39563066808/job119199634283 on8841a2bdf. Restore succeeded. The header dependency control ran at14:55:07Z; a replacement bootstrap run-kh7hcanu completed all17 steps and raw Stage1/Stage2 equality. At16:16:16Z verify-resume correctly refused: `I require continuation of the original successful checkpoint, not a replacement build`. This is not a timeout or instrumentation finding.

The current aggregate contains af4d870f9 (isolated header-dependency OBJ_DIR);8841a2bdf does not. I made no new product change and preserved the integrity guard. Current-head hosted qualification remains pending. Diagnostics artifact12360810577 is195220 bytes, zip SHA256 de90170d92dbc5fc698aee101b4f28986016dde950a3d43f463e28b612b9c7d2. Both retained manifests have17 steps; their distinct work paths establish replacement rather than continuation.
