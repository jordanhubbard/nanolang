# My Linux Stage 1 timeout evidence

At source 5dc3a17fb, run 37931895899 fails both Code Coverage (job 113824390605) and Build and Test (ubuntu-latest, x64; job 113824390616). Each terminates raw Stage 1 module generation after the unchanged 1,800-second bound. Neither reaches the module-identity suite. I retain the complete job logs.

The coverage run uses run-e0g9n58l; the ordinary x64 run uses run-5cmlc6aw. Their earlier `stage1/stage2/stage3` Make messages refer to seed/component preparation, not a completed raw NanoISA fixed point. The ordinary x64 terminal means I cannot attribute this solely to sanitizer or coverage overhead.

The same frozen source passed raw generation locally on Darwin in 851.87 seconds for Stage 1 and 862.29 seconds for Stage 2; the complete manifest is in ../imported-managed-bootstrap-5dc3a17fb. Host differences do not establish the timeout's cause. I require phase/profile evidence and successful bounded Linux execution before claiming qualification under #982. The subsequent af2b25da6 Darwin bootstrap is separate and still running when I record this evidence.

Three bounded attempts to retrieve the hosted run artifact list failed with an API connection error. The job logs were retrieved successfully earlier; their failure evidence remains authoritative. I do not claim to have inspected the internal hosted bootstrap logs or manifest.
