# My native CPU sampler controls

I add optional CPU sampling around the owned original Stage2 fixture process. My default observer remains unchanged when sampling is absent. With sampling I keep the original argv, pipes and300-second deadline, calibrate first, stop and decode the sampler after reaping the product, and retain raw data, tool identities, call stacks and build IDs. I record intentional SIGINT status without converting it to exit0; nonempty valid decoded samples remain mandatory.

My local software PMU refuses cpu-clock sampling. My first hardware calibration captures93 samples but its observer incorrectly expects exit0 after intentionally sending SIGINT. I retain both refusals. The corrected lifecycle captures92 samples on the already measured ARM hardware PMU and resolves nano_stage2_sample_control, with compiler/source/binary and profiler identities unchanged. I ran this bounded local calibration with privileges because my host restricts perf access; this does not change my host configuration.

My20 process/restoration controls pass, including timeout cleanup and refused-sampler cleanup. An initial protocol selector names a restorer fixture absent from the current product checkout; I load its exact51c revision outside the product tree. My workflow YAML and every shell block parse using the system Python prerequisite; the alternate local Python lacks PyYAML.

Hosted Linux still must calibrate its own cpu-clock:u sampling before touching the retained product. I provision the diagnostic tool and permit per-process user-space perf events on that ephemeral worker. I do not claim a hosted profile, current function-level cause, uninstrumented timing or qualification from these controls. My product source and acceptance limits stay unchanged.

Run `python3 verify.py --workspace /path/on/large/volume` to verify the retained bytes without running a compiler or profiler.
