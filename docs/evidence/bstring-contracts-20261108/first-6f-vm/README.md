# My first VM ownership attempt

At `6f3f5c5bce0ef550a29ecc213f5926464cc68032`, both Linux ownership-test executables fail to link because I omitted their `g_argc` and `g_argv` entry globals. Neither sanitizer executable runs; the source VM methods and checker component are skipped. I retain complete early artifact text and ZIP hashes here from run [42509206230](https://github.com/jordanhubbard/nanolang/actions/runs/42509206230). The original translator and alias gates continue independently; this early receipt does not claim their terminal results.

I also find standalone heap consumers that do not link `nl_string.c`. I share the existing two-allocation payload destructor in the header, preserving the external C free wrapper and the standalone heap link closure. Corrected execution remains required. I do not classify these failures as infrastructure.
