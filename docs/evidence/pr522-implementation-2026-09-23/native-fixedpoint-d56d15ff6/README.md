# My final-source standalone-native fixed point

I qualify compiler-source commit `d56d15ff6934dcd4872aa0f90bfe7ea828cf79fa` in a clean Linux ARM64 worktree on the 16 GiB qualification VM. My retained driver builds the C seed, emits and verifies the initial module, then translates it into a standalone compiler. That compiler emits Stage 1; its translated successor emits Stage 2.

My raw Stage 1 and Stage 2 modules are identical: 530,296 bytes, SHA-256 `d678e0e1fe9e290592721b265f204b2917fc9369bf8db32d1c28835cc2c3f2ba`. I independently compared the copied raw bytes and checked their hash again before archiving this evidence. Both modules verify. Each generated compiler builds with C11, `-Wall -Wextra -Werror -O0`, and links without `libnanovm`. My final compiler emits the unchanged hello example; its verified module executes with `Hello from NanoLang!`.

I preserve exact commands, durations, provider hashes and host-library hashes in `manifest.json`, the executable procedure in `native_gate.py`, and every stage log alongside them. Provider and host-library hashes remain unchanged through the run. Stage 1 emission takes 482.639 seconds; Stage 2 takes 450.257 seconds. The per-command bound remains 1,800 seconds and the existing shadow deadline remains unchanged.

An earlier attempt stopped when a second full worktree exhausted the qualification VM disk; GCC reported an assembly write failure. I preserved that attempt separately, excluded only archived `docs/evidence` from the qualification worktrees, and reran into a fresh evidence directory. I did not exclude compiler sources, tests, or build inputs. This successful run does not explain unrelated historical sanitizer deadlines.

My VM-route fixed point is recorded separately in `../vm-fixedpoint-d56d15ff6`. Its modules have a different host-path closure and byte count; I claim equality within each pinned route, not equality between those two environments. Complete hosted platform, coverage and sanitizer qualification remains required.
