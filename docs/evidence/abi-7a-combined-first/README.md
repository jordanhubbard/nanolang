# My exact7a combined qualification

I froze production at `7a001106b`, with 4012 Git inputs. My corrected driver SHA256 is `b0fcbdebc6c706bf473ba90b3a086bb7f61ee505d83ed25099d1d2f397c5f184`; it builds the binding generator before pinning the provider closure.

Linux passes all 22 ordinary/sanitizer phases. Darwin passes all 11 ordinary phases and its first eight sanitizer phases, then original FFI stops in its retained-dispatch prerequisite: assertions pass, followed by LSan 14,848 bytes in 232 libdispatch continuation allocations. I retain this terminal under `task_a810098e557045a88aa7b412953ce233`; later Darwin lifecycle/borrowed controls remain unreached. I do not infer a platform defect from allocator stacks.

Both archives retain source, compiler, provider, process-group and phase records. Actual executables/providers remain in the host qualification roots; these archives contain reports only. Installed/source-hidden qualification and full bootstrap remain separate obligations.

| Host | Reports | Archive SHA256 |
|---|---:|---|
| Linux | 84 | `b8db54ddd77db1e7e62de2e11fe1a7673d5d0c1c1771a5fd9ceabc03106a2135` |
| Darwin | 79 | `58d2c9f9c3c306f4dfda2834e7de83c0bf5668493bb3ab0bde81fa4b0a8c024b` |
