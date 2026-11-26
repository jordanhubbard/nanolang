# My publication worktree storage backing

I apply the superseding Sparky storage instruction from2026-11-26. I verify
`$HOME/.local/share/fleet-migration/puck-volume` is mounted before using it.

I copy my5.9GiB publication worktree, including its189544 files and ignored
content, to `nanolang/worktrees/darwin-publication-qkk55gi3` on that volume.
An independent `rsync -acni --delete` comparison exits0 with no differences.
I quiesce use of the old checkout before switching its directory entry.

My logical path remains `/var/tmp/nanolang-darwin-publication-qkk55gi3`, now a
symlink to the shared copy. The common Git store and worktree Gitfile remain
unchanged. Before and after the switch, HEAD is
`cdbb2dfafb9f8131abfb5f332814fab82a755edf`, the pending status contains only my
roadmap change, and its binary diff SHA256 remains
`9f0274487f25469954ea2de795e9ab406da2d388289fe9058ef1dea710533be9`.
The recorded commit and status describe the cutover, not future branch state.

I retain the original copy through post-cutover reader checks and successful
publication9aafa84d91. I then remove only that verified duplicate, after checking
for active use, nested mounts and correct Git registration. My reclamation receipt
records6236938240 allocated bytes removed and6236897280 bytes of observed root
headroom gained; other filesystem activity remains possible. I preserve other
owners' worktrees and Puck originals. Further large outputs, temporary files and caches use the
mounted volume. This storage receipt adds no requirement to my69-entry release
contract and establishes no compiler or whole-release acceptance.
