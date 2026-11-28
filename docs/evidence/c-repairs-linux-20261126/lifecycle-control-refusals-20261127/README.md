# My two pre-observation control refusals

I retain selected exact diagnostic entries from runs45982190971 and45990677237,
after checking each downloaded wrapper ZIP against its GitHub API digest.
`selection.json` has SHA-256
`c6deb2ce860a99e20bdeaa5cb9fddb5247959e2fcb9c0851cc160f3ee119503e`.
I retain the control streams, actual compiled native observer, build/source
hashes, exact observer source bytes, source-tree guard and pre-observation
identity/library snapshots. I omit the duplicated nested original-artifact ZIP
here. The complete downloaded wrappers remain on the mounted volume; this
selected receipt does not reconstruct or independently recheck their whole
ZIP digests. My reader checks the selected bytes and actual source against Git.

Run45982190971 compiles the observer and its C success control passes. The
marker occurs178.168ms after the pre-fork origin; NOTE_EXIT receipt follows
0.192ms after that marker. My Python assertion incorrectly compares pre-marker
time with a100ms exit-tail bound, so no compiler observation runs. I measure
the success100ms/late150ms distinction from the actual marker in fb31bad2dc.

Run45990677237 passes success and late C outcomes and their Python checks.
The blocked control reaches result3 after102.881ms from origin, without a
marker, then kills/reaps its owned child (status9). Its window began before
the child's20ms-delay marker. I retain the refusal; no compiler observation
runs. This diagnoses my control synchronization, not the original compiler
timeout. Neither run has completed-control acceptance or after-observation
guards, and neither passes native/emitter qualification.

```sh
python3 docs/evidence/c-repairs-linux-20261126/lifecycle-control-refusals-20261127/read_controls.py
```

My isolated diagnostic branch starts at fb31bad2dc so the restored original
C product-tree guard remains strict despite newer pure-helper additions on
the release branch. In5ca56494a9dc8eb84c66da61402343a2a34994f2 I arm observation
before releasing the owned child, then start the window after its marker.
I strengthen armed/marker/trigger ordering and blocked-duration checks.
Success100ms, late150ms, blocked100ms, command5s and product300/10 remain.
Native validation is pending in
[run45991837855](https://github.com/jordanhubbard/nanolang/actions/runs/45991837855).
I count no retry or diagnostic as a passing original qualification gate.
