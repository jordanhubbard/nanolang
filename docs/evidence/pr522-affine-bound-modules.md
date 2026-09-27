# My PR522 affine bound-module qualification

I qualify only the mixed bound-module slice at exact source
`1e35f611d0f74c61452dd3fb545bc5c78104fe3b`. I do not claim the scalar-union
sibling, the complete held product, a new fixed point, another platform, or a
release.

My retained first terminals distinguish each dependency:

- `6c62d17c` passes the first five module-identity routes and stops on the
  qualified ordinary call before call-result lowering. The focused log is
  `a23a411993ada4c021d9e84b74f38e24a3c02f100f7b1babe6a03ff0308342ed`.
- `582ca52e` advances through that call and stops in NanoVM's mixed return
  boundary. The focused log is
  `96b85f464c1cc11495dde3a9a8ef8436c2b76f97d7919d0ec93a6c9f407bc9b3`.
- `b6af7b9f` exposes the stale synthetic ownership-row offset before the new
  result case. The runtime log is
  `b42bbd6d2673062be83f950d0033514da904a2007190927739c166e43a9bd111`.
- `c752fccc` restores the prior cases and shows that my first new fixture used
  the separately unsupported pending FLOAT-array result profile. The runtime
  log is
  `ad5d698cef09b7342e0424e2a8547f7f34cb7aa5758cfe0ea63b9cdeb58d5d17`.
- `f727fb5d` passes all 13 C lifecycle cases and 2,157 checks, then the Python
  wrapper retains its old exact count of 12. The runtime log is
  `d85ec928d47c54adb043860b9e04e8d1f458566de98d3f110f030cd24c971627`.

At the final source I use two clean, isolated Linux checkouts:

- `make -j8 test-mixed-samples-runtime` passes in 40.69 seconds. It executes
  every case through four public VM APIs and direct core, regenerates exact C,
  runs `-O0` and `-O2` ASan/UBSan native programs, sweeps allocation failures,
  and proves zero retained test allocations. Log SHA-256:
  `3dad5c85045247b93c1aac20a6af746b1048af249d10a81da15ea73aff6b1cb4`.
- A separate fresh `make -j8 bootstrap` passes in 4:39.83 with Stage1, Stage2,
  hello, installed/no-C-seed and mandatory shadows. Peak RSS is 6,570,628 KiB.
  Log SHA-256:
  `108fc3b30ec1ff1d7fc0bee0ffcdfaf30fbd1e1900a566c8a9ee53b17f5fd5cf`.
- `python3 -m unittest -f -v tests.test_affine_module_identity` then passes all
  nine methods in 19.139 seconds against C-seed, Stage1 and Stage2. This includes
  import-order reversal, long paths, nested owner transport, qualified
  same-spelled ordinary records, foreign collision, duplicate and unresolved
  refusals, and prior-output preservation. Log SHA-256:
  `771ba5522d1da314a861152e0d6c2239f9b147e513d7eddd0d9bb8f21513a7ac`.

Both final source trees remain clean at the exact tested commit. My earlier
nested shape query also passes 351 base checks and 114,816 composed checks at
`6c62d17c`; its retained log is
`2edfe7d2988518d1f0d4732357366a740a5656d28cb11767413e120bcd000595`.
