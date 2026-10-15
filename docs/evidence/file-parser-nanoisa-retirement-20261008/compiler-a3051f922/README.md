# I qualify the compiler after native local-clear correction

My unchanged `a3051f922301aefae2b7fb848903e33e0fa981bc` source runs
`make -j2 test-one-ir-compiler` to exit zero in 877.544 seconds.
All 109 methods pass in 874.732 seconds. My runner confirms clean tracked
source and the unchanged sole user-owned untracked test by SHA-256.

This pin includes list insertion and the local-clear correction. It predates
my isolated array-slice and list-removal/pop work. I do not use this gate to
qualify those later changes or claim complete File parser or 5.1 acceptance.
