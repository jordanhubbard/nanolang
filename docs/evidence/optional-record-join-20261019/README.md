# I preserve late array-read shapes at record joins

I fix a native classifier defect exposed by my fresh `36930e1d3` compiler.
An unresolved array read joined with zero was unified with the exact scalar
alternate before its OPTIONAL shape became known. I now give that join its own
directed destination, and keep it when I adopt a later carrier kind. My exact
shape solver and runtime checked projections remain unchanged.

I retain the original full Stage1 failure and reduced source evidence. The
previously failing module SHA256 is
`f076f0fd825b969645e0a1f64683e78956dc778ee8fd8c7279ef069a0b152cf5`;
it now translates successfully. This is not a completed fresh bootstrap.

I pass 3,126 shape checks, nine scalar-join methods under ASan/UBSan,
11 neighboring record/string methods, and two new source methods (15.916s).
The new methods exercise six cases through three retained source producers,
verified VM and O0/O2 sanitized native execution. Present/empty/unchecked values
are 7/0/0. Missing raw reads still fail at checked consumers, and the unselected
missing-read branch is not executed. Provider hashes are retained; these are not
newly bootstrapped Stage1/Stage2 binaries.

My first neighbor run exposed a stale U8/INT refusal. The independently rebuilt
unchanged HEAD produces the same failure; `9b1ed4a26` had explicitly added this
carrier. I replace only that expectation with both-arm/both-order exact-tag and
value controls. Other heap/union/three-way refusal controls remain unchanged.

Full artifacts remain at
`/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-optional-record-join-mptyp6a3`.
I still require a fresh combined bootstrap and exact Linux/Darwin qualification.
