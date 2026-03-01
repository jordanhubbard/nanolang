\ test_stack.fs — Stack manipulation tests

testing dup
T{ 1 dup -> 1 1 }T
T{ 0 dup -> 0 0 }T
T{ -1 dup -> -1 -1 }T

testing drop
T{ 1 2 drop -> 1 }T
T{ 0 drop -> }T

testing swap
T{ 1 2 swap -> 2 1 }T
T{ 0 1 swap -> 1 0 }T

testing over
T{ 1 2 over -> 1 2 1 }T
T{ 0 0 over -> 0 0 0 }T

testing rot
T{ 1 2 3 rot -> 2 3 1 }T
T{ 0 1 2 rot -> 1 2 0 }T

testing -rot
T{ 1 2 3 -rot -> 3 1 2 }T

testing ?dup
T{ 0 ?dup -> 0 }T
T{ 1 ?dup -> 1 1 }T
T{ -1 ?dup -> -1 -1 }T

testing nip
T{ 1 2 nip -> 2 }T
T{ 0 1 nip -> 1 }T

testing tuck
T{ 1 2 tuck -> 2 1 2 }T

testing 2dup
T{ 1 2 2dup -> 1 2 1 2 }T
T{ 0 0 2dup -> 0 0 0 0 }T

testing 2drop
T{ 1 2 3 2drop -> 1 }T
T{ 1 2 2drop -> }T

testing 2swap
T{ 1 2 3 4 2swap -> 3 4 1 2 }T

testing 2over
T{ 1 2 3 4 2over -> 1 2 3 4 1 2 }T

testing pick
T{ 0 1 2 0 pick -> 0 1 2 2 }T
T{ 0 1 2 1 pick -> 0 1 2 1 }T
T{ 0 1 2 2 pick -> 0 1 2 0 }T

testing roll
T{ 1 2 3 4 0 roll -> 1 2 3 4 }T
T{ 1 2 3 4 1 roll -> 1 2 4 3 }T
T{ 1 2 3 4 2 roll -> 1 3 4 2 }T
T{ 1 2 3 4 3 roll -> 2 3 4 1 }T

testing depth
T{ depth -> 0 }T
T{ 1 depth -> 1 1 }T
T{ 1 2 depth -> 1 2 2 }T

test-summary
