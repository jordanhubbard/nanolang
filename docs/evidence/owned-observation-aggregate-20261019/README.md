# I qualify integrated observation pretests

At `647250aca` I pass all 13 observation/loop/shared reconstruction methods in 107.648 seconds through NanoVirt and both fresh `74a565150` stages. All tool/runtime/stage hashes remain unchanged from mixed qualification; I pin the changed reconstruction script separately.

Actual source-produced modules pass original verification/execution, C and Nano reconstruction, regenerated verification/execution, exact layout/signature checks and O0/O2 ASan/UBSan native checks. The supplemental 16-read counter measures generated C only. Inverted exits are separately verified before reconstruction; malformed module refusal and direct-facts defensive controls are distinguished. Existing pure loops and shared CALL_REF paths remain green.

I add the new module to `test-owned-reconstruction`. This is Darwin evidence, not full #999 closure or hosted platform qualification. Arithmetic pretests remain deliberately refused pending their separate exact wrapping-operation contract. Log copies normalize trailing display whitespace and retain original hashes.
