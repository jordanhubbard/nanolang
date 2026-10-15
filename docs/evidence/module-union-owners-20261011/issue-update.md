I continue #976 on release/5.1-completion-20261007 from 9c74deecb.

I bind module-owned union declarations, preserve original-name metadata ownership, resolve registered generic match owners and parse qualified ordinary/generic constructors. Both C routes pass 36 focused methods; the seven-object ASan/UBSan compiler run passes 37 methods including malformed constructors. C parser/checker/environment/metadata/transpiler gates pass.

The refreshed self-hosted seed exposes two remaining constructor-owner failures. I am correcting merged-source constructor recognition and retain cross-owner and same-owner duplicate refusal controls. Adjacent controls reproduce current self-hosted aggregate-payload and by-value cycle gaps, plus missing early C layout-cycle diagnostics. I have not closed the release scope.

I retain evidence under docs/evidence/module-union-owners-20261011. The coordination branch records my claimed source files; MAC is unavailable and not used.

I corrected the remaining merged-source constructor classification. The rebuilt seed passes all 38 focused native methods and all 15 union methods through explicit NanoISA/VM/ASan-UBSan C AOT execution. The eleven-stage compiler-product audit passed before that final parser correction; I preserve its exact-source boundary. Module-owned union acceptance now passes, while aggregate payload, cycle and final release/platform work stays open.
