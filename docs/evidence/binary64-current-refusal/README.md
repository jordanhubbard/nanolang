# I distinguish admitted binary64 arithmetic from source refusal

My PR948 units-00 and units-02 histories fail the same old assertion that finite F64_ADD reconstruction must refuse. The canonical binary64 parser controls pass in units-02. I retain the original seventeen bit patterns and add verified finite-arithmetic reconstruction acceptance. My refusal control adds actual unsupported PRINT to the same valid module and still requires exact prior output bytes from both source writers.

My frozen source is `d260a0a16ab3c677f96d29b612f229b792cbe39f`. Fresh GCC ordinary and ASan/UBSan owning Make gates each pass all three methods; the original thirty-second subprocess limits remain unchanged. LSan is disabled in these two gates and is not claimed. I do not infer acceptance of every floating instruction.

The two lossless bundles retain thirty-two reports, exact before/after source and tool maps, command logs, terminals and surviving product inventories. Local immutable CAS indexes cover forty-one objects per configuration; the bundle and each member and CAS object were rehashed before publication. Temporary inner products removed by the original fixture are not reconstructed or claimed retained. The hosted first failures remain in the fixed PR948 raw-log collection and will be sealed with its complete terminal history.
