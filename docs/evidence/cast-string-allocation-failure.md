# My CAST_STRING allocation-result contract

I track task_822750774e0040298e8297e96145f3d5 after the U8 conversion repair.
My integer, float, boolean and default empty-string allocation helpers can
return NULL. I report VM_ERR_MEMORY rather than pushing a TAG_STRING value
with a null allocation. Successful formatting and string-identity conversion
remain unchanged. My existing checked U8 arm retains its behavior.

For default conversion I release the consumed non-string value before asking
for the empty string, including on failure. A failed ordinary invocation
unwinds its frame and operands; the caller's independently held references
remain valid. I test these boundaries using isolated heap allocation hooks,
then disable injection and check successful conversion and complete cleanup.
String identity conversion must not allocate. I do not replay a historical
product abort or attribute one to this static finding.

## My measured acceptance

My base is merged U8 formatting `c3166cd8`; contract `f3841ead` precedes the
four defensive result checks and allocation harness at `a47bec64`.

I pass 91 isolated allocation/cleanup checks with integer, float, both boolean
values, U8, void, a caller-owned array and string identity. I inject failure
only into heap malloc after VM and input setup. Every allocating case reports
VM_ERR_MEMORY, removes temporary frames/operands, and retains the caller's
input. Disabling injection permits a subsequent successful invocation. A
string identity conversion performs no heap allocation. Releasing results and
inputs restores the initialized VM heap baseline.

The ordinary target passes (`/tmp/nanolang-cast-string-allocation-r2.log`).
The same 91 checks pass with VM execution, heap allocation and the new harness
instrumented by ASan/UBSan and leak detection
(`/tmp/nanolang-cast-string-sanitized.log`); other linked objects are ordinary
builds. The existing VM suite passes 274,416 checks
(`/tmp/nanolang-cast-string-allocation-r1.log`).

The first harness run incorrectly expected a completely empty heap after
releasing the input; VM initialization retains module strings. I preserve
that test failure and compare against the measured initialized baseline in
the corrected test. It is not a demonstrated product leak. I did not reproduce
historical compiler aborts or run a full compiler acceptance for this slice.
