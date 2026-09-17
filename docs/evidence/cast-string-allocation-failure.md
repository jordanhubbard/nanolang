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
