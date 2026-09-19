# My owner ARRAY activation integration with pending File operations

I integrate canonical PR841 `fd56e78a9dc11f7b951cacad7a51faabbb1f3c42`
into my separately qualified public activation branch. I preserve the frozen
c510 runtime corpus and sourcec9fb tree. This amendment precedes guard edits.

My new canonical `nvm_service_execution_pending` combines service declarations
with decoded File operations in every declared function. Bare File operations
remain pending even without a service section. I use that predicate before both
my owner ARRAY candidate scan and public admission wrapper. A pending module
returns NOT_SELECTED from the classifier and UNRESOLVED from direct admission,
without changing caller output. This does not grant fallback: the common verifier,
VM readiness and translator guards independently refuse pending service execution.

I resolve the two VM conflicts by retaining canonical service_execution_pending
first and then my owner ARRAY selection/admission branches. I preserve both Make
fixture targets and both roadmap entries. Auto-merged verifier and native entry
checks already call service_execution_pending before owner selection. My existing
private descriptor/origin queries acquire the canonical pending guard unchanged.

I preserve v2 private File transport semantics. Both converter directions already
call their service validation before owner selection; bare File opcodes without
nominal declarations refuse there. Nominal declared private File transport keeps
its existing path and grants no execution. I do not replace those transport
validators with a blanket execution refusal or change ordinary immediate bytes
that happen to equal File opcodes.

I audit normal VM initialization/readiness, synchronous invocation, direct core,
linked modules, native output and closed backends after the merge. No File
handler executes in these gates. I add an owner ARRAY fixture with a bare File
operation in the root and then an otherwise uncalled helper, requiring service
priority, direct-admission sentinel preservation and public refusal. I retain
canonical File opcode/ordinary-immediate/provider controls and run bounded combined
ordinary and sanitizer public owner ARRAY, authority and File opcode gates on
fresh integrated providers after independent review. No paired source producer
or File execution admission follows from this integration.
