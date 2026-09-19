# I preserve legacy OpenCL buffer identity

I implement task_87ca2c78f1cd1e8a56230beaaae38e6c under d03c. This is my
preimplementation contract at canonical88e7dc7469787e5c4bbbdb437ef9df03a8feb367.
My separate [static audit](OPENCL_BUFFER_IDENTITY_AUDIT.md) retains the original
finding. I have performed no reproduction, driver initialization, device query,
allocation, kernel execution or fixture execution for this repair.

## My bounded compatibility decision

I preserve the public int64 allocation/copy/free/launch signatures and zero
allocation-failure result. CUDA selection and raw CUDA pointer handling stay
unchanged. Only the selected OpenCL runtime interprets my OpenCL tokens. Tokens
remain process-local opaque values: I promise neither persisted-token migration
nor compatibility with tokens issued by another runtime instance.

I reserve the positive integer prefix 0x0C1A in bits63..48 for OpenCL buffer
arguments. Bits47..8 encode a nonzero 40-bit generation, and bits7..0 encode one
of 256 stable slots. I construct/decode using uint64_t and convert only values
known to fit int64_t. Each slot starts at generation1; successful retirement
permits its next generation, never wrap. At generation 0xffffffffff I permanently
exhaust that slot after retirement. Generation0, including old positional tokens,
is invalid in the repaired runtime. Exhaustion/capacity refusal happens before
calling clCreateBuffer. I do not move live entries during retirement.

This explicitly corrects the old comment claiming sentinel integers cannot
collide with scalar integers. In the untyped legacy kernel-argument ABI, an exact
live token means buffer; any other reserved-prefix value is rejected rather than
passed as a scalar. Integers outside that prefix remain scalars. The reserved
scalar interval is a documented compatibility restriction, not a new typed ABI
or an unforgeable capability claim. A future tagged argument interface remains
separate; this repair does not claim to disambiguate an integer equal to a live
token.

## My registry and release obligations

I keep a fixed 256-entry table with explicit FREE/LIVE/QUARANTINED/EXHAUSTED
state, generation, full issued token, cl_mem and allocation size. Lookup requires
reserved prefix, nonzero generation, LIVE state and full token equality. No
consumer may resolve merely by position or accept a retired identity. My runtime
remains subject to its existing serialized-call assumption; I add no thread
safety claim.

I publish a LIVE entry only after successful clCreateBuffer with nonnull object.
I preserve allocation errors. If an allocation reports an error with an object,
I make one checked release attempt before returning zero; an uncertain release
keeps the raw object and error in its reserved QUARANTINED slot. It cannot become
available for another allocation. I do not forget or retry uncertain objects.

For an accepted free I invalidate the public identity before one
clReleaseMemObject call. CL_SUCCESS records successful release of my reference,
not proof that every queued command/reference has ceased to exist. An error
records an unknown release outcome and keeps the slot/object quarantined; repeated
free does not retry. The void free ABI reports failure through existing last-error
text. Null free stays harmless; other unknown/stale tokens report refusal without
a driver call. I retain bounded per-slot release outcome/error accounting for
inspection in ordinary tests, not a new public service interface. With no legacy
shutdown API I retain quarantine until process exit and make no driver cleanup
claim for it. At most256 slots can be retained; capacity fails closed.

## I update every identity consumer together

My inventory is modules/gpu/opencl_runtime.c: allocation publication, free,
blocking host-to-device and device-to-host copy, and ocl_set_args shared by
all OpenCL launch wrappers. Copy requires exact live identity and requested byte
count within the stored allocation, in addition to existing host-array checks.
Unknown identities and excess size refuse before enqueue. I do not promise
atomic host/device bytes after an actual driver transfer failure.

Kernel argument classification distinguishes live buffer, nonreserved scalar,
and invalid reserved token. I validate all argument identities before calling
clSetKernelArg, so an invalid later argument does not partially update arguments.
Actual clSetKernelArg failure can leave partial driver state; no launch follows
failure and each future accepted call supplies its complete argument list. I do
not change kernel caching, scheduling, language imports, integer CUDA ABI,
private nsi_gpu adapter, File transport or public capability admission.

## My ordered acceptance gates

1. I obtain contract review, implement the bounded registry/consumer changes,
   and obtain complete production review before compilation or fixtures.
2. I freeze source, harness and compiler identities. Strict GCC/Clang ordinary
   lifecycle controls check stable live identities, full-token matching, bounded
   reuse and terminal generation exhaustion, zero/capacity behavior, consumer
   refusal before driver calls, scalar/reserved classification, every release
   outcome and allocation rollback. Instrumented interfaces are host-model
   evidence only; no known failed legacy artifact is executed.
3. I qualify fresh ordinary allocate/copy/kernel/free behavior on an actual
   OpenCL GPU with platform/device/type/driver/library identity recorded. Since
   automatic selection prefers CUDA and OpenCL selection may choose CPU, I
   require an isolated explicit OpenCL test selection and a positively identified
   GPU device; production selection policy stays unchanged. CPU OpenCL or a
   simulated driver cannot satisfy this gate. I preserve unavailable-device
   evidence rather than label private CUDA success as OpenCL acceptance.
4. I run affected existing unified GPU/array ABI controls and preserve first
   terminal failures, source/tool manifests and platform scope. Actual GPU
   availability/platform coverage and remaining d03c public-service obligations
   are stated explicitly. I close only the measured bounded repair after review
   and canonical merge; absent real GPU acceptance this task stays open.

## I require the cached kernel's complete argument list

Review of production350c43138 finds an unmet contract obligation: my cache reuses
kernels by path/name, while ocl_set_args checks only the supplied list. Its bound
of five does not establish the kernel's actual argument count. Therefore my
claim that every accepted call replaces every argument was not yet implemented.
This is a static source finding, with no reproduction or driver operation.

Before correcting production I add required dynamic symbol clGetKernelInfo:
`cl_int (*)(cl_kernel, cl_uint, size_t, void *, size_t *)` on the currently
qualified Linux/Darwin ABI. The selector is CL_KERNEL_NUM_ARGS (0x1191); its
result is cl_uint. I checked the Khronos [OpenCL header](https://raw.githubusercontent.com/KhronosGroup/OpenCL-Headers/main/CL/cl.h),
which defines cl_kernel_info as cl_uint and declares this OpenCL1.0 query.
Windows calling-convention qualification remains separate from existing
Linux/Darwin scope; I do not infer it from this abbreviated typedef.

I load the symbol as mandatory using my existing checked loader. For every
argument-setting call, after checking my local argc bounds but before any
clSetKernelArg, I query the actual kernel with sizeof(cl_uint), an initialized
result and returned-size output. Query error, wrong returned size or count
unequal to argc refuses with a diagnostic and no argument updates or enqueue.
I do not infer count from cache history or permit partial lists. Then I validate
all token identities, set exactly the complete list and enqueue only on complete
success. This preserves the serialized-call precondition and permits recovery
from earlier partial Set failures without relying on retained driver arguments.

My fresh host controls will cover symbol absence, query error/size/count refusal,
exact count acceptance and absence of Set/enqueue calls after refusal. Ordinary
actual GPU controls will qualify the queried signature alongside complete kernel
argument updates. No old failed artifact or defect reproduction is required.
The initial local header search found no installed CL/cl.h; a separate urllib
fetch returned HTTP404, so I claim web-header inspection, not a sealed downloaded
header or measured ABI acceptance. Production and execution remain held for
review of this addition.

## I preserve my first compile refusal and check source-reader publication

At frozen634595907, my first gate stopped in strict GCC compilation: two
existing unchecked fread results and one possibly truncated path diagnostic in
the unified runtime. No fixture or driver operation ran. The before/after source
and tool maps agree; the compiler log SHA256 is
c3d009d078fb100681a6822b40bcaeeb0fef423390d28b263ff1d5c98084ea84,
retained under /tmp/nanolang-opencl-first-634595907. Static review finds the
adjacent CUDA-only reader also discards fread. I track this prerequisite as
task_f13e8a01c0e4466b8aa4e523c17bac96 before correction.

I propose to check both seeks, nonnegative tell and size+terminator
representability before allocation; require the exact read count, no stream
error and successful close before handing source bytes to any driver. On refusal
I free allocated bytes and close the file exactly once. I bound paths in
first-person diagnostics without disabling warnings. This covers only the
three existing PTX/OpenCL source readers in unified and CUDA-only runtimes;
valid source bytes, backend selection, cache semantics and public ABI remain
unchanged. Fresh strict compilation and ordinary host reader publication/cleanup
controls qualify the repair after review. I do not rerun the old failed artifact
or claim a device outcome from the compile failure.
