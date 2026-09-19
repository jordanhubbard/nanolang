# I record an unpreserved OpenCL buffer identity invariant

I track `task_87ca2c78f1cd1e8a56230beaaae38e6c` under d03c. This is a static
finding at canonical `1cbe8c6f4c8dd2130ca44eb89e2d1f62594d32f4`.
The audited `modules/gpu/opencl_runtime.c` SHA256 is `395b8f4d74af29a5aeb227dc9eee920fe8faa2e42162c5517e62d254e3250cbe`.
I have not loaded a driver, performed GPU operations, run fixtures or constructed
a reproduction. This record contains no input sequence or crash artifact.

My source facts are:

- Lines359-363 decode a positional index from the token and select that current
  table position. Lookup does not compare the stored full identity.
- Lines580-584 publish a token derived from the allocation's current table index.
- Lines594-596 retire by copying a surviving table entry into another position;
  its previously issued token is not changed.

The positional relation used by lookup is therefore not an invariant of table
compaction. Stored identity and physical index can diverge, and reused indices
carry no generation distinction. This is a static identity finding, not measured
runtime impact. I do not derive a security claim from it.

Before repair I require a separately reviewed stable identity/retirement contract,
checked release errors and ordinary non-crashing lifecycle controls. Existing
integer ABI compatibility, all lookup/launch/free consumers, capacity reuse and
unsupported identities need explicit decisions. Driver-stub tests qualify only
those static interfaces; actual GPU resource lifetime remains separate evidence.
I do not change legacy code or broaden service admission in this record.
