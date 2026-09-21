# My tuple checker cleanup boundary

I retain the first e456006ad Puck sanitizer failure under MAC
`task_633d96d841b1df945dd2bcd3199e46bc`. Apple ordinary controls pass; the
Homebrew identity fixture completes its assertions and then LeakSanitizer
reports 746 bytes in 20 allocations. I do not replay that unfixed binary.

The retained allocation stacks identify two boundaries. The manually built
field AST in my fixture owns its final resolved_type_info cache, but no heap AST
destructor visits that stack node. I explicitly destroy that final cache. A
second payload-field node must start independently instead of shallow-copying
an already cached field node.

The Environment destructor omits StructDef field_type_names (each string and
the vector) and field_element_types. My two checker collectors and nanovirt's
register_imported_struct allocate independent copies before env_define_struct.
These are Environment-owned arrays just like field_names and field_types.
I reclaim them exactly once at Environment teardown, including NULL and
zero-field allocation cases. I leave field_type_info and module_name borrowed.
The apparent borrowed assignments in nanovirt are a different CgStructDef
representation; they do not overwrite Environment StructDef arrays. The
resource-classification stack records are never destroyed as heap Environments.

I require the original parsed tuple/constructor, assignment-growth and all
allocation assertions unchanged. An actual hooked Environment control must
show owned auxiliary vectors reclaimed while a borrowed complete child
annotation remains untouched. Corrected source review precedes execution.
I keep this repair based on e456, separate from unqualified SDK emitter source.
