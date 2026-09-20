; I exercise exactly the native revision1 callback ABI, not an emitted module.
declare i32 @npr_file_read(ptr, ptr, i32, ptr, i32, ptr)
define i32 @pr_llvm_read(ptr %context, ptr %path, i32 %length,
                       ptr %destination, i32 %capacity, ptr %length_out) {
entry:
  %status = call i32 @npr_file_read(ptr %context, ptr %path, i32 %length,
                                 ptr %destination, i32 %capacity, ptr %length_out)
  ret i32 %status
}
