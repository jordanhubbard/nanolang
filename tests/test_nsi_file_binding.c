#include "nsi_file_binding.h"
#define BindingPlan NlFileBindingPlan
#define binding_interface_bytes nl_file_binding_interface_bytes
#define binding_source_bytes nl_file_binding_source_bytes
#define binding_peak_bound nl_file_binding_peak_bound
#define binding_storage_size nl_file_binding_storage_size
#define binding_prepare nl_file_binding_prepare
#define binding_free nl_file_binding_free
#define binding_allocation_bound nl_file_binding_allocation_bound
#define BINDING_INTERFACE "nsi:nanolang/filesystem"
#define BINDING_NAME "filesystem"
#define BINDING_LABEL "File"
#include "nsi_binding_cases.inc"
