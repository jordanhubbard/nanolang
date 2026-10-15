#include "nsi_socket_binding.h"
#define BindingPlan NlSocketBindingPlan
#define binding_interface_bytes nl_socket_binding_interface_bytes
#define binding_source_bytes nl_socket_binding_source_bytes
#define binding_peak_bound nl_socket_binding_peak_bound
#define binding_storage_size nl_socket_binding_storage_size
#define binding_prepare nl_socket_binding_prepare
#define binding_free nl_socket_binding_free
#define binding_allocation_bound nl_socket_binding_allocation_bound
#define BINDING_INTERFACE "nsi:nanolang/net"
#define BINDING_NAME "net"
#define BINDING_LABEL "Socket"
#include "nsi_binding_cases.inc"
