#include "nsi_socket_binding.h"
#include "nsi_socket_plan.h"

#define BindingPlan NlSocketBindingPlan
#define binding_allocation_bound nl_socket_binding_allocation_bound
#define binding_prepare nl_socket_binding_prepare
#define binding_free nl_socket_binding_free
#define binding_interface_bytes nl_socket_binding_interface_bytes
#define binding_source_bytes nl_socket_binding_source_bytes
#define binding_storage_size nl_socket_binding_storage_size
#define binding_peak_bound nl_socket_binding_peak_bound
#define binding_catalog_storage_size nl_socket_plan_storage_size
static NlBindingStatus binding_catalog_validate(const NlNsi *nsi) {
    NlSocketPlan *catalog = NULL;
    NlSocketPlanStatus status = nl_socket_plan_build(nsi, &catalog);
    nl_socket_plan_free(catalog);
    return status == NL_SOCKET_PLAN_OK ? NL_BINDING_OK :
           status == NL_SOCKET_PLAN_MEMORY ? NL_BINDING_MEMORY : NL_BINDING_INVALID;
}
#include "nsi_binding_impl.inc"

/* I emit only the declaration here. Real network shadows require a selected
 * peer fixture and belong to paired compiler/runtime qualification. */
static void fb_render_source(BindingWriter *w, const NlNsi *n) {
    fb_write(w, "# I declare catalog1; this declaration alone grants no host authority.\nservice ");
    fb_quote(w, n->iface.id);
    fb_write(w, " catalog 1 from \"interface.nsi.json\"\n");
}
