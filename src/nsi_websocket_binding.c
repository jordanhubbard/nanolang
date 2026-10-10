#include "nsi_websocket_binding.h"
#include "nsi_websocket_plan.h"

#define BindingPlan NlWebSocketBindingPlan
#define binding_allocation_bound nl_websocket_binding_allocation_bound
#define binding_prepare nl_websocket_binding_prepare
#define binding_free nl_websocket_binding_free
#define binding_interface_bytes nl_websocket_binding_interface_bytes
#define binding_source_bytes nl_websocket_binding_source_bytes
#define binding_storage_size nl_websocket_binding_storage_size
#define binding_peak_bound nl_websocket_binding_peak_bound
#define binding_catalog_storage_size nl_websocket_plan_storage_size
static NlBindingStatus binding_catalog_validate(const NlNsi *nsi) {
    NlWebSocketPlan *catalog = NULL;
    NlWebSocketPlanStatus status = nl_websocket_plan_build(nsi, &catalog);
    nl_websocket_plan_free(catalog);
    return status == NL_WEBSOCKET_PLAN_OK ? NL_BINDING_OK :
           status == NL_WEBSOCKET_PLAN_MEMORY ? NL_BINDING_MEMORY : NL_BINDING_INVALID;
}
#include "nsi_binding_impl.inc"

/* I emit only the declaration here. Real network shadows require a selected
 * peer fixture and belong to paired compiler/runtime qualification. */
static void fb_render_source(BindingWriter *w, const NlNsi *n) {
    fb_write(w, "# I declare catalog1; this declaration alone grants no host authority.\nservice ");
    fb_quote(w, n->iface.id);
    fb_write(w, " catalog 1 from \"interface.nsi.json\"\n");
}
