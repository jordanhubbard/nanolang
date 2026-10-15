#include "nsi_file_binding.h"
#include "nsi_file_plan.h"

#define BindingPlan NlFileBindingPlan
#define binding_allocation_bound nl_file_binding_allocation_bound
#define binding_prepare nl_file_binding_prepare
#define binding_free nl_file_binding_free
#define binding_interface_bytes nl_file_binding_interface_bytes
#define binding_source_bytes nl_file_binding_source_bytes
#define binding_storage_size nl_file_binding_storage_size
#define binding_peak_bound nl_file_binding_peak_bound
#define binding_catalog_storage_size nl_file_plan_storage_size
static NlBindingStatus binding_catalog_validate(const NlNsi *nsi) {
    NlFilePlan *catalog = NULL;
    NlFilePlanStatus status = nl_file_plan_build(nsi, &catalog);
    nl_file_plan_free(catalog);
    return status == NL_FILE_PLAN_OK ? NL_BINDING_OK :
           status == NL_FILE_PLAN_MEMORY ? NL_BINDING_MEMORY : NL_BINDING_INVALID;
}
#include "nsi_binding_impl.inc"

/* I render the reviewed forward source surface, not an ordinary int wrapper.
 * Later paired parsing/lowering and actual shadows must qualify these bytes. */
static void fb_render_source(BindingWriter *w, const NlNsi *n) {
    fb_write(w, "# I declare catalog1; this declaration alone grants no host authority.\nservice ");
    fb_quote(w, n->iface.id);
    fb_write(w, " catalog 1 from \"interface.nsi.json\"\n\n");
    for (size_t i = 0; i < n->method_count; i++) {
        fb_write(w, "shadow "); fb_write(w, n->methods[i].name);
        fb_write(w, " {\n    let opened: OpenResult = (temp)\n    match opened {\n        Ok(file) => {\n            let mut owned: File = file\n");
        if (i == 1) {
            fb_write(w, "            let zero: WriteResult = (write_byte &mut owned 0)\n            match zero {\n                Ok(count) => { assert (== count 1) }\n                Error(error) => { assert false }\n            }\n");
        }
        if (i >= 1 && i <= 3) {
            fb_write(w, "            let written: WriteResult = (write_byte &mut owned ");
            fb_write(w, i == 3 ? "0" : "255");
            fb_write(w, ")\n            match written {\n                Ok(count) => { assert (== count 1) }\n                Error(error) => { assert false }\n            }\n");
        }
        if (i == 2 || i == 3) {
            fb_write(w, "            let positioned: PositionResult = (rewind &mut owned)\n            match positioned {\n                Ok() => {}\n                Error(error) => { assert false }\n            }\n            let read: ReadResult = (read_byte &mut owned)\n            match read {\n                Ok(octet) => {\n                    assert (== octet.value ");
            fb_write(w, i == 3 ? "0" : "255");
            fb_write(w, ")\n                    assert (not octet.eof)\n                }\n                Error(error) => { assert false }\n            }\n");
        }
        if (i == 3) {
            fb_write(w, "            let ended: ReadResult = (read_byte &mut owned)\n            match ended {\n                Ok(octet) => { assert octet.eof assert (== octet.value 0) }\n                Error(error) => { assert false }\n            }\n");
        }
        fb_write(w, "            let closed: CloseResult = (close owned)\n            match closed {\n                Ok() => {}\n                Error(error) => { assert false }\n            }\n        }\n        Error(error) => { assert false }\n    }\n}\n\n");
    }
}
