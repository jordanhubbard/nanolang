/* I exercise the production configuration with no fault-injection hooks. */
#include "../../src/nanoisa/managed_strings.h"
int nms_product_smoke(void) {
    static const unsigned char bytes[] = {0,1,255};
    NmsRuntime runtime;
    nms_init(&runtime, NULL, 0);
    NmsHandle value;
    NmsView view;
    if (nms_begin(&runtime) != NMS_OK ||
        nms_create(&runtime, bytes, sizeof bytes, &value) != NMS_OK ||
        nms_retain(&runtime, value) != NMS_OK ||
        nms_release(&runtime, value) != NMS_OK ||
        nms_view(&runtime, value, &view) != NMS_OK || view.length != 3 ||
        view.data[0] || view.data[1] != 1 || view.data[2] != 255 ||
        nms_release(&runtime, value) != NMS_OK || runtime.live_objects ||
        nms_finish(&runtime, NMS_OK, 0) != 0 ||
        nms_dispose(&runtime) != NMS_OK) return 1;
    return 0;
}
#ifndef __wasm32__
int main(void) { return nms_product_smoke(); }
#endif
