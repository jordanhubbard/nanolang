#if NANO_CAPTURE_CPP_PROBE != 17 || NANO_CAPTURE_C_PROBE != 25
#error I require caller preprocessing and compiler flags
#endif
const int nano_capture_flag_probe = NANO_CAPTURE_CPP_PROBE + NANO_CAPTURE_C_PROBE;
