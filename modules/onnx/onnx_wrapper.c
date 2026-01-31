/*
 * ONNX Runtime Wrapper for nanolang
 * 
 * Provides simple CPU-based neural network inference using ONNX Runtime.
 * 
 * Installation (macOS):
 *   brew install onnxruntime
 * 
 * Installation (Linux):
 *   sudo apt-get install libonnxruntime-dev
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <onnxruntime_c_api.h>

/* ONNX Runtime API */
static const OrtApi* g_ort = NULL;

/* Session struct to hold model and related data */
typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    size_t num_input_nodes;
    size_t num_output_nodes;
    char** input_names;
    char** output_names;
    int64_t* input_dims;
    int64_t* output_dims;
    size_t input_rank;
    size_t output_rank;
} ONNXSession;

/* Initialize ONNX Runtime (call once at startup) */
int64_t onnx_init(void) {
    if (g_ort != NULL) {
        return 0; /* Already initialized */
    }
    
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "ONNX: Failed to init ONNX Runtime API\n");
        return -1;
    }
    
    return 0;
}

/* Load an ONNX model from file */
int64_t onnx_load_model(const char* model_path) {
    if (onnx_init() < 0) {
        return -1;
    }
    
    ONNXSession* sess = (ONNXSession*)calloc(1, sizeof(ONNXSession));
    if (!sess) {
        fprintf(stderr, "ONNX: Failed to allocate session\n");
        return -1;
    }
    
    /* Create environment */
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "nanolang", &sess->env);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to create environment: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        free(sess);
        return -1;
    }
    
    /* Create session options */
    status = g_ort->CreateSessionOptions(&sess->session_options);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to create session options: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(sess->env);
        free(sess);
        return -1;
    }
    
    /* Set to use CPU only */
    g_ort->SetIntraOpNumThreads(sess->session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(sess->session_options, ORT_ENABLE_BASIC);
    
    /* Create session */
    status = g_ort->CreateSession(sess->env, model_path, sess->session_options, &sess->session);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to create session: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(sess->session_options);
        g_ort->ReleaseEnv(sess->env);
        free(sess);
        return -1;
    }
    
    /* Get input/output counts */
    OrtAllocator* allocator;
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get allocator: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSession(sess->session);
        g_ort->ReleaseSessionOptions(sess->session_options);
        g_ort->ReleaseEnv(sess->env);
        free(sess);
        return -1;
    }
    
    g_ort->SessionGetInputCount(sess->session, &sess->num_input_nodes);
    g_ort->SessionGetOutputCount(sess->session, &sess->num_output_nodes);
    
    /* For simplicity, we only support single input/output models */
    if (sess->num_input_nodes != 1 || sess->num_output_nodes != 1) {
        fprintf(stderr, "ONNX: Model must have exactly 1 input and 1 output (has %zu inputs, %zu outputs)\n",
                sess->num_input_nodes, sess->num_output_nodes);
        g_ort->ReleaseSession(sess->session);
        g_ort->ReleaseSessionOptions(sess->session_options);
        g_ort->ReleaseEnv(sess->env);
        free(sess);
        return -1;
    }
    
    /* Get input name */
    char* input_name;
    status = g_ort->SessionGetInputName(sess->session, 0, allocator, &input_name);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get input name: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSession(sess->session);
        g_ort->ReleaseSessionOptions(sess->session_options);
        g_ort->ReleaseEnv(sess->env);
        free(sess);
        return -1;
    }
    
    sess->input_names = (char**)malloc(sizeof(char*));
    sess->input_names[0] = input_name;
    
    /* Get output name */
    char* output_name;
    status = g_ort->SessionGetOutputName(sess->session, 0, allocator, &output_name);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get output name: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        allocator->Free(allocator, input_name);
        free(sess->input_names);
        g_ort->ReleaseSession(sess->session);
        g_ort->ReleaseSessionOptions(sess->session_options);
        g_ort->ReleaseEnv(sess->env);
        free(sess);
        return -1;
    }
    
    sess->output_names = (char**)malloc(sizeof(char*));
    sess->output_names[0] = output_name;
    
    /* Return session handle as pointer cast to int64_t */
    return (int64_t)sess;
}

/* Free a model session */
void onnx_free_model(int64_t model_handle) {
    if (model_handle == 0) {
        return;
    }
    
    ONNXSession* sess = (ONNXSession*)model_handle;
    
    /* Free names */
    if (sess->input_names) {
        OrtAllocator* allocator;
        g_ort->GetAllocatorWithDefaultOptions(&allocator);
        if (sess->input_names[0]) {
            allocator->Free(allocator, sess->input_names[0]);
        }
        free(sess->input_names);
    }
    
    if (sess->output_names) {
        OrtAllocator* allocator;
        g_ort->GetAllocatorWithDefaultOptions(&allocator);
        if (sess->output_names[0]) {
            allocator->Free(allocator, sess->output_names[0]);
        }
        free(sess->output_names);
    }
    
    /* Free dims */
    if (sess->input_dims) {
        free(sess->input_dims);
    }
    if (sess->output_dims) {
        free(sess->output_dims);
    }
    
    /* Release ONNX objects */
    if (sess->session) {
        g_ort->ReleaseSession(sess->session);
    }
    if (sess->session_options) {
        g_ort->ReleaseSessionOptions(sess->session_options);
    }
    if (sess->env) {
        g_ort->ReleaseEnv(sess->env);
    }
    
    free(sess);
}

/* Run inference on a model
 * 
 * Parameters:
 *   model_handle: Handle returned from onnx_load_model
 *   input_data: Pointer to input data array (float*)
 *   input_size: Number of elements in input
 *   input_shape: Pointer to shape array (int64_t*)
 *   input_rank: Number of dimensions in shape
 *   output_data: Pointer to output data array (float*) - must be pre-allocated
 *   output_size: Number of elements in output
 * 
 * Returns: 0 on success, -1 on failure
 */
int64_t onnx_run_inference(
    int64_t model_handle,
    const double* input_data,
    int64_t input_size,
    const int64_t* input_shape,
    int64_t input_rank,
    double* output_data,
    int64_t output_size
) {
    if (model_handle == 0 || !input_data || !input_shape || !output_data) {
        fprintf(stderr, "ONNX: Invalid parameters\n");
        return -1;
    }
    
    ONNXSession* sess = (ONNXSession*)model_handle;
    
    /* Create input tensor */
    OrtMemoryInfo* memory_info;
    OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to create memory info: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    /* Convert double to float (ONNX typically uses float32) */
    float* input_float = (float*)malloc(input_size * sizeof(float));
    if (!input_float) {
        fprintf(stderr, "ONNX: Failed to allocate input buffer\n");
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    for (int64_t i = 0; i < input_size; i++) {
        input_float[i] = (float)input_data[i];
    }
    
    OrtValue* input_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_float,
        input_size * sizeof(float),
        input_shape,
        input_rank,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to create input tensor: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        free(input_float);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    /* Run inference */
    OrtValue* output_tensor = NULL;
    status = g_ort->Run(
        sess->session,
        NULL,
        (const char* const*)sess->input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        (const char* const*)sess->output_names,
        1,
        &output_tensor
    );
    
    if (status != NULL) {
        fprintf(stderr, "ONNX: Inference failed: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        free(input_float);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    /* Get output tensor data */
    float* output_float;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_float);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get output data: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
        free(input_float);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    /* Convert float to double and copy to output */
    for (int64_t i = 0; i < output_size; i++) {
        output_data[i] = (double)output_float[i];
    }
    
    /* Cleanup */
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    free(input_float);
    g_ort->ReleaseMemoryInfo(memory_info);
    
    return 0;
}

/* Get model input shape information
 * Returns the number of dimensions, and fills shape array if provided
 */
int64_t onnx_get_input_shape(int64_t model_handle, int64_t* shape_out, int64_t max_dims) {
    if (model_handle == 0) {
        return -1;
    }
    
    ONNXSession* sess = (ONNXSession*)model_handle;
    
    /* Get input type info */
    OrtTypeInfo* typeinfo;
    OrtStatus* status = g_ort->SessionGetInputTypeInfo(sess->session, 0, &typeinfo);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get input type info: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    const OrtTensorTypeAndShapeInfo* tensor_info;
    status = g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to cast type info: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseTypeInfo(typeinfo);
        return -1;
    }
    
    size_t num_dims;
    status = g_ort->GetDimensionsCount(tensor_info, &num_dims);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get dimensions count: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseTypeInfo(typeinfo);
        return -1;
    }
    
    /* Copy shape if requested */
    if (shape_out && max_dims > 0) {
        int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
        status = g_ort->GetDimensions(tensor_info, dims, num_dims);
        if (status != NULL) {
            fprintf(stderr, "ONNX: Failed to get dimensions: %s\n",
                    g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
            free(dims);
            g_ort->ReleaseTypeInfo(typeinfo);
            return -1;
        }
        
        size_t copy_dims = num_dims < (size_t)max_dims ? num_dims : (size_t)max_dims;
        for (size_t i = 0; i < copy_dims; i++) {
            shape_out[i] = dims[i];
        }
        free(dims);
    }
    
    g_ort->ReleaseTypeInfo(typeinfo);
    return (int64_t)num_dims;
}

/* Get model output shape information
 * Returns the number of dimensions, and fills shape array if provided
 */
int64_t onnx_get_output_shape(int64_t model_handle, int64_t* shape_out, int64_t max_dims) {
    if (model_handle == 0) {
        return -1;
    }
    
    ONNXSession* sess = (ONNXSession*)model_handle;
    
    /* Get output type info */
    OrtTypeInfo* typeinfo;
    OrtStatus* status = g_ort->SessionGetOutputTypeInfo(sess->session, 0, &typeinfo);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get output type info: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    const OrtTensorTypeAndShapeInfo* tensor_info;
    status = g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to cast type info: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseTypeInfo(typeinfo);
        return -1;
    }
    
    size_t num_dims;
    status = g_ort->GetDimensionsCount(tensor_info, &num_dims);
    if (status != NULL) {
        fprintf(stderr, "ONNX: Failed to get dimensions count: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseTypeInfo(typeinfo);
        return -1;
    }
    
    /* Copy shape if requested */
    if (shape_out && max_dims > 0) {
        int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
        status = g_ort->GetDimensions(tensor_info, dims, num_dims);
        if (status != NULL) {
            fprintf(stderr, "ONNX: Failed to get dimensions: %s\n",
                    g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
            free(dims);
            g_ort->ReleaseTypeInfo(typeinfo);
            return -1;
        }
        
        size_t copy_dims = num_dims < (size_t)max_dims ? num_dims : (size_t)max_dims;
        for (size_t i = 0; i < copy_dims; i++) {
            shape_out[i] = dims[i];
        }
        free(dims);
    }
    
    g_ort->ReleaseTypeInfo(typeinfo);
    return (int64_t)num_dims;
}

