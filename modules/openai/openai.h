#ifndef NANOLANG_OPENAI_H
#define NANOLANG_OPENAI_H

#include <stdint.h>

// OpenAI API client functions
// All functions return JSON strings or error messages
// Authentication uses OPENAI_API_KEY environment variable or explicit key parameter
// Default endpoint is https://api.openai.com/v1 but can be overridden

// Configuration
void nl_openai_set_api_base(const char* base_url);
char* nl_openai_get_api_base(void);
char* nl_openai_get_key_from_env(void);

// Chat completions - main API
char* nl_openai_chat_completion(const char* model, const char* messages_json, const char* api_key);
char* nl_openai_chat_completion_simple(const char* model, const char* system_prompt, const char* user_message, const char* api_key);
char* nl_openai_chat_completion_with_temperature(const char* model, const char* messages_json, double temperature, int64_t max_tokens, const char* api_key);

// Embeddings
char* nl_openai_create_embedding(const char* model, const char* input, const char* api_key);

// Model management
char* nl_openai_list_models(const char* api_key);
char* nl_openai_get_model(const char* model_id, const char* api_key);

// Utility
int64_t nl_openai_count_tokens_estimate(const char* text);

#endif // NANOLANG_OPENAI_H
