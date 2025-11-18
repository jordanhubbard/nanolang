// Module Build System for Nanolang
// Handles automatic compilation of C sources and dependency tracking

#ifndef MODULE_BUILDER_H
#define MODULE_BUILDER_H

#include <stdbool.h>
#include <stddef.h>
#include "nanolang.h"

// Module build metadata structure (from module.json)
typedef struct {
    char *name;
    char *version;
    char *description;
    
    // C headers to include (e.g., ["SDL.h", "SDL_mixer.h"])
    char **headers;
    size_t headers_count;
    
    // Header priority (higher = included first, default = 0)
    int header_priority;
    
    // C sources to compile
    char **c_sources;
    size_t c_sources_count;
    
    // System libraries to link (-l flags)
    char **system_libs;
    size_t system_libs_count;
    
    // pkg-config packages
    char **pkg_config;
    size_t pkg_config_count;
    
    // Include directories (-I flags)
    char **include_dirs;
    size_t include_dirs_count;
    
    // Custom compiler flags
    char **cflags;
    size_t cflags_count;
    
    // Custom linker flags
    char **ldflags;
    size_t ldflags_count;
    
    // Module dependencies (other nanolang modules)
    char **dependencies;
    size_t dependencies_count;
    
    // Module directory path
    char *module_dir;
} ModuleBuildMetadata;

// Build information for tracking
typedef struct {
    char *object_file;      // Path to compiled .o file
    char **link_flags;      // All flags needed for linking
    size_t link_flags_count;
    char **compile_flags;   // Compile flags (include paths, defines)
    size_t compile_flags_count;
    bool needs_rebuild;     // Whether C sources need recompilation
} ModuleBuildInfo;

// Module builder context
typedef struct ModuleBuilder ModuleBuilder;

// Create a new module builder
ModuleBuilder* module_builder_new(const char *module_path);

// Free module builder
void module_builder_free(ModuleBuilder *builder);

// Load module metadata from module.json
// Returns NULL if no module.json exists (pure nanolang module)
ModuleBuildMetadata* module_load_metadata(const char *module_dir);

// Free module metadata
void module_metadata_free(ModuleBuildMetadata *meta);

// Build a module (compile C sources if needed)
// Returns build info with object file path and link flags
ModuleBuildInfo* module_build(ModuleBuilder *builder, ModuleBuildMetadata *meta);

// Free build info
void module_build_info_free(ModuleBuildInfo *info);

// Get all link flags for a set of modules
char** module_get_link_flags(ModuleBuildInfo **modules, size_t count, size_t *out_count);

// Get all compile flags for a set of modules
char** module_get_compile_flags(ModuleBuildInfo **modules, size_t count, size_t *out_count);

// Check if module needs rebuild
bool module_needs_rebuild(const char *module_dir, ModuleBuildMetadata *meta);

// Get module build directory path
char* module_get_build_dir(const char *module_dir);

// Create build directory if it doesn't exist
bool module_ensure_build_dir(const char *module_dir);

// Verbose build output (controlled by NANO_VERBOSE_BUILD env var)
extern bool module_builder_verbose;

// Parse C header to extract #define constants
// Returns array of ConstantDef, caller must free
// Note: ConstantDef is defined in nanolang.h
ConstantDef* parse_c_header_constants(const char *header_path, int *count_out);

#endif // MODULE_BUILDER_H

