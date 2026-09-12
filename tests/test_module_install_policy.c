/* I intercept every system command in the production module builder. No test
 * here invokes a package manager, sudo, or a manifest-provided shell command. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int pkg_config_present = 1;
static int package_present = 0;
static int authority_commands = 0;
static int install_commands = 0;

static int policy_access(const char *path, int mode) {
    (void)mode;
    return strstr(path, "pkg-config") && pkg_config_present ? 0 : -1;
}

static int policy_system(const char *command) {
    if (strstr(command, "command -v pkg-config")) return pkg_config_present ? 0 : 1;
    if (strstr(command, "pkg-config") && strstr(command, "--exists"))
        return package_present ? 0 : 1;
    authority_commands++;
    if (strcmp(command, "fixture-install") == 0) install_commands++;
    return 0;
}

#define access policy_access
#define system policy_system
#include "../src/module_builder.c"
#undef system
#undef access

int main(void) {
    char *packages[] = {"fixture-unavailable-package"};
    ModuleBuildMetadata meta = {0};
    meta.name = "fixture";
    meta.pkg_config = packages;
    meta.pkg_config_count = 1;
    meta.system_packages = packages;
    meta.system_packages_count = 1;
    meta.install_brew = "fixture";
    meta.install_apt = "fixture";

    const char *disabled[] = {NULL, "", "0", "true", "yes", "01", "1 "};
    for (size_t i = 0; i < sizeof(disabled) / sizeof(disabled[0]); i++) {
        if (disabled[i]) setenv("NANO_ALLOW_PACKAGE_INSTALL", disabled[i], 1);
        else unsetenv("NANO_ALLOW_PACKAGE_INSTALL");
        authority_commands = 0;
        assert(!install_single_package_ex("fixture", PKG_MGR_BREW,
                                          "fixture-install", "fixture-probe"));
        assert(authority_commands == 0);
        assert(!install_single_package("fixture", PKG_MGR_APT));
        assert(authority_commands == 0);
        assert(!install_system_packages(&meta));
        assert(authority_commands == 0);

        assert(!ensure_module_system_deps(&meta));
        assert(authority_commands == 0);
        assert(!module_build(NULL, &meta));
        assert(authority_commands == 0);
        meta.system_packages_count = 0; /* I also check legacy install mappings. */
        assert(!ensure_module_system_deps(&meta));
        assert(authority_commands == 0);
        meta.system_packages_count = 1;

        package_present = 1;
        assert(ensure_module_system_deps(&meta));
        assert(authority_commands == 0);
        package_present = 0;
        meta.pkg_config_count = 0; /* No declared probe means no known missing dependency. */
        assert(ensure_module_system_deps(&meta));
        assert(authority_commands == 0);
        meta.pkg_config_count = 1;

        pkg_config_present = 0;
        pkg_config_install_attempted = false;
        assert(!ensure_pkg_config());
        assert(authority_commands == 0);
        pkg_config_present = 1;
    }

    /* I test opt-in routing with an intercepted command, not a real install. */
    setenv("NANO_ALLOW_PACKAGE_INSTALL", "1", 1);
    authority_commands = 0;
    assert(install_single_package_ex("fixture", PKG_MGR_BREW, "fixture-install", NULL));
    assert(install_commands == 1);
    assert(authority_commands > 0);
    unsetenv("NANO_ALLOW_PACKAGE_INSTALL");
    puts("I passed module installation authority checks.");
    return 0;
}
