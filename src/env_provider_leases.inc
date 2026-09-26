/* Identity carries no Environment pointer; retained task references prevent
 * address reuse even after a scalar task's Environment is destroyed. */
struct EnvTaskIdentity { size_t references; };
bool env_task_identity_retain(EnvTaskIdentity *identity) {
    if (!identity || !identity->references || identity->references == SIZE_MAX) return false;
    ++identity->references;
    return true;
}
void env_task_identity_release(EnvTaskIdentity *identity) {
    if (!identity || !identity->references) {
        fprintf(stderr, "I cannot release an invalid task identity.\n"); exit(1);
    }
    if (--identity->references == 0) free(identity);
}

/* I track exact AST-provider generations. No pointer into a ModuleCache is
 * retained here; references keep this identity alive after its cache closes. */
struct EnvEvaluationProvider {
    size_t references;
    size_t leases;
    bool alive;
};
struct EnvProviderEdge {
    struct EnvProviderEdge *next;
    EnvEvaluationProvider *provider;
};

EnvEvaluationProvider *env_provider_new(void) {
    EnvEvaluationProvider *provider = calloc(1, sizeof(*provider));
    if (provider) { provider->references = 1; provider->alive = true; }
    return provider;
}

bool env_register_provider(Environment *env, EnvEvaluationProvider *provider) {
    if (!env || !provider || !provider->alive || !provider->references) return false;
    for (struct EnvProviderEdge *edge = env->evaluation_providers; edge; edge = edge->next)
        if (edge->provider == provider) return true;
    if (provider->references == SIZE_MAX ||
        env->evaluation_leases > SIZE_MAX - provider->leases) return false;
    struct EnvProviderEdge *edge = malloc(sizeof(*edge));
    if (!edge) return false;
    edge->provider = provider;
    edge->next = env->evaluation_providers;
    ++provider->references;
    provider->leases += env->evaluation_leases;
    env->evaluation_providers = edge;
    return true;
}

bool env_provider_close(EnvEvaluationProvider *provider) {
    if (!provider || !provider->alive || !provider->references || provider->leases) return false;
    provider->alive = false;
    return true;
}

void env_provider_release(EnvEvaluationProvider *provider) {
    if (!provider || !provider->references || (provider->references == 1 && provider->leases)) {
        fprintf(stderr, "I cannot release an invalid provider reference.\n"); exit(1);
    }
    if (--provider->references == 0) free(provider);
}

bool env_acquire_evaluation_lease(Environment *env) {
    if (!env || env->evaluation_leases == SIZE_MAX) return false;
    for (struct EnvProviderEdge *edge = env->evaluation_providers; edge; edge = edge->next) {
        EnvEvaluationProvider *provider = edge->provider;
        if (!provider->alive || !provider->references || provider->leases == SIZE_MAX) return false;
    }
    /* My sequential two-pass transaction allocates nothing and calls no hooks. */
    for (struct EnvProviderEdge *edge = env->evaluation_providers; edge; edge = edge->next)
        ++edge->provider->leases;
    ++env->evaluation_leases;
    return true;
}

void env_release_evaluation_lease(Environment *env) {
    if (!env || !env->evaluation_leases) {
        fprintf(stderr, "I cannot release an absent evaluator lease.\n"); exit(1);
    }
    for (struct EnvProviderEdge *edge = env->evaluation_providers; edge; edge = edge->next) {
        if (!edge->provider->alive || edge->provider->leases < env->evaluation_leases) {
            fprintf(stderr, "I cannot release an inconsistent provider lease.\n"); exit(1);
        }
    }
    for (struct EnvProviderEdge *edge = env->evaluation_providers; edge; edge = edge->next)
        --edge->provider->leases;
    --env->evaluation_leases;
}

static void env_provider_edges_free(Environment *env) {
    while (env->evaluation_providers) {
        struct EnvProviderEdge *edge = env->evaluation_providers;
        env->evaluation_providers = edge->next;
        env_provider_release(edge->provider);
        free(edge);
    }
}
