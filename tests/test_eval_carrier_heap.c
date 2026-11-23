/* I test my private ownership engine before adapting source allocations. */
#include "eval/carrier_heap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(test) do { if (!(test)) { \
    fprintf(stderr, "I fail carrier ownership check %s at line %d.\n", #test, __LINE__); \
    exit(1); } } while (0)

typedef struct Node {
    struct Node *edges[3];
    size_t id;
    char *text;
} Node;

typedef struct {
    NlCarrierHeap *heap;
    size_t destroys, traces;
    bool reentry, abort_trace;
    Node *spare;
} Context;

static size_t allocated, released;
static bool retired[100000];

static Node *node(const char *text) {
    Node *value = calloc(1, sizeof *value); CHECK(value);
    value->text = malloc(strlen(text) + 1); CHECK(value->text);
    strcpy(value->text, text);
    value->id = allocated++; CHECK(value->id < sizeof retired / sizeof *retired);
    return value;
}

static void release_node(Node *value) {
    CHECK(!retired[value->id]); retired[value->id] = true; released++;
    free(value->text); free(value);
}

static void destroy_node(void *object, void *data);
static bool trace_node(const void *object, NlCarrierVisit visit, void *visitor, void *data);

static void reject_reentry(Context *ctx, const Node *value) {
    CHECK(!nl_carrier_heap_track(ctx->heap, ctx->spare, trace_node, destroy_node, ctx));
    CHECK(!nl_carrier_heap_forget(ctx->heap, value));
    CHECK(!nl_carrier_heap_collect(ctx->heap, NULL, 0));
    CHECK(!nl_carrier_heap_clear(ctx->heap));
    CHECK(!nl_carrier_heap_free(ctx->heap));
}

static void destroy_node(void *object, void *data) {
    Context *ctx = data;
    CHECK(!nl_carrier_heap_contains(ctx->heap, object));
    if (ctx->reentry) reject_reentry(ctx, object);
    ctx->destroys++; release_node(object);
}

static bool trace_node(const void *object, NlCarrierVisit visit, void *visitor, void *data) {
    const Node *value = object; Context *ctx = data;
    ctx->traces++;
    if (ctx->reentry) reject_reentry(ctx, value);
    for (size_t i = 0; i < 3; ++i) visit(value->edges[i], visitor);
    return !ctx->abort_trace;
}

static void track(Context *ctx, Node *value) {
    CHECK(nl_carrier_heap_track(ctx->heap, value, trace_node, destroy_node, ctx));
}

static void balance(Context *ctx, size_t registered, size_t destroyed, size_t transferred) {
    NlCarrierHeapStats stats = nl_carrier_heap_stats(ctx->heap);
    CHECK(stats.registered == registered); CHECK(stats.destroyed == destroyed);
    CHECK(stats.transferred == transferred);
    CHECK(stats.live == registered - destroyed - transferred);
    CHECK(ctx->destroys == destroyed);
}

static void roots_and_replacement(void) {
    Context ctx = {.heap = nl_carrier_heap_new()}; CHECK(ctx.heap);
    Node *before = node("before"), *after = node("after"), *list = node("typed root");
    track(&ctx, before); track(&ctx, after); track(&ctx, list);
    list->edges[0] = before;
    const void *roots[] = {list, after, list, NULL};
    CHECK(nl_carrier_heap_collect(ctx.heap, roots, 4));
    CHECK(list->edges[0] == before); CHECK(strcmp(before->text, "before") == 0);
    CHECK(strcmp(after->text, "after") == 0); balance(&ctx, 3, 0, 0);
    CHECK(nl_carrier_heap_collect(ctx.heap, roots + 1, 1));
    balance(&ctx, 3, 2, 0); CHECK(strcmp(after->text, "after") == 0);
    CHECK(nl_carrier_heap_collect(ctx.heap, NULL, 0)); balance(&ctx, 3, 3, 0);
    CHECK(nl_carrier_heap_free(ctx.heap));
    puts("I preserve replaced storage through an explicit list root, then reclaim it once.");
}

static void sharing_and_cycles(void) {
    Context ctx = {.heap = nl_carrier_heap_new()}; CHECK(ctx.heap);
    Node *a = node("a"), *b = node("b"), *c = node("shared"), *d = node("dead"), *e = node("cycle");
    Node *borrowed = node("caller-owned");
    track(&ctx, a); track(&ctx, b); track(&ctx, c); track(&ctx, d); track(&ctx, e);
    a->edges[0] = c; b->edges[0] = c; c->edges[0] = a;
    a->edges[1] = borrowed; d->edges[0] = e; e->edges[0] = d;
    const void *roots[] = {a, a, b, borrowed, NULL};
    CHECK(nl_carrier_heap_collect(ctx.heap, roots, 5));
    CHECK(ctx.traces == 3); balance(&ctx, 5, 2, 0);
    CHECK(a->edges[0] == b->edges[0]); CHECK(c->edges[0] == a);
    CHECK(strcmp(borrowed->text, "caller-owned") == 0);
    CHECK(!nl_carrier_heap_contains(ctx.heap, borrowed));
    CHECK(nl_carrier_heap_collect(ctx.heap, NULL, 0)); balance(&ctx, 5, 5, 0);
    CHECK(strcmp(borrowed->text, "caller-owned") == 0);
    release_node(borrowed); CHECK(nl_carrier_heap_free(ctx.heap));
    puts("I preserve shared and cyclic reachable identities and retire unreachable cycles without owning borrowed storage.");
}

static void transfers_and_readoption(void) {
    Context ctx = {.heap = nl_carrier_heap_new()}; CHECK(ctx.heap);
    Node *a = node("external parent"), *b = node("external child"), *early = node("early discard");
    a->edges[0] = b; track(&ctx, a); track(&ctx, b); track(&ctx, early);
    CHECK(nl_carrier_heap_forget(ctx.heap, a)); CHECK(nl_carrier_heap_forget(ctx.heap, b));
    CHECK(nl_carrier_heap_forget(ctx.heap, early)); release_node(early);
    CHECK(!nl_carrier_heap_forget(ctx.heap, a));
    CHECK(nl_carrier_heap_collect(ctx.heap, NULL, 0)); balance(&ctx, 3, 0, 3);
    CHECK(a->edges[0] == b); CHECK(strcmp(b->text, "external child") == 0);
    track(&ctx, a); track(&ctx, b);
    CHECK(nl_carrier_heap_clear(ctx.heap)); balance(&ctx, 5, 2, 3);
    CHECK(nl_carrier_heap_free(ctx.heap));
    puts("I account for explicit graph transfer, early discard and readoption at the same addresses.");
}

static void transfer_between_heaps(void) {
    Context first = {.heap = nl_carrier_heap_new()}, second = {.heap = nl_carrier_heap_new()};
    CHECK(first.heap && second.heap);
    Node *parent = node("transferred"), *child = node("shared descendant");
    parent->edges[0] = child; track(&first, parent); track(&first, child);
    CHECK(nl_carrier_heap_forget(first.heap, parent));
    CHECK(nl_carrier_heap_forget(first.heap, child));
    track(&second, parent); track(&second, child);
    const void *roots[] = {parent};
    CHECK(nl_carrier_heap_collect(first.heap, roots, 1)); balance(&first, 2, 0, 2);
    CHECK(nl_carrier_heap_collect(second.heap, roots, 1)); balance(&second, 2, 0, 0);
    CHECK(parent->edges[0] == child); CHECK(strcmp(child->text, "shared descendant") == 0);
    CHECK(nl_carrier_heap_free(first.heap));
    CHECK(strcmp(parent->text, "transferred") == 0);
    CHECK(nl_carrier_heap_clear(second.heap)); balance(&second, 2, 2, 0);
    CHECK(nl_carrier_heap_free(second.heap));
    puts("I transfer explicit graph ownership between interpreter heaps without copying or cross-heap destruction.");
}

static void refusal_and_reentry(void) {
    Context ctx = {.heap = nl_carrier_heap_new(), .reentry = true}; CHECK(ctx.heap);
    Node *value = node("registered"), *spare = node("caller spare"); ctx.spare = spare;
    CHECK(!nl_carrier_heap_track(NULL, value, trace_node, destroy_node, &ctx));
    CHECK(!nl_carrier_heap_track(ctx.heap, NULL, trace_node, destroy_node, &ctx));
    CHECK(!nl_carrier_heap_track(ctx.heap, value, trace_node, NULL, &ctx));
    track(&ctx, value);
    CHECK(!nl_carrier_heap_track(ctx.heap, value, NULL, destroy_node, NULL));
    CHECK(!nl_carrier_heap_forget(ctx.heap, spare));
    CHECK(!nl_carrier_heap_collect(ctx.heap, NULL, 1));
    const void *roots[] = {value};
    CHECK(nl_carrier_heap_collect(ctx.heap, roots, 1)); balance(&ctx, 1, 0, 0);
    CHECK(ctx.traces == 1); CHECK(!nl_carrier_heap_contains(ctx.heap, spare));
    CHECK(nl_carrier_heap_clear(ctx.heap)); balance(&ctx, 1, 1, 0);
    CHECK(strcmp(spare->text, "caller spare") == 0); release_node(spare);
    CHECK(nl_carrier_heap_free(ctx.heap)); CHECK(nl_carrier_heap_free(NULL));
    puts("I refuse duplicate, missing-policy, unknown and reentrant ownership changes without adopting caller storage.");
}

static void incomplete_trace(void) {
    Context ctx = {.heap = nl_carrier_heap_new(), .abort_trace = true}; CHECK(ctx.heap);
    Node *parent = node("root"), *child = node("queued child"), *dead = node("unreachable");
    parent->edges[0] = child;
    track(&ctx, parent); track(&ctx, child); track(&ctx, dead);
    const void *roots[] = {parent};
    CHECK(!nl_carrier_heap_collect(ctx.heap, roots, 1)); balance(&ctx, 3, 0, 0);
    CHECK(strcmp(parent->text, "root") == 0); CHECK(strcmp(child->text, "queued child") == 0);
    CHECK(strcmp(dead->text, "unreachable") == 0);
    CHECK(nl_carrier_heap_contains(ctx.heap, dead));
    ctx.abort_trace = false;
    CHECK(nl_carrier_heap_collect(ctx.heap, roots, 1)); balance(&ctx, 3, 1, 0);
    CHECK(parent->edges[0] == child); CHECK(strcmp(child->text, "queued child") == 0);
    CHECK(nl_carrier_heap_clear(ctx.heap)); balance(&ctx, 3, 3, 0);
    CHECK(nl_carrier_heap_free(ctx.heap));
    puts("I abort incomplete tracing before any destruction and recover with the original graph intact.");
}

static void deep_graph_and_churn(void) {
    Context ctx = {.heap = nl_carrier_heap_new()}; CHECK(ctx.heap);
    const size_t count = 65536; Node *first = NULL;
    for (size_t i = 0; i < count; ++i) {
        Node *next = node("deep"); next->edges[0] = first; track(&ctx, next); first = next;
    }
    const void *roots[] = {first};
    CHECK(nl_carrier_heap_collect(ctx.heap, roots, 1)); CHECK(ctx.traces == count);
    balance(&ctx, count, 0, 0);
    CHECK(nl_carrier_heap_collect(ctx.heap, NULL, 0)); balance(&ctx, count, count, 0);
    for (size_t round = 0; round < 64; ++round) {
        for (size_t i = 0; i < 33; ++i) track(&ctx, node("churn"));
        CHECK(nl_carrier_heap_clear(ctx.heap));
    }
    balance(&ctx, count + 64 * 33, count + 64 * 33, 0);
    CHECK(nl_carrier_heap_free(ctx.heap));
    puts("I trace 65536 linked allocations iteratively and balance repeated allocation/removal cohorts.");
}

int main(void) {
    roots_and_replacement(); sharing_and_cycles(); transfers_and_readoption();
    refusal_and_reentry(); transfer_between_heaps(); incomplete_trace(); deep_graph_and_churn();
    CHECK(allocated == released);
    printf("I balance %zu carrier payload allocations and releases.\n", allocated);
    return 0;
}
