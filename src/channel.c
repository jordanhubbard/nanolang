/*
 * channel.c — Channel implementation for nanolang async/coroutine concurrency
 */

#include "channel.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ── Channel lifecycle ──────────────────────────────────────────────────── */

Channel *channel_new(uint32_t capacity) {
    Channel *ch = calloc(1, sizeof(Channel));
    if (!ch) return NULL;
    ch->capacity = capacity;
    if (capacity > 0) {
        if (capacity > CHANNEL_MAX_CAPACITY) capacity = CHANNEL_MAX_CAPACITY;
        ch->buf = calloc(capacity, sizeof(ChanVal));
        if (!ch->buf) { free(ch); return NULL; }
    }
    ch->blocked_sender   = TASK_NONE;
    ch->blocked_receiver = TASK_NONE;
    return ch;
}

void channel_free(Channel *ch) {
    if (!ch) return;
    free(ch->buf);
    free(ch);
}

void channel_close(Channel *ch) {
    if (!ch) return;
    ch->closed = true;
}

bool channel_is_closed(const Channel *ch) {
    return ch && ch->closed;
}

/* ── Ring buffer helpers ────────────────────────────────────────────────── */

static bool buf_full(const Channel *ch) {
    return ch->capacity > 0 && ch->count >= ch->capacity;
}

static bool buf_empty(const Channel *ch) {
    return ch->count == 0;
}

static void buf_push(Channel *ch, ChanVal val) {
    if (ch->capacity == 0 || ch->count >= ch->capacity) return;
    ch->buf[ch->write_pos] = val;
    ch->write_pos = (ch->write_pos + 1) % ch->capacity;
    ch->count++;
}

static ChanVal buf_pop(Channel *ch) {
    if (ch->count == 0) return 0;
    ChanVal v = ch->buf[ch->read_pos];
    ch->read_pos = (ch->read_pos + 1) % ch->capacity;
    ch->count--;
    return v;
}

/* ── Non-blocking operations ─────────────────────────────────────────────── */

bool channel_try_send(Channel *ch, ChanVal val) {
    if (!ch || ch->closed) return false;

    /* Unbuffered: check for waiting receiver */
    if (ch->capacity == 0) {
        if (ch->blocked_receiver != TASK_NONE) {
            /* Deliver directly to waiting receiver */
            if (ch->recv_out_ptr) *ch->recv_out_ptr = val;
            ch->pending_send_val    = val;
            ch->blocked_receiver    = TASK_NONE;
            return true;
        }
        return false;  /* no receiver waiting */
    }

    /* Buffered: check buffer space */
    if (buf_full(ch)) return false;
    buf_push(ch, val);

    /* Wake any waiting receiver */
    if (ch->blocked_receiver != TASK_NONE)
        ch->blocked_receiver = TASK_NONE;

    return true;
}

bool channel_try_recv(Channel *ch, ChanVal *val_out) {
    if (!ch) return false;

    /* Buffered: pop from buffer */
    if (ch->capacity > 0 && !buf_empty(ch)) {
        if (val_out) *val_out = buf_pop(ch);
        /* Wake any waiting sender */
        if (ch->blocked_sender != TASK_NONE) {
            buf_push(ch, ch->pending_send_val);
            ch->blocked_sender = TASK_NONE;
        }
        return true;
    }

    /* Unbuffered or empty buffered: check for waiting sender */
    if (ch->blocked_sender != TASK_NONE) {
        if (val_out) *val_out = ch->pending_send_val;
        ch->blocked_sender = TASK_NONE;
        return true;
    }

    /* Channel closed and empty: signal EOF */
    if (ch->closed) {
        if (val_out) *val_out = 0;
        return false;  /* EOF */
    }

    return false;  /* would block */
}

/* ── Blocking operations (cooperative — park via scheduler) ────────────── */
/*
 * The "blocking" variants record the waiting task handle.
 * The coroutine scheduler calls channel_try_* on each tick and unparks
 * tasks whose channels become ready.
 */

bool channel_send(Channel *ch, ChanVal val, TaskHandle sender_task) {
    if (!ch) return false;
    if (ch->closed) {
        fprintf(stderr, "[channel] send on closed channel\n");
        return false;
    }
    if (channel_try_send(ch, val)) return true;
    /* Park sender */
    ch->blocked_sender     = sender_task;
    ch->pending_send_val   = val;
    return true;  /* scheduler will resume when ready */
}

bool channel_recv(Channel *ch, ChanVal *val_out, TaskHandle receiver_task) {
    if (!ch) return false;
    if (channel_try_recv(ch, val_out)) return true;
    if (ch->closed) return false;  /* closed and empty */
    /* Park receiver */
    ch->blocked_receiver = receiver_task;
    /* Store pointer for direct delivery in channel_try_send */
    ch->recv_out_ptr     = val_out;
    return true;  /* scheduler will resume when ready */
}

/* ── Select ──────────────────────────────────────────────────────────────── */

int channel_select(SelectCase *cases, int n_cases) {
    int default_idx = -1;
    /* First pass: check each case without blocking */
    for (int i = 0; i < n_cases; i++) {
        SelectCase *c = &cases[i];
        c->ready = false;
        if (c->kind == SELECT_DEFAULT) { default_idx = i; continue; }
        if (!c->ch) continue;
        if (c->kind == SELECT_SEND) {
            if (channel_try_send(c->ch, c->send_val)) { c->ready = true; return i; }
        } else if (c->kind == SELECT_RECV) {
            ChanVal v = 0;
            if (channel_try_recv(c->ch, &v)) {
                if (c->recv_out) *c->recv_out = v;
                c->ready = true;
                return i;
            }
        }
    }
    /* No case ready — return default if present, else -1 */
    return default_idx;
}

/* ── Channel registry ────────────────────────────────────────────────────── */

ChannelRegistry *channel_registry_new(void) {
    return calloc(1, sizeof(ChannelRegistry));
}

void channel_registry_free(ChannelRegistry *reg) {
    if (!reg) return;
    for (int i = 0; i < reg->count; i++)
        channel_free(reg->channels[i]);
    free(reg);
}

int channel_registry_add(ChannelRegistry *reg, Channel *ch) {
    if (!reg || !ch || reg->count >= CHANNEL_REGISTRY_MAX) return -1;
    int idx = reg->count++;
    reg->channels[idx] = ch;
    return idx;
}

Channel *channel_registry_get(ChannelRegistry *reg, int handle) {
    if (!reg || handle < 0 || handle >= reg->count) return NULL;
    return reg->channels[handle];
}
