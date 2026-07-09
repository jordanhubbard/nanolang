/*
 * channel.h — Typed channels for nanolang async/coroutine concurrency
 *
 * Channels are the primary communication primitive between async tasks
 * (coroutines) in nanolang.  They follow the CSP (Communicating Sequential
 * Processes) model, similar to Go channels.
 *
 * Channel types:
 *   - Unbuffered (capacity = 0): synchronous rendez-vous — send blocks
 *     until a receiver is ready, and recv blocks until a sender arrives.
 *   - Buffered (capacity > 0): send returns immediately if buffer has space;
 *     blocks when full.  Recv blocks when empty.
 *
 * Usage from C (runtime):
 *   Channel *ch = channel_new(8);   // buffered, capacity 8
 *   channel_send(ch, 42);           // send int value
 *   int v = channel_recv(ch);       // receive (blocks if empty)
 *   channel_close(ch);              // signal no more sends
 *   channel_free(ch);
 *
 * Usage from nanolang (desugared by the async pass):
 *   let ch = chan int [8]            // declare buffered channel
 *   let ch2 = chan int               // unbuffered
 *   ch <- 42                         // send (may yield if full)
 *   let v = <- ch                    // recv (may yield if empty)
 *   close ch                         // close channel
 *
 * Select statement (multi-channel wait):
 *   select {
 *     v <- ch1 => { print(v) }
 *     ch2 <- 99 => { }
 *     default   => { }               // optional: no-op if none ready
 *   }
 *
 * The scheduler (coroutine.c) parks tasks waiting on channels and
 * re-schedules them when their channel operation becomes ready.
 *
 * Thread safety: all operations are single-threaded (cooperative scheduling).
 * No locks needed — channels are accessed only from the coroutine scheduler.
 */

#ifndef NANOLANG_CHANNEL_H
#define NANOLANG_CHANNEL_H

#include <stdint.h>
#include <stdbool.h>

/* ── Channel state ──────────────────────────────────────────────────────── */

#define CHANNEL_MAX_CAPACITY  4096

typedef struct Channel Channel;

/*
 * The channel value type is int64_t for the interpreter.
 * The transpiler and LLVM backend emit typed wrappers.
 */
typedef int64_t ChanVal;

/* Waiting task handle (opaque — resolved by coroutine.c) */
typedef uint32_t TaskHandle;
#define TASK_NONE  ((TaskHandle)UINT32_MAX)

struct Channel {
    ChanVal    *buf;          /* ring buffer (NULL if unbuffered) */
    uint32_t    capacity;     /* buffer capacity (0 = unbuffered) */
    uint32_t    count;        /* items currently in buffer */
    uint32_t    read_pos;     /* ring buffer read head */
    uint32_t    write_pos;    /* ring buffer write head */
    bool        closed;       /* no more sends after close */
    /* Blocked-task queues (single sender/receiver for v1) */
    TaskHandle  blocked_sender;
    ChanVal     pending_send_val;  /* value waiting to be delivered */
    TaskHandle  blocked_receiver;
    ChanVal    *recv_out_ptr;      /* direct-delivery pointer for parked receiver */
};

/* ── Channel lifecycle ──────────────────────────────────────────────────── */

/* Allocate a new channel (capacity = 0 for unbuffered) */
Channel *channel_new(uint32_t capacity);

/* Free channel and its buffer */
void channel_free(Channel *ch);

/* Close channel: no more sends allowed; pending receivers will drain buffer */
void channel_close(Channel *ch);

bool channel_is_closed(const Channel *ch);

/* ── Send / Recv ─────────────────────────────────────────────────────────── */

/*
 * channel_try_send — non-blocking send.
 * Returns true if val was buffered or delivered to a waiting receiver.
 * Returns false if the channel is full/unbuffered with no waiting receiver.
 */
bool channel_try_send(Channel *ch, ChanVal val);

/*
 * channel_try_recv — non-blocking recv.
 * Returns true + writes *val_out if a value was available.
 * Returns false if the channel is empty / no pending sender.
 */
bool channel_try_recv(Channel *ch, ChanVal *val_out);

/*
 * channel_send — blocking send from within the coroutine scheduler.
 * Parks the current task if the channel is full; unparked when space becomes available.
 * Panics if channel is closed.
 *
 * sender_task — the TaskHandle of the calling coroutine (TASK_NONE if main)
 * Returns true on success, false on panic/error.
 */
bool channel_send(Channel *ch, ChanVal val, TaskHandle sender_task);

/*
 * channel_recv — blocking recv from within the coroutine scheduler.
 * Parks the current task if the channel is empty; unparked when data arrives.
 * Returns false (with *val_out = 0) when channel is closed and empty.
 */
bool channel_recv(Channel *ch, ChanVal *val_out, TaskHandle receiver_task);

/* ── Select ──────────────────────────────────────────────────────────────── */

/* Select case type */
typedef enum {
    SELECT_SEND,    /* ch <- val */
    SELECT_RECV,    /* val <- ch */
    SELECT_DEFAULT, /* no-op */
} SelectCaseKind;

typedef struct {
    SelectCaseKind  kind;
    Channel        *ch;
    ChanVal         send_val;   /* for SELECT_SEND */
    ChanVal        *recv_out;   /* for SELECT_RECV (output) */
    bool            ready;      /* set by select_poll */
} SelectCase;

/*
 * channel_select — evaluate a set of select cases.
 * Returns the index of the first ready case, or -1 if none ready and
 * no DEFAULT case.  If a DEFAULT case exists and no other is ready,
 * returns the DEFAULT case index.
 *
 * Does not block — blocking select is implemented in the scheduler
 * by parking the task and re-polling on each scheduler tick.
 */
int channel_select(SelectCase *cases, int n_cases);

/* ── Channel registry (for named channels in interpreted mode) ───────────── */

#define CHANNEL_REGISTRY_MAX  256

typedef struct {
    Channel *channels[CHANNEL_REGISTRY_MAX];
    int      count;
} ChannelRegistry;

/* Per-program channel registry (one per interpreter context) */
ChannelRegistry *channel_registry_new(void);
void             channel_registry_free(ChannelRegistry *reg);

/* Register a channel; returns its index (used as the int handle in nanolang) */
int channel_registry_add(ChannelRegistry *reg, Channel *ch);

/* Look up a channel by handle */
Channel *channel_registry_get(ChannelRegistry *reg, int handle);

#endif /* NANOLANG_CHANNEL_H */
