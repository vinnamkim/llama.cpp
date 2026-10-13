#ifndef HMX_MM_CORE_H
#define HMX_MM_CORE_H

// Building blocks shared by the engines of the C++ HMX matmul pipeline:
//   component contracts (concepts), the composition check that derives the planner `format`,
//   problem arguments, the chunk grid, typed VTCM buffers, the asynchronous MAC dispatcher and a few
//   lock-free primitives.
// Freestanding C++20: no exceptions, no RTTI, no libc++.

#include <stdint.h>
#include <stddef.h>

#ifndef restrict
#define restrict __restrict
#endif

extern "C" {
#include <HAP_farf.h>
#include "hmx-utils.h"
#include "dma-queue.h"
#include "hmx-queue.h"
#include "work-queue.h"
#include "htp-ctx.h"
#include "hex-profile.h"
}

#include "hmx-mm-plan.h"

namespace hmx_mm {

// --- type helpers (no <type_traits>) --------------------------------------------------------------

template <class A, class B> inline constexpr bool is_same_v = false;
template <class A> inline constexpr bool is_same_v<A, A> = true;

template <class A, class B>
concept same_as = is_same_v<A, B>;

// --- component contracts --------------------------------------------------------------------------
//
// A matmul is composed from four kinds of components. Each one is a stateless type: constants and types
// the engines and the planner read (traits) plus static functions that run on one thread. Components never
// touch DMA rings, counters or the MAC queue; that is the engines' job.

// Weight pre-processor: raw weight tiles (DDR layout, DMAed into VTCM unchanged) -> MAC weight tiles.
//   source           describes the weight matrix in DDR: tile_addr(ct, n_kt) = raw tile (ct, 0)
//   raw_tile_bytes   size of one raw tile; tile (ct, kt) sits at (ct * n_kt + kt) * raw_tile_bytes
//   piece_align      convert tiles in groups aligned to this count (q4_0 keeps tile pairs contiguous)
//   convert          raw tiles [t0, t1) of a chunk -> tiles [t0, t1), tile t at tiles + t * tile_dim * tile_k
template <class W>
concept weight_prep = requires(const uint8_t * raw, typename W::tile_t * tiles, uint32_t t0, uint32_t t1,
                               const typename W::source & src, uint32_t ct, uint32_t n_kt) {
    typename W::tile_t;
    typename W::source;
    W::tile_k;
    W::raw_tile_bytes;
    W::piece_align;
    W::cost_per_thread;  // elements converted per cycle per thread, for the planner
    { src.tile_addr(ct, n_kt) } -> same_as<dma_addr_t>;
    W::convert(raw, tiles, t0, t1);
};

// Activation pre-processor: rows of in_t (staged in VTCM by DMA) -> MAC activation tiles.
//   source    describes the activation matrix in DDR: row_addr(r), nb1 = bytes between rows
//   convert   rows [row0, row0 + n_rows) of a chunk from `rows` (n_rows x k, dense); rows >= n_valid are
//             padding and written as zero. Tile (rt, kt) of the chunk sits at (rt * n_kt + kt) * tile_dim * tile_k.
template <class A>
concept act_prep = requires(const typename A::in_t * rows, uint32_t k, uint32_t n_rows, uint32_t n_valid,
                            typename A::tile_t * tiles, uint32_t row0, const typename A::source & src, uint32_t r) {
    typename A::in_t;
    typename A::tile_t;
    typename A::source;
    A::tile_k;
    A::cost_per_thread;
    { src.row_addr(r) } -> same_as<dma_addr_t>;
    { src.nb1 } -> same_as<const uint32_t &>;
    A::convert(rows, k, n_rows, n_valid, tiles, row0);
};

// Post-processor: MAC output tiles of one chunk -> destination rows, with any epilogue.
//   sink    describes the destination (pointer, stride, valid columns, later bias / residual)
//   store   tiles (rt, [ct0, ct1)) of a chunk whose tile row rt holds rows_valid rows, to destination
//           rows [row, row + rows_valid), columns from col; columns past sink.cols are dropped
template <class P>
concept post_proc = requires(const typename P::in_t * tiles, uint32_t n_ct, uint32_t rt, uint32_t ct0, uint32_t ct1,
                             uint32_t rows_valid, const typename P::sink & dst, uint32_t row, uint32_t col) {
    typename P::in_t;
    typename P::sink;
    P::cost_per_thread;
    { dst.cols } -> same_as<const uint32_t &>;
    P::store(tiles, n_ct, rt, ct0, ct1, rows_valid, dst, row, col);
};

// MAC unit: out[rt][ct] = sum_kt act[rt][kt] x w[ct][kt] over whole tiles, asynchronous on the HMX thread.
//   job           one chunk product; the engine fills out, act, w, scales, n_rt, n_ct, n_kt
//   scales_bytes  size of the column scale table in VTCM, filled once by init_scales
template <class M>
concept mac_unit = requires(const typename M::job & jb, void * scales) {
    typename M::act_t;
    typename M::w_t;
    typename M::out_t;
    typename M::job;
    M::tile_k;
    M::scales_bytes;
    M::cost_macs;
    M::cost_job_overhead;
    M::cost_tile_overhead;
    M::init_scales(scales);
    M::dot(jb);
};

// The pre-processor of execution<> bundles the two input converters.
template <weight_prep W, act_prep A>
struct pre_processor {
    using weight = W;
    using act    = A;
};

// Static check that four components fit together, plus everything derived from their traits.
template <class pre, post_proc post, mac_unit mac>
struct composition {
    using W = typename pre::weight;
    using A = typename pre::act;
    using P = post;
    using M = mac;
    static_assert(weight_prep<W> && act_prep<A>);
    static_assert(same_as<typename W::tile_t, typename M::w_t>, "weight prep must produce the MAC unit's weight tiles");
    static_assert(same_as<typename A::tile_t, typename M::act_t>, "act prep must produce the MAC unit's activation tiles");
    static_assert(same_as<typename P::in_t, typename M::out_t>, "post processor must consume the MAC unit's output tiles");
    static_assert(W::tile_k == M::tile_k && A::tile_k == M::tile_k, "tile k depth must match the MAC unit");

    static constexpr uint32_t tile_k         = M::tile_k;
    static constexpr uint32_t in_tile_elems  = tile_dim * tile_k;    // one act or weight tile
    static constexpr uint32_t out_tile_elems = tile_dim * tile_dim;  // one output tile

    // what the planner sees; must equal the constant the host planned with (checked in hmx-mm-cpp.cc)
    static constexpr format fmt = {
        tile_k, sizeof(typename A::in_t), sizeof(typename M::act_t), sizeof(typename M::w_t), sizeof(typename M::out_t),
        W::raw_tile_bytes, M::scales_bytes,
        W::cost_per_thread, A::cost_per_thread, P::cost_per_thread, M::cost_macs, M::cost_job_overhead, M::cost_tile_overhead,
    };
};

// --- problem --------------------------------------------------------------------------------------

struct context {
    htp_context * ctx;
    uint8_t *     vtcm;
    uint32_t      n_threads;

    dma_queue *        dma(uint32_t i) const { return ctx->dma[i]; }
    htp_thread_trace * trace(uint32_t i) const { return &ctx->trace[i]; }
};

// One matmul: shape plus the component-specific descriptions of the three operands.
template <class pre, class post>
struct arguments {
    shape                        s;  // m, k, n with n padded to tile_dim
    typename pre::weight::source w;
    typename pre::act::source    a;
    typename post::sink          o;
};

// The (m-chunk, n-chunk) grid of a plan; job j = (mi, ni) in m-outer / n-inner order.
struct chunk_grid {
    uint32_t       m, k, n;
    uint32_t       mc, nc, ramp;
    uint32_t       n_kt;
    uint32_t       n_mb;
    uint32_t       n_nb;
    uint32_t       n_jobs;
    fastdiv_values div_nb;

    chunk_grid() = default;

    chunk_grid(const shape & s, const plan & p, uint32_t tile_k) :
        m(s.m), k(s.k), n(s.n), mc(p.mc), nc(p.nc), ramp(p.ramp) {
        n_kt   = s.k / tile_k;
        n_mb   = ceil_div_u32(s.m, p.mc);
        n_nb   = n_col_chunks(s.n, p.nc, p.ramp);
        n_jobs = n_mb * n_nb;
        div_nb = init_fastdiv_values(n_nb);
    }

    uint32_t job_mi(uint32_t j) const { return fastdiv(j, &div_nb); }
    uint32_t job_ni(uint32_t j) const { return j - job_mi(j) * n_nb; }

    uint32_t m_rows(uint32_t mi) const { return min_u32(mc, m - mi * mc); }
    uint32_t m_row_tiles(uint32_t mi) const { return ceil_div_u32(m_rows(mi), tile_dim); }
    uint32_t n_col0(uint32_t ni) const { return col_chunk_begin(nc, ramp, ni); }
    uint32_t n_cols(uint32_t ni) const { return col_chunk_rows(n, nc, ramp, ni); }
    uint32_t n_col_tiles(uint32_t ni) const { return n_cols(ni) / tile_dim; }
};

// VTCM regions of a layout, typed by the composition.
template <class C>
struct buffers {
    using act_t = typename C::M::act_t;
    using w_t   = typename C::M::w_t;
    using out_t = typename C::M::out_t;

    void *    scales;
    act_t *   act[2];                  // act[i % n_a_bufs]: with one buffer both entries alias
    uint8_t * act_stage[max_threads];  // per thread id; only act threads own one
    uint8_t * raw[max_raw_bufs];
    w_t *     w[n_w_bufs];
    out_t *   o[n_o_bufs];

    buffers() = default;

    buffers(uint8_t * base, const layout & L, const plan & p) {
        scales = base + L.off_scales;
        for (uint32_t i = 0; i < 2; i++) {
            act[i] = (act_t *) (base + L.off_act + (i % p.n_a_bufs) * L.act_bytes);
        }
        const uint32_t first_act = p.engine ? 0 : n_w_dma_threads(p) + 1;
        for (uint32_t i = 0; i < max_threads; i++) {
            const uint32_t slot = (p.act_threads && i >= first_act) ? (i - first_act) % p.act_threads : 0;
            act_stage[i] = base + L.off_act_stage + slot * L.act_stage_bytes;
        }
        for (uint32_t i = 0; i < max_raw_bufs; i++) {
            raw[i] = base + L.off_raw + (i % p.n_raw_bufs) * L.raw_bytes;
        }
        for (uint32_t i = 0; i < n_w_bufs; i++) {
            w[i] = (w_t *) (base + L.off_w + i * L.w_bytes);
        }
        for (uint32_t i = 0; i < n_o_bufs; i++) {
            o[i] = (out_t *) (base + L.off_o + i * L.o_bytes);
        }
    }
};

// --- MAC dispatcher -------------------------------------------------------------------------------

// Runs mac_unit jobs on the hmx-queue thread. Up to n_o_bufs jobs are in flight (one per output buffer).
template <mac_unit M>
struct mac_dispatcher {
    using job = typename M::job;

    hmx_queue_t q;
    job         slots[n_o_bufs];
    uint32_t    n_submitted;
    uint32_t    n_popped;

    void attach(hmx_queue_t hq) {
        q           = hq;
        n_submitted = 0;
        n_popped    = 0;
    }

    static void run(void * data) {
        M::dot(*(const job *) data);
    }

    void submit(uint32_t j, const job & jb) {
        job * slot = &slots[j % n_o_bufs];
        *slot = jb;
        hmx_queue_push(q, hmx_queue_make_desc(run, slot));
        n_submitted++;
    }

    // block until jobs [0, n_done) have completed
    void wait_until(uint32_t n_done) {
        while (n_popped < n_done && n_popped < n_submitted) {
            hmx_queue_pop(q);
            n_popped++;
        }
    }

    // pop every completed job at the head of the queue without blocking; returns how many
    uint32_t poll() {
        uint32_t n = 0;
        for (;;) {
            if (q->idx_pop == atomic_load(&q->idx_write)) {
                break;
            }
            struct hmx_queue_desc * d = &q->desc[q->idx_pop];
            if (!atomic_load(&d->done)) {
                break;
            }
            const struct hmx_queue_desc rd = hmx_queue_pop_one(q);
            const uint32_t sig = (uint32_t) (uintptr_t) rd.func;
            if (sig && sig <= HMX_QUEUE_KILL) {
                continue;  // queue control signal, not one of our jobs
            }
            n_popped++;
            n++;
        }
        return n;
    }
};

// --- lock-free primitives -------------------------------------------------------------------------

static inline void idle_pause() {
    asm volatile(" pause(#4)\n");  // short: a thread must notice new DMA / MAC completions quickly
}

static inline uint32_t atomic_inc(uint32_t * p) {
    return __atomic_fetch_add(p, 1, __ATOMIC_ACQ_REL);
}

static inline void atomic_store_rel(uint32_t * p, uint32_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

static inline uint32_t atomic_load_acq(const uint32_t * p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

// one 32-bit value per cache line, so unrelated counters do not bounce the same line
struct padded_u32 {
    uint32_t v;
    uint32_t pad[31];
} __attribute__((aligned(128)));

// Generation-tagged work counter: (gen(job) << 24) | count. A slot is reused by job + ring, whose generation
// differs, so stale values from the previous job are never mistaken for progress. One per cache line.
template <uint32_t ring>
struct job_counter {
    enum : uint32_t { gen_shift = 24, count_mask = (1u << gen_shift) - 1 };

    uint32_t v;
    uint32_t pad[31];

    static uint32_t gen(uint32_t j) { return ((j / ring) & 0xFF) + 1; }

    void clear() { v = 0; }

    // move the counter to generation g (count 0); returns the current value afterwards
    uint32_t reset_to(uint32_t g) {
        uint32_t cur = __atomic_load_n(&v, __ATOMIC_ACQUIRE);
        while ((cur >> gen_shift) != g) {
            if (__atomic_compare_exchange_n(&v, &cur, g << gen_shift, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                return g << gen_shift;
            }
        }
        return cur;
    }

    // claim one unit of job j out of n; returns the unit index or n if none left.
    // One fetch_add in the common case; a count past n is harmless (everyone sees "none left").
    uint32_t claim(uint32_t j, uint32_t n) {
        const uint32_t g = gen(j);
        uint32_t cur = __atomic_load_n(&v, __ATOMIC_RELAXED);
        if ((cur >> gen_shift) != g) {
            cur = reset_to(g);
        }
        if ((cur & count_mask) >= n) {
            return n;
        }
        const uint32_t old = __atomic_fetch_add(&v, 1, __ATOMIC_ACQ_REL);
        const uint32_t idx = old & count_mask;
        return idx < n ? idx : n;
    }

    // count one finished unit of job j; returns true when it was the last of n
    bool finish(uint32_t j, uint32_t n) {
        const uint32_t g = gen(j);
        if ((__atomic_load_n(&v, __ATOMIC_RELAXED) >> gen_shift) != g) {
            reset_to(g);
        }
        const uint32_t next = __atomic_fetch_add(&v, 1, __ATOMIC_ACQ_REL) + 1;
        return (next & count_mask) == n;
    }

    bool complete(uint32_t j, uint32_t n) const {
        const uint32_t cur = __atomic_load_n(&v, __ATOMIC_ACQUIRE);
        return (cur >> gen_shift) == gen(j) && (cur & count_mask) >= n;
    }
} __attribute__((aligned(128)));

// --- cycle accounting -----------------------------------------------------------------------------

// Per-thread cycle buckets, compiled out unless HMX_MM_STATS is defined.
template <uint32_t n_buckets>
struct cycle_buckets {
#ifdef HMX_MM_STATS
    uint64_t v[n_buckets];
    void     clear() { for (uint32_t k = 0; k < n_buckets; k++) v[k] = 0; }
    void     add(uint32_t k, uint64_t cycles) { v[k] += cycles; }
    uint64_t get(uint32_t k) const { return v[k]; }
    static uint64_t now() { return hex_get_cycles(); }
#else
    void     clear() {}
    void     add(uint32_t, uint64_t) {}
    uint64_t get(uint32_t) const { return 0; }
    static uint64_t now() { return 0; }
#endif
};

}  // namespace hmx_mm

#endif  // HMX_MM_CORE_H
