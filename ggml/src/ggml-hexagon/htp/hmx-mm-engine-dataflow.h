#ifndef HMX_MM_ENGINE_DATAFLOW_H
#define HMX_MM_ENGINE_DATAFLOW_H

// Dataflow engine of the C++ HMX matmul pipeline (selected by sched_stream).
//
// Barrier-free: every thread runs one event loop until the whole matmul is stored. Each stage of job j
// (one weight chunk against one activation chunk) is split into work units claimed with generation-tagged
// counters (hmx-mm-core.h):
//   weight DMA slice (owner thread)   -> w_ready[slot][w]
//   dequant pieces (any thread)       -> dq counters, gated on the slice and on the w buffer being free
//   MAC(j) (main thread submits)      -> gated on dequant(j), act(mi) and the o buffer being free
//   store units (any thread)          -> gated on MAC(j) done
//   act blocks (threads with staging) -> gated on the act buffer being free, converted when the DMA lands
// A thread's DMA ring is a strict FIFO, so descriptors are popped in order and dispatched by a per-thread
// tag ring (weight slice or act block). The scheduler decides job order, buffer slots and when a buffer
// may be reused; the engine owns the counters, the rings and the thread loop.

#include "hmx-mm-core.h"

namespace hmx_mm {

enum : uint32_t { no_job = 0xFFFFFFFFu };

template <class S, class pre, class post, class mac>
struct engine_dataflow;

// Streaming scheduler: m-outer / n-inner jobs, ring-indexed buffer slots, buffer reuse rules.
// Serves decode (one m-chunk) and prefill alike; the planner sizes the chunks and buffers.
struct sched_stream {
    template <class S, class pre, class post, class mac>
    using engine = engine_dataflow<S, pre, post, mac>;

    chunk_grid grid;
    uint32_t   n_raw_bufs;
    uint32_t   n_a_bufs;

    sched_stream() = default;

    sched_stream(const shape & s, const plan & p, uint32_t tile_k) :
        grid(s, p, tile_k), n_raw_bufs(p.n_raw_bufs), n_a_bufs(p.n_a_bufs) {}

    // buffer slots of a job / m-chunk
    uint32_t raw_slot(uint32_t j) const { return j % n_raw_bufs; }
    uint32_t w_slot(uint32_t j) const { return j % n_w_bufs; }
    uint32_t o_slot(uint32_t j) const { return j % n_o_bufs; }
    uint32_t act_slot(uint32_t mi) const { return mi % 2; }  // counter ring; the buffers alias when n_a_bufs == 1

    // buffer reuse: the earlier job whose stage must be complete before the slot is written again (no_job: none)
    uint32_t raw_slot_busy_until_dequant(uint32_t j) const { return j >= n_raw_bufs ? j - n_raw_bufs : no_job; }
    uint32_t o_slot_busy_until_stored(uint32_t j) const { return j >= n_o_bufs ? j - n_o_bufs : no_job; }
    // ... or the number of MAC jobs that must have completed
    uint32_t w_slot_needs_mac_done(uint32_t j) const { return j >= n_w_bufs ? j - n_w_bufs + 1 : 0; }
    uint32_t act_slot_needs_mac_done(uint32_t mi) const { return mi < n_a_bufs ? 0 : (mi - n_a_bufs + 1) * grid.n_nb; }

    // activations of m-chunks below this bound may be fetched once MAC(j) of m-chunk mi was submitted
    uint32_t act_frontier(uint32_t mi) const { return min_u32(mi + 1, grid.n_mb); }
};

enum : uint32_t {
    job_ring          = 8,     // >= n_raw_bufs + 5 jobs in flight
    tag_ring          = 16,
    dma_poll_interval = 1024,  // cycles between dmpoll calls while a descriptor is pending
    max_dq_pieces     = 32,    // dequant pieces per chunk (upper bound; pieces are >= dq_min_tiles)
    dq_min_tiles      = 16,
    max_store_units   = 48,    // store units per chunk
    store_min_tiles   = 4,     // column tiles per store unit
};

// work partition of one weight chunk shape (ramp chunks, full chunk, last partial chunk)
struct chunk_units {
    uint32_t n_ct;
    uint32_t n_cb;                     // store column blocks
    uint32_t n_pieces;                 // dequant pieces over all slices
    uint32_t n_slices;                 // slices with at least one piece
    uint32_t ct0[max_threads + 1];     // slice w covers col tiles [ct0[w], ct0[w + 1])
    uint32_t piece0[max_threads + 1];  // slice w covers pieces [piece0[w], piece0[w + 1])
};

enum : uint32_t { max_chunk_shapes = 8, shape_full = 5, shape_last = 6 };

// what a thread is waiting for on its DMA ring, in push order
struct dma_tag {
    uint8_t  kind;  // 0 = weight slice, 1 = act block
    uint8_t  half;  // act staging half
    uint32_t job;   // weight: job; act: m-chunk
    uint32_t row0;  // act: first row of the block (chunk local)
};

struct dma_tag_ring {
    dma_tag  tags[tag_ring];
    uint32_t n_push;
    uint32_t n_pop;

    void     clear() { n_push = n_pop = 0; }
    bool     empty() const { return n_pop == n_push; }
    bool     full() const { return n_push - n_pop >= tag_ring - 2; }
    void     push(dma_tag t) { tags[n_push++ % tag_ring] = t; }
    dma_tag  pop() { return tags[n_pop++ % tag_ring]; }
};

enum cyc_bucket : uint32_t { cyc_dequant, cyc_store, cyc_act, cyc_dma_svc, cyc_idle, cyc_dq_scan, cyc_st_scan, cyc_issue, n_cyc_buckets };

struct thread_state {
    dma_tag_ring     dma;
    bool             half_busy[2];
    uint32_t         act_rows_in_half[2];
    uint32_t         w_push_next;  // next job whose slice this (weight) thread pushes
    uint32_t         dq_cur;       // lowest job that may still have dequant pieces
    uint32_t         st_cur;       // lowest job that may still have store units
    uint64_t         last_poll;    // cycle of the last dmpoll
    uint32_t         iters;
    cycle_buckets<n_cyc_buckets> busy;

    void clear() {
        dma.clear();
        half_busy[0] = half_busy[1] = false;
        w_push_next = 0;
        dq_cur = st_cur = 0;
        last_poll = 0;
        iters = 0;
        busy.clear();
    }
} __attribute__((aligned(128)));

template <class S, class pre, class post, class mac>
struct engine_dataflow {
    using C      = composition<pre, post, mac>;
    using W      = typename C::W;
    using A      = typename C::A;
    using P      = typename C::P;
    using M      = typename C::M;
    using args_t = arguments<pre, post>;
    using counter = job_counter<job_ring>;

    context           c;
    args_t            p;
    plan              pl;
    layout            L;
    buffers<C>        B;
    S                 sched;
    mac_dispatcher<M> hmx;
    uint32_t          n_w_threads;
    uint32_t          dq_per;    // tiles per dequant piece
    uint32_t          st_block;  // col tiles per store unit

    chunk_units       units[max_chunk_shapes];
    uint32_t          n_ramp;
    bool              last_partial;

    // shared progress
    padded_u32        w_ready[max_raw_bufs][max_threads];
    counter           dq_claim[job_ring][max_threads];
    counter           dq_done[job_ring][max_threads];  // pieces finished per slice
    counter           dq_slices[job_ring];             // slices fully dequantized
    counter           st_claim[job_ring];
    counter           st_done[job_ring];
    job_counter<2>    act_claim[2];
    job_counter<2>    act_done[2];
    uint32_t          mac_next;     // main thread only
    padded_u32        mac_done;     // jobs [0, mac_done) completed on the MAC unit
    padded_u32        jobs_stored;  // jobs fully written to DDR
    padded_u32        act_mi_next;  // m-chunks below may be fetched (main thread advances)

    thread_state      ts[max_threads];
    uint64_t          t_start, t_loop, submit_cycles;
    uint32_t          n_reports;

    const chunk_grid & grid() const { return sched.grid; }

    // ---- work partition (table driven, no divisions on the hot path) ----
    void build_units(chunk_units & u, uint32_t n_ct) const {
        u.n_ct      = n_ct;
        u.n_cb      = ceil_div_u32(n_ct, st_block);
        u.ct0[1]    = 0;
        u.piece0[1] = 0;
        u.n_slices  = 0;
        for (uint32_t w = 1; w <= n_w_threads; w++) {
            u.ct0[w + 1]    = w * n_ct / n_w_threads;
            u.piece0[w + 1] = u.piece0[w] + ceil_div_u32((u.ct0[w + 1] - u.ct0[w]) * grid().n_kt, dq_per);
            u.n_slices     += (u.ct0[w + 1] > u.ct0[w]) ? 1 : 0;
        }
        u.n_pieces = u.piece0[n_w_threads + 1];
    }

    const chunk_units & units_of(uint32_t ni) const {
        if (ni < n_ramp) {
            return units[ni];
        }
        return (last_partial && ni == grid().n_nb - 1) ? units[shape_last] : units[shape_full];
    }

    uint32_t store_units(uint32_t j) const { return grid().m_row_tiles(grid().job_mi(j)) * units_of(grid().job_ni(j)).n_cb; }
    uint32_t act_blocks(uint32_t mi) const { return grid().m_row_tiles(mi) * tile_dim / pl.act_rows_per_dma; }

    bool dq_complete(uint32_t j) { return dq_slices[j % job_ring].complete(j, units_of(grid().job_ni(j)).n_slices); }
    bool st_complete(uint32_t j) { return st_done[j % job_ring].complete(j, store_units(j)); }
    bool act_complete(uint32_t mi) { return act_done[sched.act_slot(mi)].complete(mi, act_blocks(mi)); }

    uint32_t mac_done_acq() const { return atomic_load_acq(&mac_done.v); }

    // ---- weight stream: slice DMA per thread ----
    bool raw_slot_free(uint32_t j) {
        const uint32_t prev = sched.raw_slot_busy_until_dequant(j);
        return prev == no_job || dq_complete(prev);
    }

    bool issue_weight_dma(uint32_t i, thread_state & t) {
        const uint32_t j = t.w_push_next;
        if (j >= grid().n_jobs || t.dma.full() || !raw_slot_free(j)) {
            return false;
        }
        const uint32_t      ni  = grid().job_ni(j);
        const chunk_units & u   = units_of(ni);
        const uint32_t      ct0 = u.ct0[i];
        const uint32_t      ct1 = u.ct0[i + 1];
        if (ct1 == ct0) {
            atomic_store_rel(&w_ready[sched.raw_slot(j)][i].v, j + 1);
        } else {
            const uint32_t   n_kt  = grid().n_kt;
            const size_t     off   = (size_t) ct0 * n_kt * W::raw_tile_bytes;
            const size_t     bytes = (size_t) (ct1 - ct0) * n_kt * W::raw_tile_bytes;
            const dma_addr_t src   = p.w.tile_addr(grid().n_col0(ni) / tile_dim, n_kt) + off;
            dma_queue_push(c.dma(i), dma_make_data(B.raw[sched.raw_slot(j)] + off, src), bytes, bytes, bytes, 1);
            t.dma.push(dma_tag{ 0, 0, j, 0 });
        }
        t.w_push_next++;
        return true;
    }

    // ---- activation stream: row blocks DMAed into a per-thread staging half, converted when landed ----
    bool act_slot_free(uint32_t mi) { return mac_done_acq() >= sched.act_slot_needs_mac_done(mi); }

    bool issue_act_dma(uint32_t i, thread_state & t) {
        if (i >= pl.act_threads) {
            return false;
        }
        const uint32_t mi = atomic_load_acq(&act_mi_next.v);
        if (mi >= grid().n_mb || !act_slot_free(mi)) {
            return false;
        }
        uint32_t half = 2;
        if (!t.half_busy[0]) half = 0;
        else if (!t.half_busy[1]) half = 1;
        if (half == 2 || t.dma.full()) {
            return false;
        }
        const uint32_t n_blk = act_blocks(mi);
        const uint32_t b     = act_claim[sched.act_slot(mi)].claim(mi, n_blk);
        if (b >= n_blk) {
            return false;
        }
        const uint32_t R     = pl.act_rows_per_dma;
        const uint32_t row0  = b * R;
        const uint32_t valid = grid().m_rows(mi);
        const uint32_t nv    = row0 < valid ? min_u32(R, valid - row0) : 0;
        t.half_busy[half]        = true;
        t.act_rows_in_half[half] = nv;
        if (nv) {
            const size_t row_bytes = (size_t) p.s.k * sizeof(typename A::in_t);
            uint8_t *    dst       = B.act_stage[i] + half * R * row_bytes;
            dma_queue_push(c.dma(i), dma_make_data(dst, p.a.row_addr(mi * pl.mc + row0)), row_bytes, p.a.nb1, row_bytes, nv);
            t.dma.push(dma_tag{ 1, (uint8_t) half, mi, row0 });
        } else {
            convert_act_block(i, t, mi, row0, half);  // padding rows: nothing to wait for
        }
        return true;
    }

    void convert_act_block(uint32_t i, thread_state & t, uint32_t mi, uint32_t row0, uint32_t half) {
        const uint32_t R         = pl.act_rows_per_dma;
        const size_t   row_bytes = (size_t) p.s.k * sizeof(typename A::in_t);
        const auto *   rows      = (const typename A::in_t *) (B.act_stage[i] + half * R * row_bytes);
        A::convert(rows, p.s.k, R, t.act_rows_in_half[half], B.act[sched.act_slot(mi)], row0);
        t.half_busy[half] = false;
        act_done[sched.act_slot(mi)].finish(mi, act_blocks(mi));
    }

    // pop every finished descriptor at the head of this thread's ring
    bool service_dma(uint32_t i, thread_state & t) {
        bool any = false;
        dma_ring * r = c.dma(i)->ring0;
        while (!t.dma.empty()) {
            // the done bit only becomes visible through dmpoll, which is not cheap: poll at a fixed rate
            volatile dma_descriptor_2d * desc = &r->desc[r->pop_idx];
            if (!desc->done) {
                const uint64_t now = hex_get_cycles();
                if (now - t.last_poll < dma_poll_interval) {
                    break;
                }
                t.last_poll = now;
                dmpoll();
                if (!desc->done) {
                    break;
                }
            }
            dma_ring_pop(r);
            const dma_tag tag = t.dma.pop();
            if (tag.kind == 0) {
                atomic_store_rel(&w_ready[sched.raw_slot(tag.job)][i].v, tag.job + 1);
            } else {
                const uint64_t c0 = t.busy.now();
                convert_act_block(i, t, tag.job, tag.row0, tag.half);
                t.busy.add(cyc_act, t.busy.now() - c0);
            }
            any = true;
        }
        return any;
    }

    // ---- MAC stage (main thread): collect completions, submit the next job when its inputs are ready ----
    bool o_slot_free(uint32_t j) {
        const uint32_t prev = sched.o_slot_busy_until_stored(j);
        return prev == no_job || st_complete(prev);
    }

    bool service_mac() {
        bool any = false;
        if (hmx.poll()) {
            atomic_store_rel(&mac_done.v, hmx.n_popped);
            any = true;
        }

        const uint32_t j = mac_next;
        if (j < grid().n_jobs) {
            const uint32_t mi = grid().job_mi(j);
            if (dq_complete(j) && act_complete(mi) && o_slot_free(j)) {
                typename M::job jb;
                jb.out    = B.o[sched.o_slot(j)];
                jb.act    = B.act[sched.act_slot(mi)];
                jb.w      = B.w[sched.w_slot(j)];
                jb.scales = B.scales;
                jb.n_rt   = grid().m_row_tiles(mi);
                jb.n_ct   = grid().n_col_tiles(grid().job_ni(j));
                jb.n_kt   = grid().n_kt;
                const uint64_t s0 = ts[0].busy.now();
                hmx.submit(j, jb);
                submit_cycles += ts[0].busy.now() - s0;
                mac_next++;
                // let the act threads fetch one m-chunk ahead of the MAC frontier
                const uint32_t ahead = sched.act_frontier(mi);
                if (atomic_load_acq(&act_mi_next.v) < ahead) {
                    atomic_store_rel(&act_mi_next.v, ahead);
                }
                any = true;
            }
        }
        return any;
    }

    // ---- dequant stage ----
    // A thread first dequantizes the slice it DMAed itself (uncontended counter, data it just waited for),
    // then steals from the other slices.
    bool run_dequant(uint32_t i, thread_state & t) {
        const uint32_t md = mac_done_acq();
        for (uint32_t j = t.dq_cur; j < grid().n_jobs && j < t.dq_cur + 2; j++) {
            if (j == t.dq_cur && dq_complete(j)) {
                t.dq_cur++;
                continue;
            }
            if (md < sched.w_slot_needs_mac_done(j)) {
                return false;  // w buffer still read by an earlier MAC job
            }
            const chunk_units & u    = units_of(grid().job_ni(j));
            const uint32_t      slot = sched.raw_slot(j);
            const uint32_t      home = (i >= 1 && i <= n_w_threads) ? i : 1 + (j % n_w_threads);
            for (uint32_t k = 0; k < n_w_threads; k++) {
                const uint32_t w    = 1 + (home - 1 + k) % n_w_threads;
                const uint32_t n_pc = u.piece0[w + 1] - u.piece0[w];
                if (n_pc == 0 || __atomic_load_n(&w_ready[slot][w].v, __ATOMIC_RELAXED) != j + 1) {
                    continue;
                }
                const uint32_t pc = dq_claim[j % job_ring][w].claim(j, n_pc);
                if (pc >= n_pc) {
                    continue;
                }
                // no fence: the core is in order and the slice data landed before w_ready was published, and a
                // barrier here would drain the posted vector stores of the previous piece (~1K cycles)
                const uint32_t t0 = u.ct0[w] * grid().n_kt + pc * dq_per;
                const uint32_t t1 = min_u32(t0 + dq_per, u.ct0[w + 1] * grid().n_kt);
                const uint64_t c0 = t.busy.now();
                htp_trace_event_start(c.trace(i), HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) t0);
                W::convert(B.raw[slot], B.w[sched.w_slot(j)], t0, t1);
                htp_trace_event_stop(c.trace(i), HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) t0);
                t.busy.add(cyc_dequant, t.busy.now() - c0);
                if (dq_done[j % job_ring][w].finish(j, n_pc)) {
                    dq_slices[j % job_ring].finish(j, u.n_slices);
                }
                return true;
            }
        }
        return false;
    }

    // ---- store stage ----
    bool run_store(uint32_t i, thread_state & t) {
        const uint32_t md = mac_done_acq();
        for (uint32_t j = t.st_cur; j < md; j++) {
            if (j == t.st_cur && st_complete(j)) {
                t.st_cur++;
                continue;
            }
            const uint32_t n_u = store_units(j);
            const uint32_t u   = st_claim[j % job_ring].claim(j, n_u);
            if (u >= n_u) {
                continue;
            }
            const uint32_t      mi  = grid().job_mi(j);
            const uint32_t      ni  = grid().job_ni(j);
            const chunk_units & cu  = units_of(ni);
            const uint32_t      rt  = u / cu.n_cb;
            const uint32_t      ct0 = (u - rt * cu.n_cb) * st_block;
            const uint32_t      ct1 = min_u32(ct0 + st_block, cu.n_ct);
            const uint32_t      row = mi * pl.mc + rt * tile_dim;
            const uint32_t      col = grid().n_col0(ni) + ct0 * tile_dim;
            const uint64_t      c0  = t.busy.now();
            htp_trace_event_start(c.trace(i), HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) j);
            P::store(B.o[sched.o_slot(j)], cu.n_ct, rt, ct0, ct1, min_u32(tile_dim, grid().m_rows(mi) - rt * tile_dim), p.o, row, col);
            htp_trace_event_stop(c.trace(i), HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) j);
            t.busy.add(cyc_store, t.busy.now() - c0);
            if (st_done[j % job_ring].finish(j, n_u)) {
                __atomic_fetch_add(&jobs_stored.v, 1, __ATOMIC_ACQ_REL);
            }
            return true;
        }
        return false;
    }

    // ---- thread loop ----
    void thread_loop(uint32_t i) {
        thread_state & t    = ts[i];
        const bool     is_w = (i >= 1 && i <= n_w_threads);
        if (i == 0) {
            t_loop = hex_get_cycles();
        }
        while (atomic_load_acq(&jobs_stored.v) < grid().n_jobs) {
            bool busy = false;
            t.iters++;
            uint64_t c0 = t.busy.now();
            busy |= service_dma(i, t);
            uint64_t c1 = t.busy.now();
            t.busy.add(cyc_dma_svc, c1 - c0);
            if (is_w) {
                busy |= issue_weight_dma(i, t);
            }
            if (i == 0) {
                busy |= service_mac();
            }
            busy |= issue_act_dma(i, t);
            c0 = t.busy.now();
            t.busy.add(cyc_issue, c0 - c1);
            const uint64_t dq0 = t.busy.get(cyc_dequant);
            busy |= run_dequant(i, t);
            c1 = t.busy.now();
            t.busy.add(cyc_dq_scan, (c1 - c0) - (t.busy.get(cyc_dequant) - dq0));
            const uint64_t st0 = t.busy.get(cyc_store);
            busy |= run_store(i, t);
            c0 = t.busy.now();
            t.busy.add(cyc_st_scan, (c0 - c1) - (t.busy.get(cyc_store) - st0));
            if (!busy) {
                idle_pause();
                t.busy.add(cyc_idle, t.busy.now() - c0);
            }
        }
        // drain: every pushed descriptor was popped (tags empty) except possibly act DMAs of unused halves
        while (!t.dma.empty()) {
            service_dma(i, t);
        }
    }

    static void thread_main(unsigned int n, unsigned int i, void * data) {
        (void) n;
        ((engine_dataflow *) data)->thread_loop(i);
    }

    int run(const context & ctx, const args_t & args, const plan & plan_) {
        t_start = hex_get_cycles();
        c  = ctx;
        p  = args;
        pl = plan_;
        L  = build_layout(p.s, pl, C::fmt);
        if (L.total > c.ctx->vtcm_size) {
            FARF(ERROR, "hmx-mm-df: VTCM overflow: need %u have %u", (unsigned) L.total, (unsigned) c.ctx->vtcm_size);
            return -1;
        }
        B     = buffers<C>(c.vtcm, L, pl);
        sched = S(p.s, pl, C::tile_k);
        hmx.attach(c.ctx->hmx_queue);
        n_w_threads = n_w_dma_threads(pl);
        if (n_w_threads < 1 || pl.n_raw_bufs + 5 > job_ring || pl.act_threads == 0) {
            return -1;
        }

        // work unit sizes
        const uint32_t tiles = (pl.nc / tile_dim) * grid().n_kt;
        dq_per = align_up_u32(ceil_div_u32(tiles, max_dq_pieces), W::piece_align);
        if (dq_per < dq_min_tiles) {
            dq_per = dq_min_tiles;
        }
        st_block = ceil_div_u32(grid().m_row_tiles(0) * (pl.nc / tile_dim), max_store_units);
        if (st_block < store_min_tiles) {
            st_block = store_min_tiles;
        }

        n_ramp = min_u32(ramp_chunks(pl.nc, pl.ramp), grid().n_nb);
        for (uint32_t i = 0; i < n_ramp; i++) {
            build_units(units[i], grid().n_col_tiles(i));
        }
        build_units(units[shape_full], pl.nc / tile_dim);
        last_partial = (grid().n_nb > n_ramp) && (grid().n_cols(grid().n_nb - 1) != pl.nc);
        if (last_partial) {
            build_units(units[shape_last], grid().n_col_tiles(grid().n_nb - 1));
        }

        for (uint32_t s = 0; s < max_raw_bufs; s++) {
            for (uint32_t w = 0; w < max_threads; w++) {
                w_ready[s][w].v = 0;
            }
        }
        for (uint32_t s = 0; s < job_ring; s++) {
            for (uint32_t w = 0; w < max_threads; w++) {
                dq_claim[s][w].clear();
                dq_done[s][w].clear();
            }
            dq_slices[s].clear();
            st_claim[s].clear();
            st_done[s].clear();
        }
        for (uint32_t s = 0; s < 2; s++) {
            act_claim[s].clear();
            act_done[s].clear();
        }
        mac_next      = 0;
        mac_done.v    = 0;
        jobs_stored.v = 0;
        act_mi_next.v = 0;
        for (uint32_t i = 0; i < max_threads; i++) {
            ts[i].clear();
        }

        M::init_scales(B.scales);
        submit_cycles = 0;
        const uint64_t t_run = hex_get_cycles();
        work_queue_run(c.ctx->work_queue, thread_main, this, c.n_threads);
        const uint64_t t_end = hex_get_cycles();
        hmx.wait_until(grid().n_jobs);  // all popped already; keeps the queue accounting consistent
        report(t_end - t_start, t_run - t_start, t_loop - t_run, t_end - t_loop);
        return 0;
    }

    // every 32nd op only: the FARF itself costs tens of us
    void report(uint64_t total, uint64_t setup, uint64_t dispatch, uint64_t loop) {
#ifdef HMX_MM_STATS
        if ((n_reports++ & 31) != 0) {
            return;
        }
        uint64_t sum[n_cyc_buckets] = { 0 };
        uint32_t iters = 0;
        for (uint32_t i = 0; i < c.n_threads; i++) {
            for (uint32_t k = 0; k < n_cyc_buckets; k++) sum[k] += ts[i].busy.get(k);
            iters += ts[i].iters;
        }
        FARF(ALWAYS, "hmx-mm-df: m %u k %u n %u mc %u nc %u ramp %u jobs %u wth %u ath %u | total %u setup %u dispatch %u loop %u submit %u",
             p.s.m, p.s.k, p.s.n, pl.mc, pl.nc, pl.ramp, grid().n_jobs, n_w_threads, pl.act_threads,
             (unsigned) total, (unsigned) setup, (unsigned) dispatch, (unsigned) loop, (unsigned) submit_cycles);
        FARF(ALWAYS, "hmx-mm-df2: dequant %u store %u act %u dma-svc %u idle %u dq-scan %u st-scan %u issue %u iters %u",
             (unsigned) sum[cyc_dequant], (unsigned) sum[cyc_store], (unsigned) sum[cyc_act], (unsigned) sum[cyc_dma_svc],
             (unsigned) sum[cyc_idle], (unsigned) sum[cyc_dq_scan], (unsigned) sum[cyc_st_scan], (unsigned) sum[cyc_issue], iters);
#else
        (void) total; (void) setup; (void) dispatch; (void) loop;
#endif
    }
};

}  // namespace hmx_mm

#endif  // HMX_MM_ENGINE_DATAFLOW_H
