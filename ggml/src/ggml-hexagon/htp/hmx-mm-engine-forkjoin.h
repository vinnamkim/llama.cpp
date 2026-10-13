#ifndef HMX_MM_ENGINE_FORKJOIN_H
#define HMX_MM_ENGINE_FORKJOIN_H

// Fork-join step engine of the C++ HMX matmul pipeline (selected by sched_decode / sched_prefill).
//
// Step j (job j = one weight chunk against one activation chunk):
//   wait MAC(j-2) -> one fork-join phase { weight DMA push(j+D-1) / pop(j) per slice, dequant(j),
//   store(j-2), act prep slice } -> submit MAC(j)
// MAC(j-1) runs during the whole phase. Three output buffers make store(j-2) independent of MAC(j-1).
// Weight chunks are split into column-tile slices, one per weight DMA thread, so several DMA engines
// stream in parallel; dequant tasks wait on a per-slice ready flag.
// The scheduler decides, per step, where the activation conversion of a chunk is placed and how far the
// MAC unit must have progressed; the engine builds the task list of the phase and runs it.

#include "hmx-mm-core.h"

namespace hmx_mm {

template <class S, class pre, class post, class mac>
struct engine_forkjoin;

enum : uint32_t { no_act = 0xFFFFFFFFu, no_task_job = 0xFFFFFFFFu };

// what one step does besides dequant(j) and the pending stores
struct step_plan {
    uint32_t act_mi;    // m-chunk whose rows [act_r0, act_r1) are converted in this phase, or no_act
    uint32_t act_r0;
    uint32_t act_r1;
    uint32_t wait_mac;  // MAC jobs [0, wait_mac) must be complete before the phase
};

static inline uint32_t default_wait(uint32_t j) {
    return j >= 1 ? j - 1 : 0;  // MAC(j-2) complete: its output buffer is stored in this phase
}

// m-outer / n-inner. With two activation buffers the next m-chunk is converted in slices during steps
// 1..S-1 of the current m-chunk, so the conversion overlaps dequant and MAC work.
struct sched_prefill {
    template <class S, class pre, class post, class mac>
    using engine = engine_forkjoin<S, pre, post, mac>;

    chunk_grid grid;
    uint32_t   n_a_bufs;

    sched_prefill() = default;

    sched_prefill(const shape & s, const plan & p, uint32_t tile_k) : grid(s, p, tile_k), n_a_bufs(p.n_a_bufs) {}

    void step(uint32_t j, step_plan * sp) const {
        const uint32_t mi = grid.job_mi(j);
        const uint32_t si = grid.job_ni(j);  // step index inside the m-chunk
        const uint32_t S  = grid.n_nb;
        sp->act_mi   = no_act;
        sp->wait_mac = default_wait(j);

        if (j == 0) {
            full(0, sp);
            return;
        }
        if (n_a_bufs == 1) {
            if (si == 0) {
                full(mi, sp);
                sp->wait_mac = j;  // the single buffer is still read by the previous m-chunk
            }
            return;
        }

        const uint32_t next = mi + 1;
        if (next >= grid.n_mb) {
            return;
        }
        if (S < 2) {
            // no steps to hide the conversion behind: convert at the start of its own m-chunk
            if (si == 0 && mi > 0) {
                full(mi, sp);
            }
            return;
        }
        if (si >= 1) {
            const uint32_t units = grid.m_row_tiles(next);
            const uint32_t u0    = (si - 1) * units / (S - 1);
            const uint32_t u1    = si * units / (S - 1);
            if (u1 > u0) {
                sp->act_mi = next;
                sp->act_r0 = u0 * tile_dim;
                sp->act_r1 = u1 * tile_dim;
            }
        }
    }

    void full(uint32_t mi, step_plan * sp) const {
        sp->act_mi = mi;
        sp->act_r0 = 0;
        sp->act_r1 = grid.m_row_tiles(mi) * tile_dim;
    }
};

// one activation chunk, weight chunks streamed; the activation is converted in step 0
struct sched_decode {
    template <class S, class pre, class post, class mac>
    using engine = engine_forkjoin<S, pre, post, mac>;

    chunk_grid grid;

    sched_decode() = default;

    sched_decode(const shape & s, const plan & p, uint32_t tile_k) : grid(s, p, tile_k) {}

    void step(uint32_t j, step_plan * sp) const {
        sp->act_mi   = no_act;
        sp->wait_mac = default_wait(j);
        if (j == 0) {
            sp->act_mi = 0;
            sp->act_r0 = 0;
            sp->act_r1 = grid.m_row_tiles(0) * tile_dim;
        }
    }
};

// --- phase tasks ----------------------------------------------------------------------------------

enum class task_kind : uint8_t {
    weight,  // dequant tiles [begin, end) of job idx, all inside DMA slice rt
    store,   // tile row rt, col tiles [begin, end) of job idx
    act,     // rows [begin, end) of m-chunk idx
};

struct task {
    task_kind kind;
    uint32_t  idx;
    uint32_t  rt;
    uint32_t  begin;
    uint32_t  end;
};

// Worst case phase: act prep (mc / 32 <= 64) + dequant (32) + stores of two jobs (2 x 48) = 192.
enum : uint32_t {
    max_tasks        = 256,
    max_weight_tasks = 32,
    max_store_tasks  = 48,
    max_act_rows     = 2048,
    store_ct_block   = 4,
    weight_min_tiles = 16,
};

// One fork-join phase. Tasks [0, n_dma) use the per-thread DMA ring and are only run by threads that own
// an activation staging area; weight DMA threads push / pop their slices before taking tasks.
struct phase {
    task     tasks[max_tasks];
    uint32_t n_dma;
    uint32_t n;
    uint32_t next_dma;
    uint32_t next_any;
    uint32_t w_push_first;  // weight DMA threads push their slice of jobs [w_push_first, w_push_last)
    uint32_t w_push_last;
    uint32_t w_pop_job;     // ... then pop their slice of this job (no_task_job: none)
    bool     overflow;

    void reset() {
        n_dma        = 0;
        n            = 0;
        next_dma     = 0;
        next_any     = 0;
        w_push_first = 0;
        w_push_last  = 0;
        w_pop_job    = no_task_job;
    }

    void add(const task & t) {
        if (n < max_tasks) {
            tasks[n++] = t;
        } else {
            overflow = true;
        }
    }

    void close_dma() { n_dma = n; }
};

// Per-op cycle breakdown, printed with FARF when HMX_MM_STATS is defined.
struct fj_stats {
    uint64_t total;
    uint64_t mac_wait;
    uint64_t phase;
    uint64_t task[4][max_threads];  // per task kind (3 = weight DMA wait), per thread
    uint32_t n_phases;

    void reset() {
        total = mac_wait = phase = 0;
        n_phases = 0;
        for (uint32_t k = 0; k < 4; k++) {
            for (uint32_t t = 0; t < max_threads; t++) {
                task[k][t] = 0;
            }
        }
    }
};

template <class S, class pre, class post, class mac>
struct engine_forkjoin {
    using C      = composition<pre, post, mac>;
    using W      = typename C::W;
    using A      = typename C::A;
    using P      = typename C::P;
    using M      = typename C::M;
    using args_t = arguments<pre, post>;

    context           c;
    args_t            p;
    plan              pl;
    layout            L;
    buffers<C>        B;
    S                 sched;
    mac_dispatcher<M> hmx;
    phase             ph;
    uint32_t          n_stored;  // jobs [0, n_stored) written to DDR
    uint32_t          n_w_threads;
    uint32_t          ready[max_raw_bufs][max_threads];  // job + 1 once a slice of a raw slot landed
    fj_stats          st;

    const chunk_grid & grid() const { return sched.grid; }

    int run(const context & ctx, const args_t & args, const plan & plan_) {
        c  = ctx;
        p  = args;
        pl = plan_;
        L  = build_layout(p.s, pl, C::fmt);
        if (L.total > c.ctx->vtcm_size) {
            FARF(ERROR, "hmx-mm: VTCM overflow: need %u have %u", (unsigned) L.total, (unsigned) c.ctx->vtcm_size);
            return -1;
        }
        B     = buffers<C>(c.vtcm, L, pl);
        sched = S(p.s, pl, C::tile_k);
        hmx.attach(c.ctx->hmx_queue);
        n_stored    = 0;
        n_w_threads = n_w_dma_threads(pl);
        ph.overflow = false;
        if (pl.mc > max_act_rows || n_w_threads < 1) {
            return -1;
        }
        for (uint32_t i = 0; i < max_raw_bufs; i++) {
            for (uint32_t t = 0; t < max_threads; t++) {
                ready[i][t] = 0;
            }
        }

        M::init_scales(B.scales);

        const uint32_t J = grid().n_jobs;
        const uint32_t D = pl.n_raw_bufs;
        st.reset();
        const uint64_t t_start = hex_get_cycles();

        for (uint32_t j = 0; j < J; j++) {
            step_plan sp;
            sched.step(j, &sp);

            const uint64_t t0 = hex_get_cycles();
            hmx.wait_until(sp.wait_mac);
            st.mac_wait += hex_get_cycles() - t0;

            ph.reset();
            // raw slot (j-1) % D was consumed by the previous phase: refill it with job j+D-1
            ph.w_push_first = (j == 0) ? 0 : j + D - 1;
            ph.w_push_last  = min_u32(j + D, J);
            ph.w_pop_job    = j;
            if (sp.act_mi != no_act) {
                add_act_tasks(sp.act_mi, sp.act_r0, sp.act_r1);  // DMA tasks: act threads only
            }
            ph.close_dma();
            add_weight_tasks(j);
            add_pending_store_tasks();
            run_phase();

            submit_mac(j);
        }

        // drain: store(J-2) overlaps MAC(J-1), then store(J-1)
        if (J >= 2) {
            hmx.wait_until(J - 1);
            ph.reset();
            ph.close_dma();
            add_pending_store_tasks();
            run_phase();
        }
        hmx.wait_until(J);
        ph.reset();
        ph.close_dma();
        add_pending_store_tasks();
        run_phase();

        if (ph.overflow) {
            FARF(ERROR, "hmx-mm: task list overflow (mc %u nc %u)", pl.mc, pl.nc);
            return -1;
        }
        st.total = hex_get_cycles() - t_start;
        report();
        return 0;
    }

    void report() const {
#ifdef HMX_MM_STATS
        uint64_t sum[4] = { 0, 0, 0, 0 };
        uint64_t worst[4] = { 0, 0, 0, 0 };
        for (uint32_t k = 0; k < 4; k++) {
            for (uint32_t t = 0; t < c.n_threads; t++) {
                sum[k] += st.task[k][t];
                if (st.task[k][t] > worst[k]) worst[k] = st.task[k][t];
            }
        }
        FARF(ALWAYS, "hmx-mm-fj: m %u k %u n %u mc %u nc %u jobs %u wth %u | total %u mac-wait %u phase %u (n %u) | "
             "dequant sum %u max %u | store sum %u max %u | act sum %u max %u | dma-wait sum %u max %u",
             p.s.m, p.s.k, p.s.n, pl.mc, pl.nc, grid().n_jobs, n_w_threads,
             (unsigned) st.total, (unsigned) st.mac_wait, (unsigned) st.phase, st.n_phases,
             (unsigned) sum[0], (unsigned) worst[0], (unsigned) sum[1], (unsigned) worst[1], (unsigned) sum[2], (unsigned) worst[2],
             (unsigned) sum[3], (unsigned) worst[3]);
#endif
    }

    // ---- weight stream ----
    // column tiles [ct0, ct1) of the chunk of job j handled by weight DMA thread w (1..P)
    void slice_of(uint32_t j, uint32_t w, uint32_t * ct0, uint32_t * ct1) const {
        const uint32_t n_ct = grid().n_col_tiles(grid().job_ni(j));
        *ct0 = (w - 1) * n_ct / n_w_threads;
        *ct1 = w * n_ct / n_w_threads;
    }

    void push_weight_slice(uint32_t j, uint32_t w) {
        uint32_t ct0, ct1;
        slice_of(j, w, &ct0, &ct1);
        if (ct1 == ct0) {
            return;
        }
        const uint32_t   n_kt  = grid().n_kt;
        const size_t     off   = (size_t) ct0 * n_kt * W::raw_tile_bytes;
        const size_t     bytes = (size_t) (ct1 - ct0) * n_kt * W::raw_tile_bytes;
        const dma_addr_t src   = p.w.tile_addr(grid().n_col0(grid().job_ni(j)) / tile_dim, n_kt) + off;
        dma_queue_push(c.dma(w), dma_make_data(B.raw[j % pl.n_raw_bufs] + off, src), bytes, bytes, bytes, 1);
    }

    void pop_weight_slice(uint32_t j, uint32_t w) {
        uint32_t ct0, ct1;
        slice_of(j, w, &ct0, &ct1);
        if (ct1 != ct0) {
            dma_queue_pop(c.dma(w));
        }
        atomic_store_rel(&ready[j % pl.n_raw_bufs][w], j + 1);
    }

    void wait_weight_slice(uint32_t j, uint32_t w) const {
        while (atomic_load_acq(&ready[j % pl.n_raw_bufs][w]) != j + 1) {
            hex_pause();
        }
    }

    // ---- MAC stage ----
    void submit_mac(uint32_t j) {
        const uint32_t mi = grid().job_mi(j);
        const uint32_t ni = grid().job_ni(j);
        typename M::job jb;
        jb.out    = B.o[j % n_o_bufs];
        jb.act    = B.act[mi % 2];
        jb.w      = B.w[j % n_w_bufs];
        jb.scales = B.scales;
        jb.n_rt   = grid().m_row_tiles(mi);
        jb.n_ct   = grid().n_col_tiles(ni);
        jb.n_kt   = grid().n_kt;
        hmx.submit(j, jb);
    }

    // ---- phase construction ----
    void add_act_tasks(uint32_t mi, uint32_t r0, uint32_t r1) {
        for (uint32_t r = r0; r < r1; r += tile_dim) {
            ph.add(task{ task_kind::act, mi, 0, r, min_u32(r + tile_dim, r1) });
        }
    }

    void add_weight_tasks(uint32_t j) {
        const uint32_t tiles = grid().n_col_tiles(grid().job_ni(j)) * grid().n_kt;
        uint32_t per = align_up_u32(ceil_div_u32(tiles, max_weight_tasks), W::piece_align);
        if (per < weight_min_tiles) {
            per = weight_min_tiles;
        }
        for (uint32_t w = 1; w <= n_w_threads; w++) {
            uint32_t ct0, ct1;
            slice_of(j, w, &ct0, &ct1);
            const uint32_t t_end = ct1 * grid().n_kt;
            for (uint32_t t = ct0 * grid().n_kt; t < t_end; t += per) {
                ph.add(task{ task_kind::weight, j, w, t, min_u32(t + per, t_end) });
            }
        }
    }

    void add_pending_store_tasks() {
        for (; n_stored < hmx.n_popped; n_stored++) {
            add_store_tasks(n_stored);
        }
    }

    void add_store_tasks(uint32_t j) {
        const uint32_t mi   = grid().job_mi(j);
        const uint32_t ni   = grid().job_ni(j);
        const uint32_t n_rt = grid().m_row_tiles(mi);
        const uint32_t n_ct = grid().n_col_tiles(ni);
        uint32_t block = ceil_div_u32(n_rt * n_ct, max_store_tasks);
        if (block < store_ct_block) {
            block = store_ct_block;
        }
        for (uint32_t rt = 0; rt < n_rt; rt++) {
            for (uint32_t ct = 0; ct < n_ct; ct += block) {
                ph.add(task{ task_kind::store, j, rt, ct, min_u32(ct + block, n_ct) });
            }
        }
    }

    // ---- phase execution ----
    void run_phase() {
        if (ph.n == 0) {
            return;
        }
        const uint64_t t0 = hex_get_cycles();
        work_queue_run(c.ctx->work_queue, phase_main, this, c.n_threads);
        st.phase += hex_get_cycles() - t0;
        st.n_phases++;
    }

    static void phase_main(unsigned int n, unsigned int i, void * data) {
        engine_forkjoin * self = (engine_forkjoin *) data;
        phase &           ph   = self->ph;
        (void) n;
        if (i >= 1 && i <= self->n_w_threads) {
            for (uint32_t j = ph.w_push_first; j < ph.w_push_last; j++) {
                self->push_weight_slice(j, i);
            }
            if (ph.w_pop_job != no_task_job) {
                const uint64_t t0 = hex_get_cycles();
                self->pop_weight_slice(ph.w_pop_job, i);
                self->st.task[3][i] += hex_get_cycles() - t0;
            }
        } else if (i > self->n_w_threads) {
            for (;;) {
                const uint32_t t = atomic_inc(&ph.next_dma);
                if (t >= ph.n_dma) {
                    break;
                }
                self->run_task(ph.tasks[t], i);
            }
        }
        for (;;) {
            const uint32_t t = ph.n_dma + atomic_inc(&ph.next_any);
            if (t >= ph.n) {
                break;
            }
            self->run_task(ph.tasks[t], i);
        }
    }

    void run_task(const task & t, uint32_t thread) {
        const uint64_t t0 = hex_get_cycles();
        run_task_impl(t, thread, c.trace(thread));
        st.task[(uint32_t) t.kind][thread] += hex_get_cycles() - t0;
    }

    void run_task_impl(const task & t, uint32_t thread, htp_thread_trace * tr) {
        switch (t.kind) {
            case task_kind::weight: {
                wait_weight_slice(t.idx, t.rt);
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) t.begin);
                W::convert(B.raw[t.idx % pl.n_raw_bufs], B.w[t.idx % n_w_bufs], t.begin, t.end);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) t.begin);
                break;
            }
            case task_kind::store: {
                const uint32_t mi  = grid().job_mi(t.idx);
                const uint32_t ni  = grid().job_ni(t.idx);
                const uint32_t row = mi * pl.mc + t.rt * tile_dim;
                const uint32_t col = grid().n_col0(ni) + t.begin * tile_dim;
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) t.idx);
                P::store(B.o[t.idx % n_o_bufs], grid().n_col_tiles(ni), t.rt, t.begin, t.end,
                         min_u32(tile_dim, grid().m_rows(mi) - t.rt * tile_dim), p.o, row, col);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) t.idx);
                break;
            }
            case task_kind::act: {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) t.begin);
                stage_and_convert_act(thread, t.idx, t.begin, t.end);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) t.begin);
                break;
            }
        }
    }

    // rows [r0, r1) of m-chunk mi: DMA row blocks into the thread's staging halves two steps ahead,
    // convert each block when it lands
    void stage_and_convert_act(uint32_t thread, uint32_t mi, uint32_t r0, uint32_t r1) {
        using in_t = typename A::in_t;
        dma_queue *    q         = c.dma(thread);
        uint8_t *      stage     = B.act_stage[thread];
        const uint32_t R         = pl.act_rows_per_dma;
        const uint32_t valid     = grid().m_rows(mi);
        const size_t   row_bytes = (size_t) p.s.k * sizeof(in_t);
        const uint32_t n_steps   = ceil_div_u32(r1 - r0, R);

        auto rows_at = [&](uint32_t r) -> uint32_t {
            return r < valid ? min_u32(R, valid - r) : 0;
        };
        auto push = [&](uint32_t s) {
            const uint32_t r  = r0 + s * R;
            const uint32_t nv = rows_at(r);
            if (nv) {
                uint8_t * dst = stage + (s & 1) * R * row_bytes;
                dma_queue_push(q, dma_make_data(dst, p.a.row_addr(mi * pl.mc + r)), row_bytes, p.a.nb1, row_bytes, nv);
            }
        };

        push(0);
        if (n_steps > 1) {
            push(1);
        }
        for (uint32_t s = 0; s < n_steps; s++) {
            const uint32_t r   = r0 + s * R;
            const uint32_t nv  = rows_at(r);
            const in_t *   buf = (const in_t *) (stage + (s & 1) * R * row_bytes);
            if (nv) {
                buf = (const in_t *) dma_queue_pop(q).dst;
            }
            A::convert(buf, p.s.k, R, nv, B.act[mi % 2], r);
            if (s + 2 < n_steps) {
                push(s + 2);
            }
        }
    }
};

}  // namespace hmx_mm

#endif  // HMX_MM_ENGINE_FORKJOIN_H
