#ifndef HMX_MM_PLAN_H
#define HMX_MM_PLAN_H

// Planner of the C++ HMX matmul pipeline (hmx-mm-pipeline.h).
//
// Header-only, C++17, free of Hexagon intrinsics: the host (ggml-hexagon.cpp) and the HTP kernel build the
// same plan, the same chunking and the same VTCM layout from the same inputs. Everything that depends on
// the pipeline components (element sizes, raw tile size, throughputs) enters through `format`, so the
// planner itself knows nothing about Q4_0 or fp16.

#include <stdint.h>
#include <stddef.h>

#include "matmul-ops.h"

namespace hmx_mm {

enum : uint32_t {
    tile_dim     = HTP_MM_HMX_TILE_N_ROWS,  // 32: an output tile is tile_dim x tile_dim, act / weight tiles tile_dim x tile_k
    tile_align   = HTP_MM_HMX_TILE_SIZE,    // 2048: HMX tile buffers are aligned to this
    n_w_bufs     = 2,                       // weight chunk buffers: dequant(j) || mac(j-1)
    n_o_bufs     = 3,                       // output chunk buffers: mac(j) || store(j-2)
    max_raw_bufs = 4,
    max_nc       = 1024,
    max_mc       = 2048,
    max_threads  = HTP_MAX_NTHREADS,
};

// Everything the planner needs to know about a component configuration. The device derives it from the
// component traits (composition<...>::fmt in hmx-mm-core.h); the host uses the matching constant below.
struct format {
    uint32_t tile_k;              // k depth of one act / weight tile
    uint32_t act_in_bytes;        // activation element in DDR
    uint32_t act_bytes;           // activation tile element
    uint32_t w_bytes;             // weight tile element
    uint32_t out_bytes;           // output tile element
    uint32_t raw_tile_bytes;      // one raw weight tile (DDR layout, DMAed into VTCM unchanged)
    uint32_t scales_bytes;        // MAC unit column scale table
    // cost model: throughputs per cycle, measured on target
    uint32_t dequant_per_thread;  // raw weight elements -> tiles, per HVX thread
    uint32_t act_per_thread;      // activation elements -> tiles, per HVX thread
    uint32_t store_per_thread;    // output elements -> DDR, per HVX thread
    uint32_t macs;                // MAC unit multiply-accumulates
    uint32_t mac_job_overhead;    // cycles: queue hand-off, MAC pipeline ramp, bookkeeping per job
    uint32_t mac_tile_overhead;   // cycles: accumulator clear / store per output tile

    constexpr bool operator==(const format & o) const {
        return tile_k == o.tile_k && act_in_bytes == o.act_in_bytes && act_bytes == o.act_bytes && w_bytes == o.w_bytes &&
               out_bytes == o.out_bytes && raw_tile_bytes == o.raw_tile_bytes && scales_bytes == o.scales_bytes &&
               dequant_per_thread == o.dequant_per_thread && act_per_thread == o.act_per_thread &&
               store_per_thread == o.store_per_thread && macs == o.macs && mac_job_overhead == o.mac_job_overhead &&
               mac_tile_overhead == o.mac_tile_overhead;
    }
};

namespace formats {
// weight_prep_q4_0 + act_prep_f32 + hmx_mac_f16 + post_processor_f32 (hmx-mm-components.h), SM8850 at ~2.1 GHz
inline constexpr format q4_0_f32_f16 = { 32, 4, 2, 2, 2, HTP_MM_WEIGHT_TILE_SIZE_Q4_0, 256, 10, 2, 1, 2500, 6000, 64 };
}  // namespace formats

enum class sched_kind : uint8_t {
    decode  = 0,  // fork-join: one activation chunk, weight chunks streamed
    prefill = 1,  // fork-join: m-outer / n-inner, next activation chunk prepared under HMX compute
};

struct shape {
    uint32_t m;  // activation rows
    uint32_t k;  // reduction dim, multiple of tile_k
    uint32_t n;  // weight rows, padded to tile_dim by the repack
};

struct limits {
    uint32_t n_threads;
    size_t   vtcm_bytes;
};

struct plan {
    sched_kind kind;              // fork-join scheduler (engine 0)
    uint32_t   mc;                // activation rows per chunk, multiple of tile_dim
    uint32_t   nc;                // largest weight chunk (rows), multiple of tile_dim
    uint32_t   n_raw_bufs;        // weight DMA prefetch depth
    uint32_t   n_a_bufs;          // activation buffers (2 = act prep of the next m-chunk overlaps compute)
    uint32_t   act_rows_per_dma;  // rows per activation DMA block
    uint32_t   act_threads;       // threads that own an activation staging area
    uint32_t   ramp;              // 1: first weight chunks grow 32, 64, ... to shorten the pipeline fill
    uint32_t   engine;            // 0: fork-join step engine, 1: dataflow engine (hmx-mm-pipeline.h)
    uint32_t   n_threads;
    size_t     vtcm_bytes;
};

// Byte offsets from the VTCM base. Every region is a multiple of tile_align so HMX tiles stay aligned.
struct layout {
    size_t off_scales;     // scales_bytes used, padded to one tile
    size_t off_act;        // n_a_bufs x act_bytes
    size_t off_act_stage;  // act_threads x act_stage_bytes (DDR rows landed by DMA)
    size_t off_raw;        // n_raw_bufs x raw_bytes
    size_t off_w;          // n_w_bufs x w_bytes
    size_t off_o;          // n_o_bufs x o_bytes
    size_t act_bytes;
    size_t act_stage_bytes;
    size_t raw_bytes;
    size_t w_bytes;
    size_t o_bytes;
    size_t total;
};

static inline uint32_t align_up_u32(uint32_t v, uint32_t a) {
    return (v + a - 1) / a * a;
}

static inline size_t align_up_sz(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint64_t max_u64(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

// Thread roles. Every thread owns one DMA ring and one DMA engine moves about 20 GB/s, so weight chunks
// are split into slices streamed by P threads.
//   fork-join engine: a ring is popped blocking in FIFO order, so a thread issues one kind of DMA only:
//                     thread 0 (main) none, threads 1..P weight slices, threads P+1..P+act_threads activations
//   dataflow engine:  rings are serviced non-blocking with tags, so threads 1..n-1 stream weights and the
//                     first act_threads threads also convert activations
static inline uint32_t n_w_dma_threads(const plan & p) {
    return p.engine ? p.n_threads - 1 : p.n_threads - 1 - p.act_threads;
}

// --- weight chunking ------------------------------------------------------------------------------
// The first chunks ramp up 32, 64, 128, ... so the first HMX job starts after a tiny dequant; the
// rest of the matrix is covered by chunks of nc rows.

static inline uint32_t ramp_chunks(uint32_t nc, uint32_t ramp) {
    uint32_t r = 0;
    if (ramp) {
        for (uint32_t s = tile_dim; s < nc; s *= 2) {
            r++;
        }
    }
    return r;
}

static inline uint32_t ramp_rows(uint32_t nc, uint32_t ramp) {
    return (tile_dim << ramp_chunks(nc, ramp)) - tile_dim;  // sum of 32 * 2^i for the ramp
}

static inline uint32_t col_chunk_begin(uint32_t nc, uint32_t ramp, uint32_t i) {
    const uint32_t r = ramp_chunks(nc, ramp);
    if (i < r) {
        return (tile_dim << i) - tile_dim;
    }
    return ramp_rows(nc, ramp) + (i - r) * nc;
}

static inline uint32_t n_col_chunks(uint32_t n, uint32_t nc, uint32_t ramp) {
    uint32_t i = 0;
    while (col_chunk_begin(nc, ramp, i) < n) {
        i++;
    }
    return i;
}

static inline uint32_t col_chunk_rows(uint32_t n, uint32_t nc, uint32_t ramp, uint32_t i) {
    const uint32_t b = col_chunk_begin(nc, ramp, i);
    const uint32_t e = min_u32(col_chunk_begin(nc, ramp, i + 1), n);
    return e - b;
}

// --- layout ---------------------------------------------------------------------------------------

static inline layout build_layout(const shape & s, const plan & p, const format & f) {
    layout L = {};
    const uint32_t n_kt = s.k / f.tile_k;

    L.act_bytes       = align_up_sz((size_t) p.mc * s.k * f.act_bytes, tile_align);
    L.act_stage_bytes = align_up_sz((size_t) 2 * p.act_rows_per_dma * s.k * f.act_in_bytes, 128);
    L.raw_bytes       = align_up_sz((size_t) (p.nc / tile_dim) * n_kt * f.raw_tile_bytes, tile_align);
    L.w_bytes         = align_up_sz((size_t) p.nc * s.k * f.w_bytes, tile_align);
    L.o_bytes         = align_up_sz((size_t) p.mc * p.nc * f.out_bytes, tile_align);

    size_t off = 0;
    L.off_scales    = off; off += align_up_sz(f.scales_bytes, tile_align);
    L.off_act       = off; off += (size_t) p.n_a_bufs * L.act_bytes;
    L.off_act_stage = off; off += align_up_sz((size_t) p.act_threads * L.act_stage_bytes, tile_align);
    L.off_raw       = off; off += (size_t) p.n_raw_bufs * L.raw_bytes;
    L.off_w         = off; off += (size_t) n_w_bufs * L.w_bytes;
    L.off_o         = off; off += (size_t) n_o_bufs * L.o_bytes;
    L.total         = off;
    return L;
}

// --- cost model -----------------------------------------------------------------------------------
// Rough cycle estimates used only to rank candidate plans. Component throughputs come from `format`;
// the DMA and step constants below are properties of the engines.
// TODO: calibrate on target with the trace profiler.

struct cost_model {
    static constexpr uint64_t dma_engine_bytes = 10;    // DDR -> VTCM bytes per cycle per DMA engine
    static constexpr uint64_t dma_max_bytes    = 30;    // DDR bandwidth cap
    static constexpr uint64_t step_overhead_fj = 4000;  // fork-join barrier + HMX submit/pop
    static constexpr uint64_t step_overhead_df = 800;   // dataflow: per-job bookkeeping
};

// Cycles of one pipeline step for a chunk of `rows` weight rows. A step is bounded by the slowest of the
// MAC job, the HVX phase and the weight DMA.
struct step_cost {
    uint64_t mac;
    uint64_t phase;
    uint64_t dma;
    uint64_t bound() const {
        return max_u64(max_u64(mac, phase), dma);
    }
};

static inline step_cost chunk_cost(const shape & s, const plan & p, const format & f, uint64_t rows, uint64_t dma_rate) {
    using cm = cost_model;
    const uint64_t n_th  = p.n_threads;
    const uint64_t tiles = (p.mc / tile_dim) * (rows / tile_dim);
    step_cost c;
    c.mac   = f.mac_job_overhead + tiles * f.mac_tile_overhead + (uint64_t) p.mc * rows * s.k / f.macs;
    c.phase = rows * s.k / (f.dequant_per_thread * n_th) + (uint64_t) p.mc * rows / (f.store_per_thread * n_th) +
              (p.engine ? cm::step_overhead_df : cm::step_overhead_fj);
    c.dma   = rows * s.k * f.raw_tile_bytes / ((uint64_t) tile_dim * f.tile_k * dma_rate);
    return c;
}

// Estimated cycles of one matmul for a plan: ramp chunks explicitly, uniform chunks in closed form, plus the
// pipeline fill (first DMA + first phase) and drain (last store).
static inline uint64_t estimate_cycles(const shape & s, const plan & p, const format & f) {
    using cm = cost_model;
    const uint64_t n_th    = p.n_threads;
    const uint64_t n_dma   = p.act_threads;
    uint64_t dma_rate = cm::dma_engine_bytes * n_w_dma_threads(p);
    if (dma_rate > cm::dma_max_bytes) {
        dma_rate = cm::dma_max_bytes;
    }
    const uint32_t n_mb    = ceil_div_u32(s.m, p.mc);
    const uint32_t n_nb    = n_col_chunks(s.n, p.nc, p.ramp);
    const uint32_t n_ramp  = min_u32(ramp_chunks(p.nc, p.ramp), n_nb);
    const uint64_t act_cyc = (uint64_t) p.mc * s.k / (f.act_per_thread * n_dma);  // one m-chunk conversion

    // one m-chunk without activation work
    uint64_t per_mb = 0;
    for (uint32_t ni = 0; ni < n_ramp; ni++) {
        per_mb += chunk_cost(s, p, f, col_chunk_rows(s.n, p.nc, p.ramp, ni), dma_rate).bound();
    }
    const uint32_t n_full = n_nb - n_ramp;
    if (n_full > 0) {
        const uint64_t last_rows = col_chunk_rows(s.n, p.nc, p.ramp, n_nb - 1);
        per_mb += (uint64_t) (n_full - 1) * chunk_cost(s, p, f, p.nc, dma_rate).bound();
        per_mb += chunk_cost(s, p, f, last_rows, dma_rate).bound();
    }

    uint64_t total = per_mb * n_mb;

    // activation conversion: synchronous (MAC idle) for the first m-chunk, with one buffer for every m-chunk,
    // otherwise sliced under the previous m-chunk's steps (extra phase time, mostly hidden under the MAC)
    const uint32_t sync_chunks = (p.n_a_bufs == 1 || n_nb < 2) ? n_mb : 1;
    total += sync_chunks * act_cyc;
    if (n_mb > sync_chunks) {
        const step_cost c = chunk_cost(s, p, f, p.nc, dma_rate);
        const uint64_t slice = act_cyc / (n_nb - 1);
        const uint64_t extra = (c.phase + slice > c.bound()) ? (c.phase + slice - c.bound()) : 0;
        total += (n_mb - sync_chunks) * extra * (n_nb - 1);
    }

    // fill: first DMA and first dequant are not overlapped; drain: last store
    const uint32_t first_rows = col_chunk_rows(s.n, p.nc, p.ramp, 0);
    total += chunk_cost(s, p, f, first_rows, dma_rate).dma + first_rows * s.k / (f.dequant_per_thread * n_th);
    total += (uint64_t) p.mc * col_chunk_rows(s.n, p.nc, p.ramp, n_nb - 1) / (f.store_per_thread * n_th);
    return total;
}

// --- search ---------------------------------------------------------------------------------------

struct planner {
    shape    s;
    format   f;
    size_t   budget;
    plan     best;
    uint64_t best_cost;

    bool fits(const plan & p) const {
        return build_layout(s, p, f).total <= budget;
    }

    void consider(plan p) {
        if (!fits(p)) {
            return;
        }
        for (uint32_t ramp = 1; ramp + 1 >= 1; ramp--) {
            p.ramp = ramp;
            const uint64_t cost = estimate_cycles(s, p, f);
            if (cost < best_cost) {
                best      = p;
                best_cost = cost;
            }
            if (ramp == 0) {
                break;
            }
        }
    }

    void search_chunks(plan p) {
        const uint32_t m_pad  = align_up_u32(s.m, tile_dim);
        const uint32_t nc_cap = min_u32(max_nc, align_up_u32(s.n, tile_dim));
        static const uint32_t nc_cand[] = { 1024, 768, 512, 384, 256, 192, 128, 96, 64, 32 };
        const uint32_t n_nc_cand = sizeof(nc_cand) / sizeof(nc_cand[0]);

        if (m_pad <= tile_dim) {
            p.kind     = sched_kind::decode;
            p.mc       = m_pad;
            p.n_a_bufs = 1;
            for (uint32_t depth = 3; depth >= 2; depth--) {
                p.n_raw_bufs = depth;
                for (uint32_t i = 0; i < n_nc_cand; i++) {
                    if (nc_cand[i] > nc_cap) continue;
                    p.nc = nc_cand[i];
                    consider(p);
                }
            }
            return;
        }

        p.kind       = sched_kind::prefill;
        p.n_raw_bufs = 2;
        // mc candidates: whole M, then powers of two below it
        uint32_t mc_cand[16];
        uint32_t n_mc = 0;
        mc_cand[n_mc++] = min_u32(m_pad, max_mc);
        for (uint32_t mc = 1024; mc >= tile_dim; mc /= 2) {
            if (mc < mc_cand[0]) {
                mc_cand[n_mc++] = mc;
            }
        }
        for (uint32_t im = 0; im < n_mc; im++) {
            p.mc = mc_cand[im];
            for (uint32_t n_a = 1; n_a <= 2; n_a++) {
                if (n_a == 2 && p.mc >= m_pad) {
                    break;  // a second activation buffer only pays off with a next m-chunk
                }
                p.n_a_bufs = n_a;
                for (uint32_t i = 0; i < n_nc_cand; i++) {
                    if (nc_cand[i] > nc_cap) continue;
                    p.nc = nc_cand[i];
                    consider(p);
                }
            }
        }
    }
};

// Pick the plan with the lowest estimated cycles. Costs 1-80 us on the host; callers cache per shape.
static inline bool make_plan(const shape & s, const limits & lim, const format & f, uint32_t engine, plan * out) {
    if (s.m == 0 || s.k == 0 || s.n == 0 || s.k % f.tile_k != 0 || s.n % tile_dim != 0) {
        return false;
    }
    if (lim.n_threads < 3 || lim.n_threads > max_threads) {
        return false;
    }

    planner pr;
    pr.s         = s;
    pr.f         = f;
    pr.budget    = lim.vtcm_bytes;
    pr.best      = plan{};
    pr.best_cost = ~(uint64_t) 0;

    plan p = {};
    p.n_threads = lim.n_threads;
    p.engine    = engine ? 1 : 0;
    // activation staging competes with the weight buffers for VTCM: try a few thread counts and block sizes
    const uint32_t max_act = engine ? lim.n_threads : lim.n_threads - 2;
    const uint32_t act_threads_cand[3] = { max_act, min_u32(max_act, 4), min_u32(max_act, 2) };
    for (uint32_t i = 0; i < 3; i++) {
        if (i > 0 && act_threads_cand[i] == act_threads_cand[i - 1]) {
            continue;
        }
        p.act_threads = act_threads_cand[i];
        for (uint32_t rows = 4; rows >= 2; rows -= 2) {
            p.act_rows_per_dma = rows;
            pr.search_chunks(p);
        }
    }

    if (pr.best_cost == ~(uint64_t) 0) {
        return false;
    }
    pr.best.vtcm_bytes = build_layout(s, pr.best, f).total;
    *out = pr.best;
    return true;
}

}  // namespace hmx_mm

#endif  // HMX_MM_PLAN_H
