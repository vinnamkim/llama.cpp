#ifndef HMX_MM_COMPONENTS_H
#define HMX_MM_COMPONENTS_H

// Components of the C++ HMX matmul pipeline for Q4_0 weights x F32 activations -> F32 on fp16 HMX.
// Each component satisfies one contract of hmx-mm-core.h and is a plain type: traits plus static
// functions that run on one HVX thread (or, for the MAC unit, on the HMX thread).
//
// To add a format, add a type here that satisfies the matching concept and compose it in
// hmx-mm-cpp.cc; the engines and the planner adapt through the traits.

#include "hmx-mm-core.h"

namespace hmx_mm {

// --- weight pre-processor: Q4_0 -> fp16 tiles ------------------------------------------------------
//
// Raw tile: 512 nibble bytes + 32 fp16 scales (576 bytes). Raw byte 32 * cp + row holds k = 2 * cp (low
// nibble) and k = 2 * cp + 1 (high nibble) of row n. Output vector i holds k pair (2i, 2i + 1) for the 32
// rows: [n0 k2i, n0 k2i+1, n1 k2i, ...].
// Raw tiles are contiguous (one DMA per slice), so only even tiles are 128-byte aligned; an odd tile is
// assembled with valign from the 9 aligned vectors of its pair. Hence piece_align = 2.
struct weight_prep_q4_0 {
    using tile_t = __fp16;

    static constexpr uint32_t tile_k          = HTP_MM_HMX_TILE_N_COLS;         // 32
    static constexpr uint32_t raw_tile_bytes  = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;   // 576
    static constexpr uint32_t piece_align     = 2;
    static constexpr uint32_t cost_per_thread = 10;  // weights per cycle per thread, SM8850
    static constexpr uint32_t tile_elems      = tile_dim * tile_k;

    struct source {
        dma_addr_t base;  // repacked tiles: tile (ct, kt) at (ct * n_kt + kt) * raw_tile_bytes

        dma_addr_t tile_addr(uint32_t ct, uint32_t n_kt) const {
            return base + (dma_addr_t) ct * n_kt * raw_tile_bytes;
        }
    };

    // vlut16 table: fp16(q - 8) at entry 2q (odd entries unused), same layout as iq4_nl_to_fp16_lut
    static const HVX_Vector & lut() {
        static const __fp16 table[64] __attribute__((aligned(VLEN))) = {
            -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
        };
        return *(const HVX_Vector *) table;
    }

    // 128 raw bytes (8 k values x 32 rows) -> 4 output vectors
    static inline void group(HVX_Vector vq, HVX_Vector v_scale, HVX_Vector v_lut, HVX_Vector * restrict out) {
        const HVX_Vector mask_lo = Q6_Vb_vsplat_R(0x0F);
        const HVX_Vector lo = Q6_V_vand_VV(vq, mask_lo);
        const HVX_Vector hi = Q6_Vub_vlsr_VubR(vq, 4);

        // interleave even/odd k bytes, look the nibbles up as fp16 (q - 8), scale
        const HVX_VectorPair shuf = Q6_W_vshuff_VVR(hi, lo, -1);
        const HVX_VectorPair w01  = Q6_Wh_vlut16_VbVhR(Q6_Vb_vshuff_Vb(Q6_V_lo_W(shuf)), v_lut, 0);
        const HVX_VectorPair w23  = Q6_Wh_vlut16_VbVhR(Q6_Vb_vshuff_Vb(Q6_V_hi_W(shuf)), v_lut, 0);

        out[0] = hvx_vec_mul_f16_f16(Q6_V_lo_W(w01), v_scale);
        out[1] = hvx_vec_mul_f16_f16(Q6_V_hi_W(w01), v_scale);
        out[2] = hvx_vec_mul_f16_f16(Q6_V_lo_W(w23), v_scale);
        out[3] = hvx_vec_mul_f16_f16(Q6_V_hi_W(w23), v_scale);
    }

    static inline HVX_Vector dup_scales(HVX_Vector v_sc) {
        return Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));  // per-row scale, duplicated for the k pair
    }

    // v: the 9 aligned vectors of a tile pair
    static inline void even_tile(const HVX_Vector * v, HVX_Vector * restrict out) {
        const HVX_Vector v_lut   = lut();
        const HVX_Vector v_scale = dup_scales(v[4]);
        group(v[0], v_scale, v_lut, out + 0);
        group(v[1], v_scale, v_lut, out + 4);
        group(v[2], v_scale, v_lut, out + 8);
        group(v[3], v_scale, v_lut, out + 12);
    }

    static inline void odd_tile(const HVX_Vector * v, HVX_Vector * restrict out) {
        const HVX_Vector v_lut   = lut();
        const HVX_Vector v_scale = dup_scales(Q6_V_vror_VR(v[8], 64));
        group(Q6_V_valign_VVR(v[5], v[4], 64), v_scale, v_lut, out + 0);
        group(Q6_V_valign_VVR(v[6], v[5], 64), v_scale, v_lut, out + 4);
        group(Q6_V_valign_VVR(v[7], v[6], 64), v_scale, v_lut, out + 8);
        group(Q6_V_valign_VVR(v[8], v[7], 64), v_scale, v_lut, out + 12);
    }

    static void convert(const uint8_t * restrict raw, tile_t * restrict tiles, uint32_t t0, uint32_t t1) {
        uint32_t t = t0;
        if (t < t1 && (t & 1)) {
            odd_tile((const HVX_Vector *) (raw + (size_t) (t - 1) * raw_tile_bytes), (HVX_Vector *) (tiles + (size_t) t * tile_elems));
            t++;
        }
        for (; t + 2 <= t1; t += 2) {
            const HVX_Vector * v = (const HVX_Vector *) (raw + (size_t) t * raw_tile_bytes);
            HVX_Vector * out = (HVX_Vector *) (tiles + (size_t) t * tile_elems);
            even_tile(v, out);
            odd_tile(v, out + tile_elems / VLEN_FP16);
        }
        if (t < t1) {
            even_tile((const HVX_Vector *) (raw + (size_t) t * raw_tile_bytes), (HVX_Vector *) (tiles + (size_t) t * tile_elems));
        }
    }
};

// --- activation pre-processor: f32 rows -> fp16 tiles ---------------------------------------------
//
// Each row pair (2i, 2i + 1) of a tile row is interleaved into tile vector i: [r0 k0, r1 k0, r0 k1, ...].
struct act_prep_f32 {
    using in_t   = float;
    using tile_t = __fp16;

    static constexpr uint32_t tile_k          = HTP_MM_HMX_TILE_N_COLS;  // 32
    static constexpr uint32_t cost_per_thread = 2;  // activations per cycle per thread (DMA staged)
    static constexpr uint32_t tile_elems      = tile_dim * tile_k;

    struct source {
        dma_addr_t base;  // row 0
        uint32_t   nb1;   // bytes between rows

        dma_addr_t row_addr(uint32_t r) const { return base + (dma_addr_t) r * nb1; }
    };

    static void convert(const in_t * restrict rows, uint32_t k, uint32_t n_rows, uint32_t n_valid,
                        tile_t * restrict tiles, uint32_t row0) {
        const uint32_t   n_kt  = k / tile_k;
        const HVX_Vector vzero = Q6_V_vzero();

        for (uint32_t pr = 0; pr < n_rows; pr += 2) {
            const uint32_t row   = row0 + pr;
            const bool     v0_ok = pr < n_valid;
            const bool     v1_ok = pr + 1 < n_valid;
            const HVX_Vector * p0 = (const HVX_Vector *) (rows + (size_t) pr * k);
            const HVX_Vector * p1 = (const HVX_Vector *) (rows + (size_t) (pr + 1) * k);
            HVX_Vector * tile = (HVX_Vector *) (tiles + (size_t) (row / tile_dim) * n_kt * tile_elems);
            const uint32_t vi = (row % tile_dim) / 2;

            for (uint32_t kt = 0; kt < n_kt; kt++) {
                const HVX_Vector v0 = v0_ok ? p0[kt] : vzero;
                const HVX_Vector v1 = v1_ok ? p1[kt] : vzero;
                tile[vi] = hvx_vec_f32_to_f16_shuff(v0, v1);
                tile += tile_elems / VLEN_FP16;
            }
        }
    }
};

// --- post-processor: fp16 output tiles -> f32 rows in DDR -----------------------------------------

struct post_processor_f32 {
    using in_t  = __fp16;
    using out_t = float;

    static constexpr uint32_t cost_per_thread = 1;  // outputs per cycle per thread (DDR stores)
    static constexpr uint32_t tile_elems      = tile_dim * tile_dim;

    struct sink {
        out_t *  base;
        uint32_t stride;  // elements between rows
        uint32_t cols;    // valid columns (real N)

        out_t * at(uint32_t row, uint32_t col) const { return base + (size_t) row * stride + col; }
    };

    // tiles: output tiles of the chunk, tile (rt, ct) at (rt * n_ct + ct) * tile_elems
    static void store(const in_t * restrict tiles, uint32_t n_ct, uint32_t rt, uint32_t ct0, uint32_t ct1,
                      uint32_t rows_valid, const sink & dst, uint32_t row, uint32_t col) {
        if (col >= dst.cols) {
            return;
        }
        const HVX_Vector one = hvx_vec_splat_f16(1.0);
        const HVX_Vector * v = (const HVX_Vector *) (tiles + (size_t) rt * n_ct * tile_elems);
        out_t * restrict d = dst.at(row, col);
        const uint32_t cols_valid = dst.cols - col;

        for (uint32_t r = 0; r < rows_valid; r += 2) {
            const uint32_t vi  = r / 2;
            const bool     two = r + 1 < rows_valid;
            out_t * d0 = d + (size_t) r * dst.stride;
            out_t * d1 = d0 + dst.stride;

            uint32_t cols = cols_valid;
            for (uint32_t ct = ct0; ct < ct1 && cols > 0; ct++) {
                const HVX_Vector     t  = v[ct * (tile_elems / VLEN_FP16) + vi];
                const HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(t, one);
                const HVX_Vector     o0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
                const HVX_Vector     o1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                const uint32_t       cc = min_u32(tile_dim, cols);
                out_t * c0 = d0 + (ct - ct0) * tile_dim;
                out_t * c1 = d1 + (ct - ct0) * tile_dim;

                if (cc == tile_dim) {
                    hvx_vmemu(c0) = o0;
                    if (two) {
                        hvx_vmemu(c1) = o1;
                    }
                } else {
                    hvx_vec_store_u(c0, cc * sizeof(out_t), o0);
                    if (two) {
                        hvx_vec_store_u(c1, cc * sizeof(out_t), o1);
                    }
                }
                cols -= cc;
            }
        }
    }
};

// --- MAC unit: fp16 x fp16 -> fp16 on HMX ---------------------------------------------------------

struct hmx_mac_f16 {
    using act_t = __fp16;
    using w_t   = __fp16;
    using out_t = __fp16;

    static constexpr uint32_t tile_k             = HTP_MM_HMX_TILE_N_COLS;  // 32
    static constexpr uint32_t scales_bytes       = 2 * VLEN;                // column scales + bias vectors
    static constexpr uint32_t cost_macs          = 2500;                    // fp16 MACs per cycle
    static constexpr uint32_t cost_job_overhead  = 6000;
    static constexpr uint32_t cost_tile_overhead = 64;
    static constexpr uint32_t tile_elems         = tile_dim * tile_k;
    static constexpr uint32_t tile_bytes         = tile_elems * sizeof(w_t);

    struct job {
        out_t *       out;     // n_rt x n_ct tiles
        const act_t * act;     // n_rt x n_kt tiles
        const w_t *   w;       // n_ct x n_kt tiles
        const void *  scales;
        uint32_t      n_rt;
        uint32_t      n_ct;
        uint32_t      n_kt;
    };

    // output = accumulator x 1.0 + 0
    static void init_scales(void * scales) {
        hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
    }

    // out[rt][ct] = sum_kt act[rt][kt] x w[ct][kt], one HMX accumulator per output tile
    static void dot(const job & jb) {
        asm volatile(HMX_SET_BIAS("%0") :: "r"((uintptr_t) jb.scales));

        const size_t kstride = (size_t) jb.n_kt * tile_elems;
        for (uint32_t rt = 0; rt < jb.n_rt; rt++) {
            for (uint32_t ct = 0; ct < jb.n_ct; ct++) {
                const act_t * a   = jb.act + rt * kstride;
                const w_t *   b   = jb.w + ct * kstride;
                out_t *       out = jb.out + ((size_t) rt * jb.n_ct + ct) * (tile_dim * tile_dim);

                asm volatile(HMX_CLRACC_F16());
                // the mxmem range field covers at most 32 tiles per instruction
                for (uint32_t rem = jb.n_kt; rem > 0;) {
                    const uint32_t nt    = min_u32(rem, 32);
                    const uint32_t range = nt * tile_bytes - 1;
                    asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(a), "r"(b));
                    a   += nt * tile_elems;
                    b   += nt * tile_elems;
                    rem -= nt;
                }
                asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out), "r"(0) : "memory");
            }
        }
    }
};

static_assert(weight_prep<weight_prep_q4_0>);
static_assert(act_prep<act_prep_f32>);
static_assert(post_proc<post_processor_f32>);
static_assert(mac_unit<hmx_mac_f16>);

}  // namespace hmx_mm

#endif  // HMX_MM_COMPONENTS_H
