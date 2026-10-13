// C entry point of the C++ HMX matmul pipeline: Q4_0 x F32 -> F32 on fp16 HMX.

#include "hmx-mm-pipeline.h"

extern "C" {
#include "htp-tensor.h"
#include "matmul-ops.h"
#include "hmx-mm-cpp.h"
}

using namespace hmx_mm;

// the composition this kernel runs; the host planned with the matching formats constant
using pre  = pre_processor<weight_prep_q4_0, act_prep_f32>;
using post = post_processor_f32;
using mac  = hmx_mac_f16;
static_assert(composition<pre, post, mac>::fmt == formats::q4_0_f32_f16, "formats::q4_0_f32_f16 is out of date");

static bool plan_from_kparams(const htp_mm_kernel_params * kp, uint32_t n_threads, plan * pl) {
    pl->kind             = (sched_kind) kp->pipeline;
    pl->mc               = (uint32_t) kp->m_chunk;
    pl->nc               = (uint32_t) kp->n_chunk;
    pl->n_raw_bufs       = kp->n_prefetch;
    pl->n_a_bufs         = kp->n_act_bufs;
    pl->act_rows_per_dma = kp->act_dma_rows;
    pl->act_threads      = kp->act_dma_threads;
    pl->ramp             = kp->ramp;
    pl->engine           = kp->engine;
    pl->n_threads        = n_threads;
    pl->vtcm_bytes       = (size_t) kp->vtcm_size;

    if (pl->kind != sched_kind::decode && pl->kind != sched_kind::prefill) return false;
    if (pl->mc == 0 || pl->mc % tile_dim || pl->nc == 0 || pl->nc % tile_dim) return false;
    if (pl->n_raw_bufs < 2 || pl->n_raw_bufs > max_raw_bufs) return false;
    if (pl->n_a_bufs < 1 || pl->n_a_bufs > 2) return false;
    if (pl->act_rows_per_dma != 2 && pl->act_rows_per_dma != 4) return false;
    if (pl->n_threads < 3 || pl->act_threads < 1) return false;
    if (pl->engine == 0 && pl->act_threads + 2 > pl->n_threads) return false;
    if (pl->engine == 1 && pl->act_threads > pl->n_threads) return false;
    return true;
}

extern "C" int op_matmul_hmx_cpp(struct htp_ops_context * octx) {
    const htp_mm_kernel_params * kp = (const htp_mm_kernel_params *) octx->kernel_params;
    const htp_tensor * src0 = octx->src[0];
    const htp_tensor * src1 = octx->src[1];
    const htp_tensor * dst  = octx->dst;
    htp_context * ctx = octx->ctx;

    if (ctx->mdev.count > 1 || !ctx->hmx_queue) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src0->type != HTP_TYPE_Q4_0 || !(src0->flags & HTP_TENSOR_REPACK) || src1->type != HTP_TYPE_F32) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    arguments<pre, post> args;
    args.s = { src1->ne[1], src0->ne[0], align_up_u32(src0->ne[1], tile_dim) };
    args.w = { (dma_addr_t) src0->data };
    args.a = { (dma_addr_t) src1->data, src1->nb[1] };
    args.o = { (post::out_t *) htp_tensor_data(dst), dst->nb[1] / sizeof(post::out_t), dst->ne[0] };

    if (args.s.k % tile_dim || src1->ne[0] != args.s.k || src1->nb[0] != sizeof(pre::act::in_t) ||
        dst->nb[0] != sizeof(post::out_t) || dst->ne[1] != args.s.m || args.s.m == 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    plan pl;
    if (!plan_from_kparams(kp, octx->n_threads, &pl)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    FARF(HIGH, "hmx-mm-cpp: m %u k %u n %u kind %u eng %u mc %u nc %u raw %u abuf %u th %u vtcm %u",
         args.s.m, args.s.k, args.s.n, (unsigned) pl.kind, pl.engine, pl.mc, pl.nc, pl.n_raw_bufs, pl.n_a_bufs, pl.n_threads,
         (unsigned) pl.vtcm_bytes);

    context c;
    c.ctx       = ctx;
    c.vtcm      = ctx->vtcm_base;
    c.n_threads = octx->n_threads;

    int ret;
    if (pl.engine) {
        ret = execution<sched_stream, pre, post, mac>(c, args, pl);
    } else if (pl.kind == sched_kind::decode) {
        ret = execution<sched_decode, pre, post, mac>(c, args, pl);
    } else {
        ret = execution<sched_prefill, pre, post, mac>(c, args, pl);
    }
    return ret == 0 ? HTP_STATUS_OK : HTP_STATUS_INTERNAL_ERR;
}
