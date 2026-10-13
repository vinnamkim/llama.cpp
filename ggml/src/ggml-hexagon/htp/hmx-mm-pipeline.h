#ifndef HMX_MM_PIPELINE_H
#define HMX_MM_PIPELINE_H

// C++ HMX matmul pipeline for the Hexagon backend.
//
// One matmul is composed from four parts and run with
//   execution<scheduler, pre_processor, post_processor, mac_unit>(ctx, arguments, plan)
//
//   planner        (hmx-mm-plan.h)        chunk sizes, buffer depths, thread roles and the VTCM layout, chosen
//                                         on the host from the shape and the components' `format`
//   scheduler      (hmx-mm-engine-*.h)    job order, buffer slots and reuse rules; each scheduler names the
//                                         engine it runs on: sched_stream -> dataflow (barrier-free event
//                                         loop), sched_prefill / sched_decode -> fork-join steps
//   pre_processor  (hmx-mm-components.h)  HVX workers: weight tiles (raw -> MAC tiles) and activation rows
//                                         (-> MAC tiles); bundled as pre_processor<weight_prep, act_prep>
//   post_processor (hmx-mm-components.h)  HVX workers: MAC output tiles -> destination rows, epilogue
//   mac_unit       (hmx-mm-components.h)  tile matmul on the HMX thread
//
// Contracts live in hmx-mm-core.h; composition<pre, post, mac> checks that the pieces fit and derives the
// planner `format` from their traits. Engines read element types and tile sizes from the traits only, so a
// new weight format, activation type, epilogue or MAC data type is a new component, not an engine change.

#include "hmx-mm-core.h"
#include "hmx-mm-components.h"
#include "hmx-mm-engine-forkjoin.h"
#include "hmx-mm-engine-dataflow.h"

namespace hmx_mm {

template <class scheduler, class pre, post_proc post, mac_unit mac>
static inline int execution(const context & ctx, const arguments<pre, post> & args, const plan & pl) {
    using engine = typename scheduler::template engine<scheduler, pre, post, mac>;
    static engine x;  // large (counters, task lists): one instance per composition, not on the stack
    return x.run(ctx, args, pl);
}

}  // namespace hmx_mm

#endif  // HMX_MM_PIPELINE_H
