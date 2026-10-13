# C++ HMX matmul pipeline: architecture and extension guide

This document describes the composable C++ matmul pipeline for the Hexagon NPU
(`ggml/src/ggml-hexagon/htp/hmx-mm-*.h`) and how to extend it with new weight
formats, activation types, epilogues, MAC data types and schedulers.

The pipeline is opt-in and currently covers one case: Q4_0 weights x F32
activations -> F32 output on the fp16 HMX unit. The C kernels remain the default.

## Enabling and A/B switches

| Environment variable | Values | Effect |
|---|---|---|
| `GGML_HEXAGON_MM_CPP` | `0` (default), `1`, `2` | off / use the C++ pipeline for every M / only for M > `HTP_MM_HMX_MIN_NROWS` (4) |
| `GGML_HEXAGON_MM_CPP_ENGINE` | `1` (default), `0` | dataflow engine / fork-join step engine |
| `GGML_HEXAGON_OPFUSION=0` | | disables NX / ADD fusion in the C path, needed for per-op comparisons |

Build the HTP library with `-DGGML_HEXAGON_HMX_MM_STATS=ON` to print per-op cycle
breakdowns with FARF (every 32nd op). Never benchmark with it on: a FARF line
costs tens of microseconds.

Nodes that use the C++ kernel show up as `hmx-cpp` in the op profile. The host
does not fuse them into `MUL_MAT_NX` or `MUL_MAT_ADD`, so in-model decode is
slower than the C path until NX and ADD exist as pipeline components (see
"Known limits").

## The problem and the hardware

One op computes `Y[M,N] = X[M,K] * W[N,K]^T`:

- `W` lives in DDR as repacked 32x32 tiles. A Q4_0 tile is 576 bytes (512
  nibble bytes + 32 fp16 scales); tile `(ct, kt)` sits at
  `(ct * n_kt + kt) * 576`.
- `X` is F32 in DDR with an arbitrary row stride.
- `Y` is F32 in DDR.

Resources: 8 HVX threads (vector work), one HMX unit (32x32 fp16 tile matmul)
driven by its own thread through `hmx-queue`, 8 MB of VTCM, and one user DMA
engine per thread with a strict FIFO descriptor ring.

Two constraints drive the design:

1. HMX reads fp16 tiles from VTCM only. Q4_0 weights must be expanded to fp16
   in VTCM first (dequant), and that expansion is the dominant HVX cost.
2. One DMA engine moves about 20 GB/s. Weight chunks are therefore split into
   column-tile slices and fetched by several threads at once.

## Files

| File | Contents |
|---|---|
| `hmx-mm-plan.h` | Planner. Host and device, C++17, no intrinsics. `format`, `plan`, `layout`, chunking helpers, cost model, `make_plan()` |
| `hmx-mm-core.h` | Component contracts (C++20 concepts), `composition<>`, `arguments<>`, `chunk_grid`, typed `buffers<>`, `mac_dispatcher<>`, `job_counter<>`, `cycle_buckets<>` |
| `hmx-mm-components.h` | `weight_prep_q4_0`, `act_prep_f32`, `post_processor_f32`, `hmx_mac_f16` |
| `hmx-mm-engine-dataflow.h` | `sched_stream` and `engine_dataflow` (barrier-free event loop) |
| `hmx-mm-engine-forkjoin.h` | `sched_prefill`, `sched_decode` and `engine_forkjoin` (step engine) |
| `hmx-mm-pipeline.h` | Umbrella header and `execution<>()` |
| `hmx-mm-cpp.cc` | C entry point `op_matmul_hmx_cpp()`, selected by `kernel_type == HTP_MM_KERNEL_HMX_Q4_0_CPP` |
| `ggml-hexagon.cpp` | `ggml_hexagon_precompute_hmx_cpp_mm_params()`: eligibility, planner call, plan cache, kparams |

The C++ translation unit uses the `.cc` suffix on purpose: `hexagon_fun.cmake`
globs `*.cpp` and then tries to copy libc++ files the SDK does not ship. The code
is freestanding C++20 (no exceptions, RTTI or libc++); the concepts are written
without `<concepts>` or `<type_traits>`.

## How one op flows

```
host   planner: shape + format + VTCM budget -> plan (mc, nc, buffers, threads) -> kparams
device entry:   kparams -> plan, tensors -> arguments -> execution<scheduler, pre, post, mac>()
```

### Chunk grid and jobs

The activation is cut into m-chunks of `mc` rows and the weight into n-chunks of
up to `nc` rows. One (m-chunk, n-chunk) pair is a job; jobs run m-outer /
n-inner. With `ramp = 1` the first n-chunks are 32, 64, 128, ... rows so the
first MAC job starts after a tiny dequant. Decode (M <= 32) has one m-chunk and
simply streams weight chunks.

### Stages of a job

```
DDR  --DMA-->  raw tiles (VTCM)  --HVX dequant-->  W tiles (VTCM)
                                                       |
DDR  --DMA-->  X rows (VTCM)     --HVX convert-->  X tiles (VTCM)  --HMX-->  Y tiles (VTCM)
                                                                                 |
                                                       DDR Y  <--HVX store-------+
```

| Stage | Who | Component |
|---|---|---|
| weight DMA | threads 1..P, one slice each, own DMA engine | engine |
| dequant | any thread, in pieces of `dq_per` tiles | `weight_prep::convert` |
| act DMA + convert | the first `act_threads` threads, row blocks of `act_rows_per_dma` | engine + `act_prep::convert` |
| MAC | submitted by thread 0, runs on the HMX thread | `mac_unit::dot` |
| store | any thread, in units of `st_block` column tiles | `post_proc::store` |

### Buffers

Each stage writes into a small ring so that neighbouring stages overlap:

| Region | Count | Overlap it enables |
|---|---|---|
| raw | `n_raw_bufs` (2..3) | weight DMA runs 1..2 jobs ahead of dequant |
| w | 2 | dequant(j) while MAC(j-1) reads |
| o | 3 | MAC(j), store(j-1) and store(j-2) never wait for each other |
| act | `n_a_bufs` (1..2) | prefill converts the next m-chunk under the current one |
| act_stage | `act_threads` x 2 halves | DMA of the next row block while this one converts |

```
job:      0      1      2      3      4
DMA     [0][1][2]   [3]    [4]    ...
dequant    [0]    [1]    [2]    [3]
HMX              [ 0 ][ 1 ][ 2 ][ 3 ]
store                   [0]   [1]   [2]
```

`build_layout()` turns the plan and the `format` into byte offsets. Every region
is aligned to `tile_align` (2 KB) so HMX tiles stay aligned.

## Planner

`make_plan(shape, limits, format, engine, &plan)` searches `mc`, `nc`,
`n_raw_bufs`, `n_a_bufs`, `act_threads`, `act_rows_per_dma` and `ramp`, ranks
candidates with `estimate_cycles()` and keeps the cheapest plan that fits the
VTCM budget. The host caches the result per (M, K, N, engine) because it runs
on every graph build.

The planner never names Q4_0 or fp16. Everything component specific enters
through `format`:

```cpp
struct format {
    uint32_t tile_k;              // k depth of one act / weight tile
    uint32_t act_in_bytes;        // activation element in DDR
    uint32_t act_bytes;           // activation tile element
    uint32_t w_bytes;             // weight tile element
    uint32_t out_bytes;           // output tile element
    uint32_t raw_tile_bytes;      // one raw weight tile
    uint32_t scales_bytes;        // MAC column scale table
    uint32_t dequant_per_thread;  // throughputs per cycle for the cost model
    uint32_t act_per_thread;
    uint32_t store_per_thread;
    uint32_t macs;
    uint32_t mac_job_overhead;
    uint32_t mac_tile_overhead;
};
```

The host uses the constant `formats::q4_0_f32_f16`. On the device
`composition<pre, post, mac>::fmt` derives the same struct from the component
traits, and `hmx-mm-cpp.cc` checks with `static_assert` that the two are equal.
Change a component trait and the build tells you to update the host constant.

## Schedulers and engines

`execution<scheduler, pre, post, mac>()` instantiates
`scheduler::engine<scheduler, pre, post, mac>`: a scheduler names the engine it
is written for. A scheduler decides job order, buffer slots and when a slot may
be reused. An engine owns the mechanisms: counters, DMA rings, task lists and the
thread loop.

### Dataflow engine (`sched_stream`, default)

Barrier-free. Every thread runs the same loop until the last job is stored:

1. `service_dma`: pop finished descriptors from the thread's ring. A weight
   slice publishes `w_ready`; an act block is converted on the spot.
2. `issue_weight_dma` (threads 1..P): push the thread's slice of the next job
   once its raw slot is free.
3. `service_mac` (thread 0): collect completed MAC jobs, submit the next job when
   dequant(j), act(mi) and the output slot are ready, advance the act frontier.
4. `issue_act_dma` (first `act_threads` threads): claim the next row block of
   the next m-chunk into a free staging half.
5. `run_dequant`: claim one piece, own slice first, then steal.
6. `run_store`: claim one store unit of a completed job.
7. Nothing to do: `pause`.

Dependencies are tracked with `job_counter<ring>`: a 32-bit word holding
`(generation << 24) | count`, one per cache line. The generation is derived from
the job index, so a ring slot reused by job `j + ring` never confuses the old
count with progress. `claim()` is one `fetch_add` in the common case.

Two details matter for performance. The DMA done bit only becomes visible after
`dmpoll()`, which is expensive, so it is called at most every
`dma_poll_interval` cycles per thread. Dequant pieces take no memory barrier:
the core is in order, the slice landed before `w_ready` was published, and a
barrier would drain the posted vector stores of the previous piece.

`sched_stream` provides:

```cpp
uint32_t raw_slot(j), w_slot(j), o_slot(j), act_slot(mi);      // which ring slot a job uses
uint32_t raw_slot_busy_until_dequant(j);   // job whose dequant must finish before raw slot reuse
uint32_t o_slot_busy_until_stored(j);      // job whose store must finish before o slot reuse
uint32_t w_slot_needs_mac_done(j);         // MAC jobs that must be complete before dequant(j)
uint32_t act_slot_needs_mac_done(mi);      // MAC jobs that must be complete before act(mi)
uint32_t act_frontier(mi);                 // m-chunks that may be fetched once MAC of mi started
```

### Fork-join engine (`sched_prefill`, `sched_decode`)

Step `j`: wait for MAC(j-2), build a task list (weight DMA push/pop per slice,
dequant pieces, pending stores, an activation slice), run it on all threads with
`work_queue_run`, submit MAC(j). The scheduler's `step(j, step_plan*)` says which
activation rows to convert in this step and how far the MAC unit must have
progressed. Simpler to read and useful as a reference, but each step pays a
barrier.

## Component contracts

Components are stateless types: traits the engines and the planner read, plus
static functions that run on one thread. They never touch DMA rings, counters or
the MAC queue. The concepts live in `hmx-mm-core.h`.

### `weight_prep`

```cpp
struct weight_prep_x {
    using tile_t = ...;                       // element of the produced tiles (== mac::w_t)
    static constexpr uint32_t tile_k;          // k depth of one tile (== mac::tile_k)
    static constexpr uint32_t raw_tile_bytes;  // one raw tile in DDR / VTCM
    static constexpr uint32_t piece_align;     // convert tiles in groups aligned to this
    static constexpr uint32_t cost_per_thread; // elements per cycle per HVX thread (planner)
    struct source { dma_addr_t tile_addr(uint32_t ct, uint32_t n_kt) const; };
    static void convert(const uint8_t * raw, tile_t * tiles, uint32_t t0, uint32_t t1);
};
```

The engine DMAs raw tiles unchanged into VTCM, `n_kt * raw_tile_bytes`
contiguous bytes per column tile, and calls `convert` for tile ranges
`[t0, t1)` of a chunk. Output tile `t` goes to `tiles + t * tile_dim * tile_k`.
`piece_align` lets a format keep tile groups together (Q4_0 uses 2: raw tiles
are 576 bytes, so only even tiles are 128-byte aligned and odd tiles are built
with `valign` from the pair).

### `act_prep`

```cpp
struct act_prep_x {
    using in_t   = ...;                        // DDR element
    using tile_t = ...;                        // == mac::act_t
    static constexpr uint32_t tile_k;
    static constexpr uint32_t cost_per_thread;
    struct source { dma_addr_t row_addr(uint32_t r) const; uint32_t nb1; };
    static void convert(const in_t * rows, uint32_t k, uint32_t n_rows, uint32_t n_valid,
                        tile_t * tiles, uint32_t row0);
};
```

The engine DMAs `n_rows` dense rows of `k` elements into a staging half and
calls `convert` with the chunk-local first row `row0`. Rows at or past
`n_valid` are padding and must be written as zero. Tile `(rt, kt)` of the chunk
is at `(rt * n_kt + kt) * tile_dim * tile_k`.

### `post_proc`

```cpp
struct post_processor_x {
    using in_t = ...;                          // == mac::out_t
    static constexpr uint32_t cost_per_thread;
    struct sink { ...; uint32_t cols; };       // destination and epilogue inputs
    static void store(const in_t * tiles, uint32_t n_ct, uint32_t rt, uint32_t ct0, uint32_t ct1,
                      uint32_t rows_valid, const sink & dst, uint32_t row, uint32_t col);
};
```

`tiles` are the output tiles of one chunk, tile `(rt, ct)` at
`(rt * n_ct + ct) * tile_dim * tile_dim`. The call covers tile row `rt`, column
tiles `[ct0, ct1)`, destination rows `[row, row + rows_valid)` from column
`col`. Columns at or past `sink.cols` must be dropped. Everything an epilogue
needs (bias pointer, residual, scale) is a field of `sink`; the engines copy the
sink and never look inside beyond `cols`.

### `mac_unit`

```cpp
struct mac_x {
    using act_t = ...; using w_t = ...; using out_t = ...;
    static constexpr uint32_t tile_k, scales_bytes;
    static constexpr uint32_t cost_macs, cost_job_overhead, cost_tile_overhead;
    struct job { out_t * out; const act_t * act; const w_t * w; const void * scales;
                 uint32_t n_rt, n_ct, n_kt; };
    static void init_scales(void * scales);    // fill the column scale table once per op
    static void dot(const job & jb);           // out[rt][ct] = sum_kt act[rt][kt] x w[ct][kt]
};
```

The MAC unit owns the tile layouts: the preps of a MAC family produce the tiles
its `dot` consumes. `dot` runs on the HMX thread through `mac_dispatcher`, up to
`n_o_bufs` jobs in flight.

### Composition

```cpp
using pre = pre_processor<weight_prep_q4_0, act_prep_f32>;
execution<sched_stream, pre, post_processor_f32, hmx_mac_f16>(ctx, args, plan);
```

`composition<pre, post, mac>` asserts that `weight::tile_t == mac::w_t`,
`act::tile_t == mac::act_t`, `post::in_t == mac::out_t` and that all `tile_k`
agree, with a named message for each. `arguments<pre, post>` bundles the shape
with `weight::source`, `act::source` and `post::sink`, so component specific
operand descriptions never pass through the engines.

## Extension guide

Every extension follows the same steps:

1. Add a type that satisfies one concept in `hmx-mm-components.h` (or a new
   header). Add `static_assert(concept<type>)` next to it.
2. Compose it in `hmx-mm-cpp.cc`. If the choice depends on tensor types, switch
   on them there and call a different `execution<>` instantiation.
3. Add a `formats::` constant for the new composition in `hmx-mm-plan.h` and
   pass it from the host in `ggml_hexagon_precompute_hmx_cpp_mm_params()`. The
   `static_assert` in `hmx-mm-cpp.cc` keeps the two in sync.
4. Widen the host eligibility checks (tensor types, strides) and, if the kernel
   needs a new `kernel_type`, add it to `matmul-ops.h` and the dispatch in
   `matmul-ops.c`.
5. Add `test-backend-ops` cases for the new types, including M in {1, 4, 33, 100,
   512} and K that is not a power of two, run the strict spill check, then A/B.

The engines do not change. If an extension seems to need an engine change, it is
usually a missing trait or a scheduler rule; add that instead.

### A new weight format (example: Q8_0)

```cpp
struct weight_prep_q8_0 {
    using tile_t = __fp16;
    static constexpr uint32_t tile_k          = 32;
    static constexpr uint32_t raw_tile_bytes  = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;  // 1088
    static constexpr uint32_t piece_align     = 1;   // 1088 = 8.5 vectors: handle alignment in convert
    static constexpr uint32_t cost_per_thread = 20;  // measure, then update formats::
    struct source {
        dma_addr_t base;
        dma_addr_t tile_addr(uint32_t ct, uint32_t n_kt) const { return base + (dma_addr_t) ct * n_kt * raw_tile_bytes; }
    };
    static void convert(const uint8_t * raw, tile_t * tiles, uint32_t t0, uint32_t t1);
};
```

The raw layout is whatever the repack produced (`ggml-hexagon.cpp` repacks
Q8_0 for the C path already). `convert` must produce the fp16 tile layout that
`hmx_mac_f16::dot` reads: vector `i` of a tile holds the k pair `(2i, 2i+1)` for
32 rows, interleaved. Then: `formats::q8_0_f32_f16`, a `src0->type` switch in
`hmx-mm-cpp.cc`, and the host check `wtype == GGML_TYPE_Q4_0` becomes a set.

### A new activation type (example: F16 input)

```cpp
struct act_prep_f16 {
    using in_t   = __fp16;
    using tile_t = __fp16;
    static constexpr uint32_t tile_k = 32, cost_per_thread = 4;
    struct source { dma_addr_t base; uint32_t nb1; dma_addr_t row_addr(uint32_t r) const { return base + (dma_addr_t) r * nb1; } };
    static void convert(const in_t * rows, uint32_t k, uint32_t n_rows, uint32_t n_valid, tile_t * tiles, uint32_t row0);
};
```

Only the interleave remains (no f32 -> f16 conversion). The planner picks up the
smaller DDR element through `format::act_in_bytes` and sizes the staging area
accordingly. On the host, allow `src1->type == GGML_TYPE_F16` and
`src1->nb[0] == 2`.

### A new epilogue (example: fused bias / residual add)

```cpp
struct post_processor_f32_add {
    using in_t  = __fp16;
    using out_t = float;
    static constexpr uint32_t cost_per_thread = 1;
    struct sink {
        out_t *       base;
        uint32_t      stride;
        uint32_t      cols;
        const float * add;        // src2 rows, same stride as the output
        uint32_t      add_stride;
    };
    static void store(const in_t * tiles, uint32_t n_ct, uint32_t rt, uint32_t ct0, uint32_t ct1,
                      uint32_t rows_valid, const sink & dst, uint32_t row, uint32_t col);
};
```

`store` converts fp16 to f32 exactly like `post_processor_f32` and adds the
matching `add` row before the store. The sink is filled in `hmx-mm-cpp.cc` from
`octx->src[2]`. On the host, stop returning `false` for the `MUL_MAT_ADD` fusion
when the node is the C++ kernel and emit this composition instead.

### A new MAC data type (example: HMX int8)

```cpp
struct hmx_mac_i8 {
    using act_t = int8_t;
    using w_t   = int8_t;
    using out_t = int32_t;                     // or whatever HMX stores after its own scaling
    static constexpr uint32_t tile_k = 64;     // k depth of an int8 HMX tile
    static constexpr uint32_t scales_bytes = 2 * VLEN;
    static constexpr uint32_t cost_macs = 5000, cost_job_overhead = 6000, cost_tile_overhead = 64;
    struct job { out_t * out; const act_t * act; const w_t * w; const void * scales; uint32_t n_rt, n_ct, n_kt; };
    static void init_scales(void * scales);
    static void dot(const job & jb);           // HMX_* int8 macros, same loop shape as hmx_mac_f16::dot
};
```

A MAC unit defines the family: it needs a `weight_prep` with
`tile_t = int8_t` and `tile_k = 64` (Q4_0 -> int8 tiles with per-tile scales
pushed into the HMX column scales), an `act_prep` that quantizes rows to int8
(and reports per-row scales through its tiles or the sink), and a `post_proc`
with `in_t = int32_t` that applies the scales. The engines size every buffer
from `act_t`, `w_t`, `out_t` and `tile_k`; `chunk_grid::n_kt` becomes
`K / 64`. A mismatched pair fails at compile time, for example:

```
error: static assertion failed: act prep must produce the MAC unit's activation tiles
```

### A new scheduler (example: NX, several weights sharing one activation)

For the dataflow engine, a scheduler is the job enumeration plus the reuse rules
listed above. An NX scheduler enumerates jobs as (m-chunk, weight index,
n-chunk), keeps `act_slot_needs_mac_done()` tied to the m-chunk and lets
`chunk_grid` carry a weight index. Give it

```cpp
template <class S, class pre, class post, class mac>
using engine = engine_dataflow<S, pre, post, mac>;
```

and a constructor `(const shape &, const plan &, uint32_t tile_k)`. The weight
`source` then needs `tile_addr(ct, n_kt)` per weight, which is a `source` with
an array of bases. Plan a new `sched_kind` only if the fork-join engine should
get the same policy; the planner's `kind` is only read by that engine.

## Measuring

Per-op numbers come from `test-backend-ops perf` with
`-p "type_a=q4_0,type_b=f32,m=N,n=M,k=K"`. Compare the C path with
`GGML_HEXAGON_OPFUSION=0`, otherwise the harness fuses identical ops into
`MUL_MAT_NX`.

Device state moves more than most code changes. In one session the unchanged
C path drifted by 22% between two runs ten minutes apart, and the first run
after an idle period is 15-25% faster than the steady state. For an A/B:

- interleave the two builds in one session (push A, run all shapes, push B,
  run all shapes, repeat 3 times) and compare medians,
- keep two shapes of the unchanged C path in every slot as a control,
- discard the first slot after a pause, or start both variants from it,
- check the NPU clock in the op profile (cycles per microsecond, about 2120 on
  SM8850 when idle).

Run `scripts/snapdragon/ggml-hexagon-inspect.py --spills --strict` on the HTP
library after any change to a component: the HVX kernels must keep zero in-loop
vector spills.

## Known limits and open work

- Q4_0 x F32 only. Other weight types, F16 activations and `matmul_id` fall back to the C kernels.
- No NX and no fused ADD: the host disables both fusions for C++ nodes, which costs about 25% in-model decode throughput against the C path even though per-op times are at parity or better for M >= 4.
- M <= 2 on HMX is bound by the Q4 -> fp16 expansion traffic in VTCM; the HVX int8 path stays faster there. `GGML_HEXAGON_MM_CPP=2` keeps small M on the C path.
- The cost model constants in `formats::` were measured on SM8850 (Hexagon v81) and are not calibrated for other targets.
