#ifndef HMX_MM_CPP_H
#define HMX_MM_CPP_H

// C entry point of the C++ HMX matmul pipeline (hmx-mm-pipeline.h), selected by
// kernel_type == HTP_MM_KERNEL_HMX_Q4_0_CPP.

#ifdef __cplusplus
extern "C" {
#endif

struct htp_ops_context;

int op_matmul_hmx_cpp(struct htp_ops_context * octx);

#ifdef __cplusplus
}
#endif

#endif  // HMX_MM_CPP_H
