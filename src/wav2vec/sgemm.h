#ifndef SGEMM_H
#define SGEMM_H

#include <stdbool.h>
#include <stdint.h>

// C (M x N) = alpha * A * op(B)   (beta = 0, overwrites C)
//
//   A: row-major M x K, leading dimension lda (row stride; >= K)
//   B: row-major: if !trans_b, K x N (ldb >= N); if trans_b, N x K (ldb >= K)
//   C: row-major M x N, leading dimension ldc (>= N)
//
// No transpose on A. If you need transA, pre-pack / transpose at the call site.
// Thread-safe. Uses pthreads for row-block parallelism when M is large.
void sgemm_rm(bool trans_b,
              int M, int N, int K,
              float alpha,
              const float *A, int lda,
              const float *B, int ldb,
              float *C, int ldc);

// Configure intra-op threading for sgemm_rm calls made from *this* thread.
// 0 or 1 (default) = single-threaded. >1 spins up a per-caller-thread pool on
// first use; that pool is destroyed when the caller thread exits.
//
// Thread-safety: sgemm_rm is safe to call concurrently from multiple threads.
// Each caller gets its own pool. No shared mutable state across callers.
void sgemm_set_threads(int n);

// C (M x N) = alpha * A * B^T   where B is int8 (N x K, row-major, ldb) and
// `scale` has length N (per-output-channel). A and C are fp32.
//
// Equivalent to sgemm_rm(trans_b=true, ...) after dequantizing B via scale,
// but dequant happens inside pack_B so the weight stays int8 in memory.
// Thread-safe (same pool model as sgemm_rm).
void sgemm_s8rm(int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const int8_t *B, const float *scale, int ldb,
                float *C, int ldc);

// Pre-pack an (N x K) int8 weight matrix into the internal block layout so
// run-time GEMM can skip pack_B entirely. Returns a handle owned by the
// caller; free with sgemm_s8_prepack_free(). `scale` is not stored in the
// handle — the caller still passes it to sgemm_s8rm_prepacked for dequant.
typedef struct sgemm_s8_prepack sgemm_s8_prepack_t;
sgemm_s8_prepack_t *sgemm_s8_prepack(int N, int K, const int8_t *B, int ldb);
void                sgemm_s8_prepack_free(sgemm_s8_prepack_t *p);

// Same as sgemm_s8rm but takes a pre-packed B handle.
void sgemm_s8rm_prepacked(int M, int N, int K,
                          float alpha,
                          const float *A, int lda,
                          const sgemm_s8_prepack_t *Bp, const float *scale,
                          float *C, int ldc);

#endif
