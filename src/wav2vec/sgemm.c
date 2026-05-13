// Hand-rolled FP32 single-precision general matrix multiply, zero external deps.
//
// Computes  C = alpha * A * op(B)  with A row-major (M x K, lda) and
// B row-major ((trans_b ? N x K : K x N), ldb). Beta is implicitly 0.
//
// Strategy: BLIS/Goto-style blocking.
//   - Outer L3-sized blocks (KC, NC).
//   - Packed A (MR x KC, column-major within KC) and B (KC x NR, row-major) panels.
//   - 4x8 register micro-kernel using NEON intrinsics (ARM64) or a tight scalar
//     loop (everything else — `-O3 -ffast-math` auto-vectorizes for x86 AVX2).
//   - Row-block parallelism across a fixed pthread pool.

#include "sgemm.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

// AVX2 + FMA on x86 (Haswell 2013+). Both macros are set when the compiler
// is targeting them (e.g. -march=x86-64-v3 or -mavx2 -mfma).
#if !HAS_NEON && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

#include <math.h>  // fabsf, lrintf

// SDOT (int8 4-wide dot-product) intrinsic. Enabled by -march=...+dotprod.
// Apple Silicon M1+ supports this unconditionally.
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
#define HAS_SDOT 1
#else
#define HAS_SDOT 0
#endif

// SMMLA (int8 2×2 matrix MAC). Enabled by -march=...+i8mm. Apple Silicon M2+.
// When available it supersedes the SDOT path: ~2× the int8 MACs per instruction.
#if HAS_NEON && defined(__ARM_FEATURE_MATMUL_INT8)
#define HAS_SMMLA 1
#else
#define HAS_SMMLA 0
#endif

// AVX-VNNI: `_mm256_dpbusd_epi32` does 32 int8 MACs per instruction. Ice Lake+
// server (AVX-512 VNNI), Alder Lake+ client (AVX-VNNI), Zen 4+. When available
// it supersedes the plain-AVX2 int8 path (~2× the MACs per instruction).
#if HAS_AVX2 && (defined(__AVXVNNI__) || defined(__AVX512VNNI__))
#define HAS_VNNI_S8 1
#else
#define HAS_VNNI_S8 0
#endif

// AVX2 int8 path: widen int8→int16 via vpmovsxbw then vpmaddwd accumulates
// pair-sums to int32. No VNNI required. Available whenever HAS_AVX2 is set.
#define HAS_AVX2_S8 HAS_AVX2

// K alignment required by the active int8 compute path:
//   SMMLA: reads 8 k elements per pair → kc must be a multiple of 8.
//   SDOT:  reads 4 k elements per chunk → kc must be a multiple of 4.
//   AVX2:  processes 16 k elements per iteration; K must be multiple of 16.
#if HAS_SMMLA
#define S8_K_ALIGN 8
#elif HAS_VNNI_S8
#define S8_K_ALIGN 32
#elif HAS_AVX2_S8
#define S8_K_ALIGN 16
#else
#define S8_K_ALIGN 4
#endif

// If neither SDOT nor SMMLA is available (x86, old ARM), the int8 pack
// functions and scalar edge kernel are still compiled. macro_kernel_s8 falls
// back to the scalar edge kernel for every tile — correct but slow. Users on
// those targets should prefer weights-fp32 for real throughput.

// Tunables.
#define MR 4
#define NR 8
#define KC 256    // K-block (rows of packed B, cols of packed A)
#define NC 1024    // N-block (cols of packed B)
#define MC 192    // M-block (rows of packed A)

#define PACK_ALIGN 64  // cacheline-friendly alignment for packed panels

// ---------------------------------------------------------------------------
// Micro-kernel: compute  C[MR x NR] += packA[KC x MR] * packB[KC x NR]
// then scale by alpha and store (beta = 0 implicit).
//
// packA layout: K contiguous blocks of MR floats   (packA[k*MR + mi] = A[mi, k])
// packB layout: K contiguous blocks of NR floats   (packB[k*NR + nj] = op(B)[k, nj])
// ---------------------------------------------------------------------------
// Micro-kernel: compute  C[MR x NR] = alpha * (packA * packB) + beta * C
//   beta = 0.0f: overwrite C
//   beta = 1.0f: accumulate into C (used when the K dimension is split across
//                multiple KC-panels, so we re-enter with the existing partial sum)
#if HAS_NEON
static inline void micro_kernel_4x8(int K, float alpha, float beta,
                                    const float *restrict packA,
                                    const float *restrict packB,
                                    float *restrict C, int ldc) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);

    for (int k = 0; k < K; k++) {
        float32x4_t b0 = vld1q_f32(packB + k * NR);
        float32x4_t b1 = vld1q_f32(packB + k * NR + 4);
        float a0 = packA[k * MR + 0];
        float a1 = packA[k * MR + 1];
        float a2 = packA[k * MR + 2];
        float a3 = packA[k * MR + 3];
        c00 = vfmaq_n_f32(c00, b0, a0);  c01 = vfmaq_n_f32(c01, b1, a0);
        c10 = vfmaq_n_f32(c10, b0, a1);  c11 = vfmaq_n_f32(c11, b1, a1);
        c20 = vfmaq_n_f32(c20, b0, a2);  c21 = vfmaq_n_f32(c21, b1, a2);
        c30 = vfmaq_n_f32(c30, b0, a3);  c31 = vfmaq_n_f32(c31, b1, a3);
    }
    float32x4_t va = vdupq_n_f32(alpha);
    c00 = vmulq_f32(c00, va); c01 = vmulq_f32(c01, va);
    c10 = vmulq_f32(c10, va); c11 = vmulq_f32(c11, va);
    c20 = vmulq_f32(c20, va); c21 = vmulq_f32(c21, va);
    c30 = vmulq_f32(c30, va); c31 = vmulq_f32(c31, va);
    if (beta != 0.0f) {
        c00 = vaddq_f32(c00, vld1q_f32(C + 0 * ldc));
        c01 = vaddq_f32(c01, vld1q_f32(C + 0 * ldc + 4));
        c10 = vaddq_f32(c10, vld1q_f32(C + 1 * ldc));
        c11 = vaddq_f32(c11, vld1q_f32(C + 1 * ldc + 4));
        c20 = vaddq_f32(c20, vld1q_f32(C + 2 * ldc));
        c21 = vaddq_f32(c21, vld1q_f32(C + 2 * ldc + 4));
        c30 = vaddq_f32(c30, vld1q_f32(C + 3 * ldc));
        c31 = vaddq_f32(c31, vld1q_f32(C + 3 * ldc + 4));
    }
    vst1q_f32(C + 0 * ldc,     c00);
    vst1q_f32(C + 0 * ldc + 4, c01);
    vst1q_f32(C + 1 * ldc,     c10);
    vst1q_f32(C + 1 * ldc + 4, c11);
    vst1q_f32(C + 2 * ldc,     c20);
    vst1q_f32(C + 2 * ldc + 4, c21);
    vst1q_f32(C + 3 * ldc,     c30);
    vst1q_f32(C + 3 * ldc + 4, c31);
}
#elif HAS_AVX2
// AVX2+FMA 4×8 micro-kernel: one 256-bit load per k-iter covers all 8 cols,
// broadcast each of MR=4 A values and issue 4 FMAs. Four persistent int32x8
// accumulators (one per row) fit easily in AVX2's 16 YMM registers.
static inline void micro_kernel_4x8(int K, float alpha, float beta,
                                    const float *restrict packA,
                                    const float *restrict packB,
                                    float *restrict C, int ldc) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    for (int k = 0; k < K; k++) {
        __m256 b  = _mm256_loadu_ps(packB + k * NR);
        __m256 a0 = _mm256_broadcast_ss(packA + k * MR + 0);
        __m256 a1 = _mm256_broadcast_ss(packA + k * MR + 1);
        __m256 a2 = _mm256_broadcast_ss(packA + k * MR + 2);
        __m256 a3 = _mm256_broadcast_ss(packA + k * MR + 3);
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
    }
    __m256 va = _mm256_set1_ps(alpha);
    c0 = _mm256_mul_ps(c0, va); c1 = _mm256_mul_ps(c1, va);
    c2 = _mm256_mul_ps(c2, va); c3 = _mm256_mul_ps(c3, va);
    if (beta != 0.0f) {
        c0 = _mm256_add_ps(c0, _mm256_loadu_ps(C + 0 * ldc));
        c1 = _mm256_add_ps(c1, _mm256_loadu_ps(C + 1 * ldc));
        c2 = _mm256_add_ps(c2, _mm256_loadu_ps(C + 2 * ldc));
        c3 = _mm256_add_ps(c3, _mm256_loadu_ps(C + 3 * ldc));
    }
    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
}
#else
static inline void micro_kernel_4x8(int K, float alpha, float beta,
                                    const float *restrict packA,
                                    const float *restrict packB,
                                    float *restrict C, int ldc) {
    float c[MR][NR] = {{0}};
    for (int k = 0; k < K; k++) {
        const float *bk = packB + k * NR;
        const float *ak = packA + k * MR;
        for (int mi = 0; mi < MR; mi++) {
            float a = ak[mi];
            for (int nj = 0; nj < NR; nj++) c[mi][nj] += a * bk[nj];
        }
    }
    for (int mi = 0; mi < MR; mi++) {
        for (int nj = 0; nj < NR; nj++) {
            float v = alpha * c[mi][nj];
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}
#endif

static inline void micro_kernel_edge(int K, float alpha, float beta,
                                     const float *packA, const float *packB,
                                     float *C, int ldc,
                                     int mr_tail, int nr_tail) {
    float c[MR][NR] = {{0}};
    for (int k = 0; k < K; k++) {
        const float *bk = packB + k * NR;
        const float *ak = packA + k * MR;
        for (int mi = 0; mi < mr_tail; mi++) {
            float a = ak[mi];
            for (int nj = 0; nj < nr_tail; nj++) c[mi][nj] += a * bk[nj];
        }
    }
    for (int mi = 0; mi < mr_tail; mi++) {
        for (int nj = 0; nj < nr_tail; nj++) {
            float v = alpha * c[mi][nj];
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

// pack A panel: (mc x kc) block starting at A[i_start, k_start] -> packed
// Layout: ceil(mc/MR) MR-row tiles, each tile stored column-major within kc.
static void pack_A(const float *A, int lda,
                   int i_start, int k_start, int mc, int kc,
                   float *pack) {
    int m_full = (mc / MR) * MR;
    // Full MR-row tiles
    for (int mi = 0; mi < m_full; mi += MR) {
        const float *src = A + (i_start + mi) * lda + k_start;
        float *dst = pack + mi * kc;
        for (int k = 0; k < kc; k++) {
            dst[k * MR + 0] = src[0 * lda + k];
            dst[k * MR + 1] = src[1 * lda + k];
            dst[k * MR + 2] = src[2 * lda + k];
            dst[k * MR + 3] = src[3 * lda + k];
        }
    }
    // Tail rows (< MR); zero-pad the missing rows of the tile
    if (m_full < mc) {
        int tail = mc - m_full;
        const float *src = A + (i_start + m_full) * lda + k_start;
        float *dst = pack + m_full * kc;
        for (int k = 0; k < kc; k++) {
            int mi;
            for (mi = 0; mi < tail;  mi++) dst[k * MR + mi] = src[mi * lda + k];
            for (;      mi < MR;    mi++) dst[k * MR + mi] = 0.0f;
        }
    }
}

// pack B panel: (kc x nc) block.
//   !trans_b: B is K x N row-major; we read B[k_start..k_start+kc, j_start..j_start+nc].
//    trans_b: B is N x K row-major; we read B[j_start..j_start+nc, k_start..k_start+kc].
// Layout: ceil(nc/NR) NR-col tiles, each tile stored row-major within kc (kc rows of NR).
static void pack_B(const float *B, int ldb,
                   int k_start, int j_start, int kc, int nc,
                   bool trans_b, float *pack) {
    int n_full = (nc / NR) * NR;
    for (int nj = 0; nj < n_full; nj += NR) {
        float *dst = pack + nj * kc;
        if (!trans_b) {
            for (int k = 0; k < kc; k++) {
                const float *src = B + (k_start + k) * ldb + (j_start + nj);
                for (int j = 0; j < NR; j++) dst[k * NR + j] = src[j];
            }
        } else {
            for (int k = 0; k < kc; k++) {
                for (int j = 0; j < NR; j++) {
                    dst[k * NR + j] = B[(j_start + nj + j) * ldb + (k_start + k)];
                }
            }
        }
    }
    if (n_full < nc) {
        int tail = nc - n_full;
        float *dst = pack + n_full * kc;
        for (int k = 0; k < kc; k++) {
            int j;
            if (!trans_b) {
                const float *src = B + (k_start + k) * ldb + (j_start + n_full);
                for (j = 0; j < tail; j++) dst[k * NR + j] = src[j];
            } else {
                for (j = 0; j < tail; j++)
                    dst[k * NR + j] = B[(j_start + n_full + j) * ldb + (k_start + k)];
            }
            for (; j < NR; j++) dst[k * NR + j] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Inner macro-kernel: given packed A and packed B, update C[0..mc, 0..nc].
// C is the top-left of the macro-block within the user's output.
// ---------------------------------------------------------------------------
static void macro_kernel(int mc, int nc, int kc, float alpha, float beta,
                         const float *packA, const float *packB,
                         float *C, int ldc) {
    for (int mi = 0; mi < mc; mi += MR) {
        int mr_tail = (mc - mi < MR) ? (mc - mi) : MR;
        for (int nj = 0; nj < nc; nj += NR) {
            int nr_tail = (nc - nj < NR) ? (nc - nj) : NR;
            const float *pA = packA + mi * kc;
            const float *pB = packB + nj * kc;
            float *pC = C + mi * ldc + nj;
            if (mr_tail == MR && nr_tail == NR) {
                micro_kernel_4x8(kc, alpha, beta, pA, pB, pC, ldc);
            } else {
                micro_kernel_edge(kc, alpha, beta, pA, pB, pC, ldc, mr_tail, nr_tail);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-caller-thread pool for row-block parallelism.
//
// The pool (including its worker threads, mutex, condvars, and job slots) is
// stored in thread-local storage keyed on the *calling* thread. So two app
// threads that both call sgemm_rm concurrently each spawn their own private
// pool and do not share state. This makes sgemm_rm safe to call from many
// concurrent workers (e.g. an inference server with one worker per request).
// ---------------------------------------------------------------------------
#define MAX_THREADS 16

typedef struct {
    int m_start, m_end;   // row-slice of the user's C assigned to this job
    int n_start, n_end;   // col-slice of the user's C (defaults to [0, N))
    int N, K;
    float alpha;
    const float *A; int lda;
    const float *B;           // fp32 B (NULL when using int8)
    const int8_t *B_s8;       // int8 B (NULL when fp32 or using pre-packed)
    const float *B_scale;     // per-output-row scale (length N), only for int8
    int ldb;
    // Pre-packed B: if non-NULL, run_job skips pack_B_s8_repack and uses these
    // blocks directly. Block [kb * num_jb + jb] covers the (KC × NC) tile at
    // (k_outer = kb*KC, j_outer = jb*NC).
    int8_t *const *B_blocks;
    int num_kb, num_jb;
    bool trans_b;
    float *C; int ldc;
} job_t;

typedef struct pool {
    pthread_t threads[MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
    job_t jobs[MAX_THREADS];
    int jobs_pending;
    int active_threads;
    int size;
    bool shutdown;
} pool_t;

// ---------------------------------------------------------------------------
// W8A8 (true int8 compute) path.
// ---------------------------------------------------------------------------
// Overview:
//   - A (fp32 activations) is quantized dynamically per-row at pack time,
//     producing int8 data + one float32 scale per row.
//   - B (int8 weights) is repacked into block layout; its per-row scale is
//     copied alongside.
//   - Micro-kernel: vdotq_laneq_s32 does 4 int8 MAC per lane per cycle.
//   - Epilogue: C[i,j] += alpha * a_scale[i] * b_scale[j] * int32_acc[i,j].
//
// Shape constraints for this path:
//   MR = 4, NR = 8, K (and kc on every tile) must be a multiple of 4.
// ---------------------------------------------------------------------------

// The rest of this block (int8 pack/kernels/macro) is only compiled when an
// int8 SIMD instruction is available. On x86/scalar hosts, `is_int8` in run_job
// takes the pack_B_s8_dequant fallback below and uses the fp32 micro-kernel.
#if HAS_SMMLA || HAS_SDOT

static inline int8_t saturate_i8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Pack A dynamic-quant: compute per-row max-abs / 127 scale, then quantize +
// pack into layout-specific tile blocks. Scale is written to scaleA_out[mi]
// for callers' epilogue dequant. Layout below differs per active kernel.
#if HAS_SMMLA
// ---- SMMLA layout ----
// Per MR=4 row tile: (kc/8) k-chunks × 2 row-pairs × 16 int8 per pair.
// Each pair lays out as [row_a_k0..7, row_b_k0..7] = 16 contiguous int8,
// matching SMMLA's "2x8 matrix" operand interpretation.
static void pack_A_s8_quant(const float *A, int lda,
                            int i_start, int k_start, int mc, int kc,
                            int8_t *pack, float *scaleA_out) {
    int m_full = (mc / MR) * MR;
    float inv[MC + MR];
    for (int mi = 0; mi < mc; mi++) {
        const float *row = A + (i_start + mi) * lda + k_start;
        float mx = 0.0f;
        for (int k = 0; k < kc; k++) { float v = fabsf(row[k]); if (v > mx) mx = v; }
        float s = mx / 127.0f;
        if (s == 0.0f) { scaleA_out[mi] = 1.0f; inv[mi] = 0.0f; }
        else           { scaleA_out[mi] = s;    inv[mi] = 1.0f / s; }
    }
    for (int mi = mc; mi < m_full + MR && mi < MC + MR; mi++) {
        scaleA_out[mi] = 1.0f; inv[mi] = 0.0f;
    }

    const int chunk_stride = MR * 8;        // 32 int8 per k-chunk
    const int pair_stride  = 16;            // per row-pair (2 rows × 8 k)
    for (int mi = 0; mi < m_full; mi += MR) {
        int8_t *dst = pack + mi * kc;
        const float *r[MR];
        float iv[MR];
        for (int t = 0; t < MR; t++) {
            r[t]  = A + (i_start + mi + t) * lda + k_start;
            iv[t] = inv[mi + t];
        }
        for (int k8 = 0; k8 < kc; k8 += 8) {
            int8_t *base = dst + (k8 >> 3) * chunk_stride;
            for (int p = 0; p < 2; p++) {   // row-pair index (0,1)
                int8_t *pair = base + p * pair_stride;
                for (int ri = 0; ri < 2; ri++) {
                    int row = p * 2 + ri;
                    int8_t *out = pair + ri * 8;
                    float ivr = iv[row];
                    const float *src = r[row] + k8;
                    for (int l = 0; l < 8; l++) out[l] = saturate_i8(lrintf(src[l] * ivr));
                }
            }
        }
    }
    if (m_full < mc) {
        int tail = mc - m_full;
        int8_t *dst = pack + m_full * kc;
        for (int k8 = 0; k8 < kc; k8 += 8) {
            int8_t *base = dst + (k8 >> 3) * chunk_stride;
            for (int p = 0; p < 2; p++) {
                int8_t *pair = base + p * pair_stride;
                for (int ri = 0; ri < 2; ri++) {
                    int row = p * 2 + ri;
                    int8_t *out = pair + ri * 8;
                    if (row < tail) {
                        const float *src = A + (i_start + m_full + row) * lda + k_start + k8;
                        float ivr = inv[m_full + row];
                        for (int l = 0; l < 8; l++) out[l] = saturate_i8(lrintf(src[l] * ivr));
                    } else {
                        for (int l = 0; l < 8; l++) out[l] = 0;
                    }
                }
            }
        }
    }
}

// Pack B (int8 weights) into SMMLA layout.
// Per NR=8 col tile: (kc/8) k-chunks × 4 col-pairs × 16 int8 per pair.
// Each pair lays out as [col_a_k0..7, col_b_k0..7] = 16 contiguous int8.
static void pack_B_s8_repack(const int8_t *B, int ldb, const float *scale,
                             int k_start, int j_start, int kc, int nc,
                             int8_t *pack, float *scaleB_out) {
    int n_full = (nc / NR) * NR;
    const int chunk_stride = NR * 8;  // 64 int8 per k-chunk
    const int pair_stride  = 16;
    for (int nj = 0; nj < n_full; nj += NR) {
        int8_t *dst = pack + nj * kc;
        for (int t = 0; t < NR; t++) scaleB_out[nj + t] = scale[j_start + nj + t];
        for (int k8 = 0; k8 < kc; k8 += 8) {
            int8_t *base = dst + (k8 >> 3) * chunk_stride;
            for (int p = 0; p < 4; p++) {     // col-pair index (0..3)
                int8_t *pair = base + p * pair_stride;
                for (int ci = 0; ci < 2; ci++) {
                    int col = p * 2 + ci;
                    const int8_t *src = B + (j_start + nj + col) * ldb + (k_start + k8);
                    int8_t *out = pair + ci * 8;
                    for (int l = 0; l < 8; l++) out[l] = src[l];
                }
            }
        }
    }
    if (n_full < nc) {
        int tail = nc - n_full;
        int8_t *dst = pack + n_full * kc;
        for (int t = 0; t < tail; t++) scaleB_out[n_full + t] = scale[j_start + n_full + t];
        for (int t = tail; t < NR; t++) scaleB_out[n_full + t] = 1.0f;
        for (int k8 = 0; k8 < kc; k8 += 8) {
            int8_t *base = dst + (k8 >> 3) * chunk_stride;
            for (int p = 0; p < 4; p++) {
                int8_t *pair = base + p * pair_stride;
                for (int ci = 0; ci < 2; ci++) {
                    int col = p * 2 + ci;
                    int8_t *out = pair + ci * 8;
                    if (col < tail) {
                        const int8_t *src = B + (j_start + n_full + col) * ldb + (k_start + k8);
                        for (int l = 0; l < 8; l++) out[l] = src[l];
                    } else {
                        for (int l = 0; l < 8; l++) out[l] = 0;
                    }
                }
            }
        }
    }
}

// SMMLA-based 4×8 micro-kernel. K must be a multiple of 8.
// Per k-chunk: 2 A pairs × 4 B pairs × 1 SMMLA = 8 SMMLA instructions, each
// covering 2x2x8 = 32 int8 MACs (vs. SDOT's 4 MACs per vdotq instruction).
static inline void micro_kernel_s8_4x8(int K, float alpha, float beta,
                                       const int8_t *packA, const float *a_scale,
                                       const int8_t *packB, const float *b_scale,
                                       float *C, int ldc) {
    int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0);
    int32x4_t acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
    int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0);
    int32x4_t acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);

    int chunks = K >> 3;
    for (int i = 0; i < chunks; i++) {
        int8x16_t a0 = vld1q_s8(packA);        // rows 0-1 × k0..7
        int8x16_t a1 = vld1q_s8(packA + 16);   // rows 2-3 × k0..7
        int8x16_t b0 = vld1q_s8(packB);        // cols 0-1 × k0..7
        int8x16_t b1 = vld1q_s8(packB + 16);   // cols 2-3 × k0..7
        int8x16_t b2 = vld1q_s8(packB + 32);   // cols 4-5 × k0..7
        int8x16_t b3 = vld1q_s8(packB + 48);   // cols 6-7 × k0..7
        // Each SMMLA accumulates a 2×2 int32 sub-tile (row-major: [r0c0 r0c1 r1c0 r1c1]).
        acc00 = vmmlaq_s32(acc00, a0, b0);
        acc01 = vmmlaq_s32(acc01, a0, b1);
        acc02 = vmmlaq_s32(acc02, a0, b2);
        acc03 = vmmlaq_s32(acc03, a0, b3);
        acc10 = vmmlaq_s32(acc10, a1, b0);
        acc11 = vmmlaq_s32(acc11, a1, b1);
        acc12 = vmmlaq_s32(acc12, a1, b2);
        acc13 = vmmlaq_s32(acc13, a1, b3);
        packA += 32;
        packB += 64;
    }

    // Epilogue: dequantize each 2×2 sub-tile and scatter into C.
    // SMMLA result lane order for a 2×2 tile: 0=[r0,c0] 1=[r0,c1] 2=[r1,c0] 3=[r1,c1].
    // So we can store the low half to row r0 cols [c0,c1] and the high half to row r1.
    float aa[MR];
    for (int i = 0; i < MR; i++) aa[i] = alpha * a_scale[i];
    // Row-factor vectors per pair, aligned with SMMLA lanes [r0,r0,r1,r1].
    float32x4_t a_pair0 = vcombine_f32(vdup_n_f32(aa[0]), vdup_n_f32(aa[1]));
    float32x4_t a_pair1 = vcombine_f32(vdup_n_f32(aa[2]), vdup_n_f32(aa[3]));
    int32x4_t acc[2][4] = { {acc00, acc01, acc02, acc03},
                            {acc10, acc11, acc12, acc13} };

    for (int bp = 0; bp < 4; bp++) {
        int c0 = bp * 2;
        // Col scales {bs0, bs1, bs0, bs1} — aligned with SMMLA's [_,_0,_,_1] lanes.
        float32x2_t bs_pair = vld1_f32(b_scale + c0);
        float32x4_t b_pair  = vcombine_f32(bs_pair, bs_pair);

        float32x4_t cf0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc[0][bp]), a_pair0), b_pair);
        float32x4_t cf1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc[1][bp]), a_pair1), b_pair);

        if (beta == 0.0f) {
            vst1_f32(C + 0 * ldc + c0, vget_low_f32 (cf0));
            vst1_f32(C + 1 * ldc + c0, vget_high_f32(cf0));
            vst1_f32(C + 2 * ldc + c0, vget_low_f32 (cf1));
            vst1_f32(C + 3 * ldc + c0, vget_high_f32(cf1));
        } else {
            vst1_f32(C + 0 * ldc + c0, vadd_f32(vld1_f32(C + 0 * ldc + c0), vget_low_f32 (cf0)));
            vst1_f32(C + 1 * ldc + c0, vadd_f32(vld1_f32(C + 1 * ldc + c0), vget_high_f32(cf0)));
            vst1_f32(C + 2 * ldc + c0, vadd_f32(vld1_f32(C + 2 * ldc + c0), vget_low_f32 (cf1)));
            vst1_f32(C + 3 * ldc + c0, vadd_f32(vld1_f32(C + 3 * ldc + c0), vget_high_f32(cf1)));
        }
    }
}

// Edge (scalar): walks the SMMLA packed layout at pair granularity, handles
// arbitrary mr_tail ∈ [1..MR], nr_tail ∈ [1..NR]. K must be a multiple of 8.
static inline void micro_kernel_s8_edge(int K, float alpha, float beta,
                                        const int8_t *packA, const float *a_scale,
                                        const int8_t *packB, const float *b_scale,
                                        float *C, int ldc,
                                        int mr_tail, int nr_tail) {
    int32_t acc[MR][NR] = {{0}};
    int chunks = K >> 3;
    const int a_chunk_stride = MR * 8;
    const int b_chunk_stride = NR * 8;
    for (int i = 0; i < chunks; i++) {
        const int8_t *a_chunk = packA + i * a_chunk_stride;
        const int8_t *b_chunk = packB + i * b_chunk_stride;
        for (int mi = 0; mi < mr_tail; mi++) {
            const int8_t *a_row = a_chunk + (mi >> 1) * 16 + (mi & 1) * 8;
            for (int nj = 0; nj < nr_tail; nj++) {
                const int8_t *b_col = b_chunk + (nj >> 1) * 16 + (nj & 1) * 8;
                int32_t s = 0;
                for (int l = 0; l < 8; l++) s += (int32_t)a_row[l] * (int32_t)b_col[l];
                acc[mi][nj] += s;
            }
        }
    }
    for (int mi = 0; mi < mr_tail; mi++) {
        for (int nj = 0; nj < nr_tail; nj++) {
            float v = alpha * a_scale[mi] * b_scale[nj] * (float)acc[mi][nj];
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}

#else  // !HAS_SMMLA → SDOT layout (MR=4, NR=8, K%4==0)

// Packed int8 A layout, per MR-row tile: (kc/4) chunks × (MR * 4 int8) = kc*MR int8,
// i.e. one int8x16 load per k-chunk.
static void pack_A_s8_quant(const float *A, int lda,
                            int i_start, int k_start, int mc, int kc,
                            int8_t *pack, float *scaleA_out) {
    int m_full = (mc / MR) * MR;
    float inv[MC + MR];
    for (int mi = 0; mi < mc; mi++) {
        const float *row = A + (i_start + mi) * lda + k_start;
        float mx = 0.0f;
        for (int k = 0; k < kc; k++) { float v = fabsf(row[k]); if (v > mx) mx = v; }
        float s = mx / 127.0f;
        if (s == 0.0f) { scaleA_out[mi] = 1.0f; inv[mi] = 0.0f; }
        else           { scaleA_out[mi] = s;    inv[mi] = 1.0f / s; }
    }
    for (int mi = mc; mi < m_full + MR && mi < MC + MR; mi++) {
        scaleA_out[mi] = 1.0f; inv[mi] = 0.0f;
    }
    for (int mi = 0; mi < m_full; mi += MR) {
        int8_t *dst = pack + mi * kc;
        const float *r[MR]; float iv[MR];
        for (int t = 0; t < MR; t++) { r[t] = A + (i_start + mi + t) * lda + k_start; iv[t] = inv[mi + t]; }
        for (int k4 = 0; k4 < kc; k4 += 4) {
            int8_t *chunk = dst + (k4 >> 2) * (MR * 4);
            for (int t = 0; t < MR; t++) {
                int8_t *row_dst = chunk + t * 4;
                row_dst[0] = saturate_i8(lrintf(r[t][k4 + 0] * iv[t]));
                row_dst[1] = saturate_i8(lrintf(r[t][k4 + 1] * iv[t]));
                row_dst[2] = saturate_i8(lrintf(r[t][k4 + 2] * iv[t]));
                row_dst[3] = saturate_i8(lrintf(r[t][k4 + 3] * iv[t]));
            }
        }
    }
    if (m_full < mc) {
        int tail = mc - m_full;
        int8_t *dst = pack + m_full * kc;
        for (int k4 = 0; k4 < kc; k4 += 4) {
            int8_t *chunk = dst + (k4 >> 2) * (MR * 4);
            for (int t = 0; t < MR; t++) {
                int8_t *row_dst = chunk + t * 4;
                if (t < tail) {
                    const float *row = A + (i_start + m_full + t) * lda + k_start;
                    float iv_t = inv[m_full + t];
                    row_dst[0] = saturate_i8(lrintf(row[k4 + 0] * iv_t));
                    row_dst[1] = saturate_i8(lrintf(row[k4 + 1] * iv_t));
                    row_dst[2] = saturate_i8(lrintf(row[k4 + 2] * iv_t));
                    row_dst[3] = saturate_i8(lrintf(row[k4 + 3] * iv_t));
                } else {
                    row_dst[0] = row_dst[1] = row_dst[2] = row_dst[3] = 0;
                }
            }
        }
    }
}

// Packed int8 B layout, per NR-col tile: (kc/4) chunks × (NR * 4 int8) = kc*NR int8.
static void pack_B_s8_repack(const int8_t *B, int ldb, const float *scale,
                             int k_start, int j_start, int kc, int nc,
                             int8_t *pack, float *scaleB_out) {
    int n_full = (nc / NR) * NR;
    for (int nj = 0; nj < n_full; nj += NR) {
        int8_t *dst = pack + nj * kc;
        for (int t = 0; t < NR; t++) scaleB_out[nj + t] = scale[j_start + nj + t];
        for (int k4 = 0; k4 < kc; k4 += 4) {
            int8_t *chunk = dst + (k4 >> 2) * (NR * 4);
            for (int t = 0; t < NR; t++) {
                const int8_t *src = B + (j_start + nj + t) * ldb + (k_start + k4);
                int8_t *col_dst = chunk + t * 4;
                col_dst[0] = src[0]; col_dst[1] = src[1];
                col_dst[2] = src[2]; col_dst[3] = src[3];
            }
        }
    }
    if (n_full < nc) {
        int tail = nc - n_full;
        int8_t *dst = pack + n_full * kc;
        for (int t = 0; t < tail; t++) scaleB_out[n_full + t] = scale[j_start + n_full + t];
        for (int t = tail; t < NR; t++) scaleB_out[n_full + t] = 1.0f;
        for (int k4 = 0; k4 < kc; k4 += 4) {
            int8_t *chunk = dst + (k4 >> 2) * (NR * 4);
            for (int t = 0; t < NR; t++) {
                int8_t *col_dst = chunk + t * 4;
                if (t < tail) {
                    const int8_t *src = B + (j_start + n_full + t) * ldb + (k_start + k4);
                    col_dst[0] = src[0]; col_dst[1] = src[1];
                    col_dst[2] = src[2]; col_dst[3] = src[3];
                } else {
                    col_dst[0] = col_dst[1] = col_dst[2] = col_dst[3] = 0;
                }
            }
        }
    }
}

#if HAS_SDOT
static inline void micro_kernel_s8_4x8(int K, float alpha, float beta,
                                       const int8_t *packA, const float *a_scale,
                                       const int8_t *packB, const float *b_scale,
                                       float *C, int ldc) {
    int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0);
    int32x4_t c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
    int32x4_t c4 = vdupq_n_s32(0), c5 = vdupq_n_s32(0);
    int32x4_t c6 = vdupq_n_s32(0), c7 = vdupq_n_s32(0);
    int chunks = K >> 2;
    for (int i = 0; i < chunks; i++) {
        int8x16_t a  = vld1q_s8(packA);
        int8x16_t b0 = vld1q_s8(packB);
        int8x16_t b1 = vld1q_s8(packB + 16);
        c0 = vdotq_laneq_s32(c0, a, b0, 0);
        c1 = vdotq_laneq_s32(c1, a, b0, 1);
        c2 = vdotq_laneq_s32(c2, a, b0, 2);
        c3 = vdotq_laneq_s32(c3, a, b0, 3);
        c4 = vdotq_laneq_s32(c4, a, b1, 0);
        c5 = vdotq_laneq_s32(c5, a, b1, 1);
        c6 = vdotq_laneq_s32(c6, a, b1, 2);
        c7 = vdotq_laneq_s32(c7, a, b1, 3);
        packA += 16; packB += 32;
    }
    float32x4_t as = vmulq_n_f32(vld1q_f32(a_scale), alpha);
    int32x4_t cv[NR] = { c0, c1, c2, c3, c4, c5, c6, c7 };
    if (beta == 0.0f) {
        for (int j = 0; j < NR; j++) {
            float32x4_t cf = vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(cv[j]), as), b_scale[j]);
            C[0 * ldc + j] = vgetq_lane_f32(cf, 0); C[1 * ldc + j] = vgetq_lane_f32(cf, 1);
            C[2 * ldc + j] = vgetq_lane_f32(cf, 2); C[3 * ldc + j] = vgetq_lane_f32(cf, 3);
        }
    } else {
        for (int j = 0; j < NR; j++) {
            float32x4_t cf = vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(cv[j]), as), b_scale[j]);
            C[0 * ldc + j] += vgetq_lane_f32(cf, 0); C[1 * ldc + j] += vgetq_lane_f32(cf, 1);
            C[2 * ldc + j] += vgetq_lane_f32(cf, 2); C[3 * ldc + j] += vgetq_lane_f32(cf, 3);
        }
    }
}
#endif

static inline void micro_kernel_s8_edge(int K, float alpha, float beta,
                                        const int8_t *packA, const float *a_scale,
                                        const int8_t *packB, const float *b_scale,
                                        float *C, int ldc,
                                        int mr_tail, int nr_tail) {
    int32_t acc[MR][NR] = {{0}};
    int chunks = K >> 2;
    for (int i = 0; i < chunks; i++) {
        const int8_t *a_chunk = packA + i * (MR * 4);
        const int8_t *b_chunk = packB + i * (NR * 4);
        for (int mi = 0; mi < mr_tail; mi++) {
            for (int nj = 0; nj < nr_tail; nj++) {
                int32_t s = 0;
                for (int l = 0; l < 4; l++) s += (int32_t)a_chunk[mi * 4 + l] * (int32_t)b_chunk[nj * 4 + l];
                acc[mi][nj] += s;
            }
        }
    }
    for (int mi = 0; mi < mr_tail; mi++) {
        for (int nj = 0; nj < nr_tail; nj++) {
            float v = alpha * a_scale[mi] * b_scale[nj] * (float)acc[mi][nj];
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}

#endif // HAS_SMMLA

static void macro_kernel_s8(int mc, int nc, int kc, float alpha, float beta,
                            const int8_t *packA, const float *a_scale,
                            const int8_t *packB, const float *b_scale,
                            float *C, int ldc) {
    for (int mi = 0; mi < mc; mi += MR) {
        int mr_tail = (mc - mi < MR) ? (mc - mi) : MR;
        for (int nj = 0; nj < nc; nj += NR) {
            int nr_tail = (nc - nj < NR) ? (nc - nj) : NR;
            const int8_t *pA = packA + mi * kc;
            const int8_t *pB = packB + nj * kc;
            float *pC = C + mi * ldc + nj;
#if HAS_SMMLA || HAS_SDOT
            if (mr_tail == MR && nr_tail == NR) {
                micro_kernel_s8_4x8(kc, alpha, beta, pA, a_scale + mi, pB, b_scale + nj, pC, ldc);
                continue;
            }
#endif
            micro_kernel_s8_edge(kc, alpha, beta, pA, a_scale + mi, pB, b_scale + nj, pC, ldc,
                                 mr_tail, nr_tail);
        }
    }
}

#endif // HAS_SMMLA || HAS_SDOT (end of int8 pack/kernels block)

// ---------------------------------------------------------------------------
// AVX-VNNI int8 path. `_mm256_dpbusd_epi32` does 32 int8 MACs per instruction —
// x86 counterpart to ARM SMMLA. Requires one operand to be *unsigned* int8, so
// we offset A by +128 at pack time and subtract a per-column correction
// (128 × sum-of-B) at the epilogue.
// ---------------------------------------------------------------------------
#if HAS_VNNI_S8 && !HAS_SMMLA && !HAS_SDOT

static inline int8_t vnni_saturate_i8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Pack A: quantize fp32 → int8, then +128 → uint8. Tight row-major.
static void pack_A_s8_quant_vnni(const float *A, int lda,
                                 int i_start, int k_start, int mc, int kc,
                                 uint8_t *pack, float *scaleA_out) {
    int m_full = (mc / MR) * MR;
    for (int mi = 0; mi < mc; mi++) {
        const float *row = A + (i_start + mi) * lda + k_start;
        float mx = 0.0f;
        for (int k = 0; k < kc; k++) { float v = fabsf(row[k]); if (v > mx) mx = v; }
        float s = mx / 127.0f;
        float inv = (s == 0.0f) ? 0.0f : 1.0f / s;
        scaleA_out[mi] = (s == 0.0f) ? 1.0f : s;
        uint8_t *dst = pack + mi * kc;
        for (int k = 0; k < kc; k++) {
            int8_t q = vnni_saturate_i8(lrintf(row[k] * inv));
            dst[k] = (uint8_t)((int32_t)q + 128);
        }
    }
    // Zero-pad (with the u8 "zero" which is 128) so the micro-kernel reads MR rows.
    for (int mi = mc; mi < m_full + MR; mi++) {
        scaleA_out[mi] = 1.0f;
        memset(pack + mi * kc, 128, kc);
    }
}

// Pack B: copy int8 weights tight + compute per-col correction = 128 × sum_B.
static void pack_B_s8_repack_vnni(const int8_t *B, int ldb, const float *scale,
                                  int k_start, int j_start, int kc, int nc,
                                  int8_t *pack, float *scaleB_out, int32_t *corrB_out) {
    int n_full = (nc / NR) * NR;
    for (int nj = 0; nj < nc; nj++) {
        scaleB_out[nj] = scale[j_start + nj];
        const int8_t *src = B + (j_start + nj) * ldb + k_start;
        memcpy(pack + nj * kc, src, kc);
        int32_t sum = 0;
        for (int k = 0; k < kc; k++) sum += (int32_t)src[k];
        corrB_out[nj] = 128 * sum;
    }
    for (int nj = nc; nj < n_full + NR; nj++) {
        scaleB_out[nj] = 1.0f;
        corrB_out[nj] = 0;
        memset(pack + nj * kc, 0, kc);
    }
}

static inline int32_t vnni_reduce_i32x8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// 4×2 micro-kernel via VNNI dpbusd. K must be a multiple of 32.
// 8 persistent int32x8 accumulators + 6 transient loads = 14 YMM, fits budget.
static inline void micro_kernel_s8_4x2_vnni(int K, float alpha, float beta,
                                            const uint8_t *packA, int lda,
                                            const float *a_scale,
                                            const int8_t *packB, int ldb,
                                            const float *b_scale,
                                            const int32_t *b_corr,
                                            float *C, int ldc) {
    __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
    for (int k = 0; k < K; k += 32) {
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(packA + 0 * lda + k));
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(packA + 1 * lda + k));
        __m256i a2 = _mm256_loadu_si256((const __m256i *)(packA + 2 * lda + k));
        __m256i a3 = _mm256_loadu_si256((const __m256i *)(packA + 3 * lda + k));
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(packB + 0 * ldb + k));
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(packB + 1 * ldb + k));
        c00 = _mm256_dpbusd_epi32(c00, a0, b0);
        c01 = _mm256_dpbusd_epi32(c01, a0, b1);
        c10 = _mm256_dpbusd_epi32(c10, a1, b0);
        c11 = _mm256_dpbusd_epi32(c11, a1, b1);
        c20 = _mm256_dpbusd_epi32(c20, a2, b0);
        c21 = _mm256_dpbusd_epi32(c21, a2, b1);
        c30 = _mm256_dpbusd_epi32(c30, a3, b0);
        c31 = _mm256_dpbusd_epi32(c31, a3, b1);
    }
    int32_t d[4][2] = {
        { vnni_reduce_i32x8(c00), vnni_reduce_i32x8(c01) },
        { vnni_reduce_i32x8(c10), vnni_reduce_i32x8(c11) },
        { vnni_reduce_i32x8(c20), vnni_reduce_i32x8(c21) },
        { vnni_reduce_i32x8(c30), vnni_reduce_i32x8(c31) },
    };
    for (int mi = 0; mi < 4; mi++) {
        float a_coef = alpha * a_scale[mi];
        for (int nj = 0; nj < 2; nj++) {
            int32_t corrected = d[mi][nj] - b_corr[nj];
            float v = a_coef * b_scale[nj] * (float)corrected;
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}

// Scalar edge for partial tiles.
static inline void micro_kernel_s8_edge_vnni(int K, float alpha, float beta,
                                             const uint8_t *packA, int lda,
                                             const float *a_scale,
                                             const int8_t *packB, int ldb,
                                             const float *b_scale,
                                             const int32_t *b_corr,
                                             float *C, int ldc,
                                             int mr_tail, int nr_tail) {
    for (int mi = 0; mi < mr_tail; mi++) {
        for (int nj = 0; nj < nr_tail; nj++) {
            int32_t s = 0;
            const uint8_t *a_row = packA + mi * lda;
            const int8_t  *b_col = packB + nj * ldb;
            for (int k = 0; k < K; k++) s += (int32_t)a_row[k] * (int32_t)b_col[k];
            int32_t corrected = s - b_corr[nj];
            float v = alpha * a_scale[mi] * b_scale[nj] * (float)corrected;
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}

static void macro_kernel_s8_vnni(int mc, int nc, int kc, float alpha, float beta,
                                 const uint8_t *packA, const float *a_scale,
                                 const int8_t *packB, const float *b_scale,
                                 const int32_t *b_corr,
                                 float *C, int ldc) {
    for (int mi = 0; mi < mc; mi += MR) {
        int mr_tail = (mc - mi < MR) ? (mc - mi) : MR;
        for (int nj = 0; nj < nc; nj += NR) {
            int nr_tail = (nc - nj < NR) ? (nc - nj) : NR;
            const uint8_t *pA = packA + mi * kc;
            const int8_t  *pB = packB + nj * kc;
            float *pC = C + mi * ldc + nj;
            if (mr_tail == MR && nr_tail == NR) {
                for (int sub = 0; sub < NR; sub += 2) {
                    micro_kernel_s8_4x2_vnni(kc, alpha, beta,
                                             pA, kc, a_scale + mi,
                                             pB + sub * kc, kc, b_scale + nj + sub, b_corr + nj + sub,
                                             pC + sub, ldc);
                }
            } else {
                micro_kernel_s8_edge_vnni(kc, alpha, beta,
                                          pA, kc, a_scale + mi,
                                          pB, kc, b_scale + nj, b_corr + nj,
                                          pC, ldc, mr_tail, nr_tail);
            }
        }
    }
}

#endif // HAS_VNNI_S8

// ---------------------------------------------------------------------------
// AVX2 int8 path (x86 without SDOT/SMMLA). Uses a tight (rows × kc) pack
// layout — different from SDOT/SMMLA because madd_epi16 needs contiguous
// per-row/per-col data to produce meaningful pair-sums.
//
// Micro-kernel: MR=4 rows × NR=8 cols processed as four 4×2 sub-tiles to keep
// accumulator pressure within AVX2's 16 YMM registers.
// ---------------------------------------------------------------------------
#if HAS_AVX2_S8 && !HAS_SMMLA && !HAS_SDOT && !HAS_VNNI_S8

static inline int8_t avx2_saturate_i8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Pack A: quantize mc fp32 rows × kc cols to int8, tight row-major.
// packA layout: [mc × kc] int8, row i at offset i*kc. Rows zero-padded to MR.
static void pack_A_s8_quant_avx2(const float *A, int lda,
                                 int i_start, int k_start, int mc, int kc,
                                 int8_t *pack, float *scaleA_out) {
    int m_full = (mc / MR) * MR;
    for (int mi = 0; mi < mc; mi++) {
        const float *row = A + (i_start + mi) * lda + k_start;
        float mx = 0.0f;
        for (int k = 0; k < kc; k++) { float v = fabsf(row[k]); if (v > mx) mx = v; }
        float s = mx / 127.0f;
        float inv = (s == 0.0f) ? 0.0f : 1.0f / s;
        scaleA_out[mi] = (s == 0.0f) ? 1.0f : s;
        int8_t *dst = pack + mi * kc;
        for (int k = 0; k < kc; k++) dst[k] = avx2_saturate_i8(lrintf(row[k] * inv));
    }
    // Zero-pad missing rows of the final partial MR tile to keep the micro-kernel
    // branch-free (it still reads MR rows).
    for (int mi = mc; mi < m_full + MR; mi++) {
        scaleA_out[mi] = 1.0f;
        memset(pack + mi * kc, 0, kc);
    }
}

// Pack B: copy int8 weights into tight row-major (nc × kc) + per-col scales.
static void pack_B_s8_repack_avx2(const int8_t *B, int ldb, const float *scale,
                                  int k_start, int j_start, int kc, int nc,
                                  int8_t *pack, float *scaleB_out) {
    int n_full = (nc / NR) * NR;
    for (int nj = 0; nj < nc; nj++) {
        scaleB_out[nj] = scale[j_start + nj];
        memcpy(pack + nj * kc, B + (j_start + nj) * ldb + k_start, kc);
    }
    for (int nj = nc; nj < n_full + NR; nj++) {
        scaleB_out[nj] = 1.0f;
        memset(pack + nj * kc, 0, kc);
    }
}

// Horizontal reduce int32x8 → scalar int32.
static inline int32_t avx2_reduce_i32x8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// 4×2 micro-kernel: 8 persistent int32x8 accumulators (well within YMM budget).
// K must be a multiple of 16 (S8_K_ALIGN on AVX2).
static inline void micro_kernel_s8_4x2_avx2(int K, float alpha, float beta,
                                            const int8_t *packA, int lda,
                                            const float *a_scale,
                                            const int8_t *packB, int ldb,
                                            const float *b_scale,
                                            float *C, int ldc) {
    __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
    for (int k = 0; k < K; k += 16) {
        __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(packA + 0 * lda + k)));
        __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(packA + 1 * lda + k)));
        __m256i a2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(packA + 2 * lda + k)));
        __m256i a3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(packA + 3 * lda + k)));
        __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(packB + 0 * ldb + k)));
        __m256i b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(packB + 1 * ldb + k)));
        c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(a0, b0));
        c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(a0, b1));
        c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(a1, b0));
        c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(a1, b1));
        c20 = _mm256_add_epi32(c20, _mm256_madd_epi16(a2, b0));
        c21 = _mm256_add_epi32(c21, _mm256_madd_epi16(a2, b1));
        c30 = _mm256_add_epi32(c30, _mm256_madd_epi16(a3, b0));
        c31 = _mm256_add_epi32(c31, _mm256_madd_epi16(a3, b1));
    }
    int32_t d[4][2] = {
        { avx2_reduce_i32x8(c00), avx2_reduce_i32x8(c01) },
        { avx2_reduce_i32x8(c10), avx2_reduce_i32x8(c11) },
        { avx2_reduce_i32x8(c20), avx2_reduce_i32x8(c21) },
        { avx2_reduce_i32x8(c30), avx2_reduce_i32x8(c31) },
    };
    for (int mi = 0; mi < 4; mi++) {
        float a_coef = alpha * a_scale[mi];
        for (int nj = 0; nj < 2; nj++) {
            float v = a_coef * b_scale[nj] * (float)d[mi][nj];
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}

// Scalar edge for partial tiles (mr_tail < MR or nr_tail < NR).
static inline void micro_kernel_s8_edge_avx2(int K, float alpha, float beta,
                                             const int8_t *packA, int lda,
                                             const float *a_scale,
                                             const int8_t *packB, int ldb,
                                             const float *b_scale,
                                             float *C, int ldc,
                                             int mr_tail, int nr_tail) {
    for (int mi = 0; mi < mr_tail; mi++) {
        for (int nj = 0; nj < nr_tail; nj++) {
            int32_t s = 0;
            const int8_t *a_row = packA + mi * lda;
            const int8_t *b_col = packB + nj * ldb;
            for (int k = 0; k < K; k++) s += (int32_t)a_row[k] * (int32_t)b_col[k];
            float v = alpha * a_scale[mi] * b_scale[nj] * (float)s;
            C[mi * ldc + nj] = (beta != 0.0f) ? (C[mi * ldc + nj] + v) : v;
        }
    }
}

static void macro_kernel_s8_avx2(int mc, int nc, int kc, float alpha, float beta,
                                 const int8_t *packA, const float *a_scale,
                                 const int8_t *packB, const float *b_scale,
                                 float *C, int ldc) {
    // Tight layout: lda=kc for packA (one row per MR slot), ldb=kc for packB.
    for (int mi = 0; mi < mc; mi += MR) {
        int mr_tail = (mc - mi < MR) ? (mc - mi) : MR;
        for (int nj = 0; nj < nc; nj += NR) {
            int nr_tail = (nc - nj < NR) ? (nc - nj) : NR;
            const int8_t *pA = packA + mi * kc;
            const int8_t *pB = packB + nj * kc;
            float *pC = C + mi * ldc + nj;
            if (mr_tail == MR && nr_tail == NR) {
                // Full tile: four 4×2 sub-kernels.
                for (int sub = 0; sub < NR; sub += 2) {
                    micro_kernel_s8_4x2_avx2(kc, alpha, beta,
                                             pA,         kc, a_scale + mi,
                                             pB + sub*kc, kc, b_scale + nj + sub,
                                             pC + sub, ldc);
                }
            } else {
                micro_kernel_s8_edge_avx2(kc, alpha, beta,
                                          pA, kc, a_scale + mi,
                                          pB, kc, b_scale + nj,
                                          pC, ldc, mr_tail, nr_tail);
            }
        }
    }
}

#endif // HAS_AVX2_S8 && !HAS_SMMLA && !HAS_SDOT

// Fallback for hosts without SDOT/SMMLA (x86, old ARM): dequantize an int8 B
// panel to fp32 at pack time, then run the fp32 micro-kernel. Slower than real
// int8 compute but lets x86 users get fp32-SIMD performance on weights-int8
// (plus the 3.7× RSS savings of int8 weights).
#if !HAS_SMMLA && !HAS_SDOT && !HAS_AVX2_S8 && !HAS_VNNI_S8
static void pack_B_s8_dequant(const int8_t *B, int ldb, const float *scale,
                              int k_start, int j_start, int kc, int nc,
                              float *pack) {
    int n_full = (nc / NR) * NR;
    for (int nj = 0; nj < n_full; nj += NR) {
        float *dst = pack + nj * kc;
        float s[NR];
        for (int j = 0; j < NR; j++) s[j] = scale[j_start + nj + j];
        for (int k = 0; k < kc; k++) {
            for (int j = 0; j < NR; j++) {
                dst[k * NR + j] =
                    (float)B[(j_start + nj + j) * ldb + (k_start + k)] * s[j];
            }
        }
    }
    if (n_full < nc) {
        int tail = nc - n_full;
        float *dst = pack + n_full * kc;
        float s[NR];
        for (int j = 0; j < tail; j++) s[j] = scale[j_start + n_full + j];
        for (int k = 0; k < kc; k++) {
            int j;
            for (j = 0; j < tail; j++) {
                dst[k * NR + j] =
                    (float)B[(j_start + n_full + j) * ldb + (k_start + k)] * s[j];
            }
            for (; j < NR; j++) dst[k * NR + j] = 0.0f;
        }
    }
}
#endif

static void run_job(const job_t *j) {
    int M = j->m_end - j->m_start;
    int N_start = j->n_start, N_end = j->n_end;
    int K = j->K;
    bool is_int8 = (j->B_s8 != NULL) || (j->B_blocks != NULL);

    if (is_int8) {
#if HAS_SMMLA || HAS_SDOT
        // W8A8 path (real int8 compute). K and every kc slice must be multiples of S8_K_ALIGN.
        size_t pA_cap = (size_t)((MC + MR) * KC);   // int8 bytes
        size_t pB_cap = (size_t)((NC + NR) * KC);   // int8 bytes
        int8_t *packA  = (int8_t *)aligned_alloc(PACK_ALIGN, ((pA_cap + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        int8_t *packB  = (int8_t *)aligned_alloc(PACK_ALIGN, ((pB_cap + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        float  *scaleA = (float  *)aligned_alloc(PACK_ALIGN, ((size_t)(MC + MR) * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN);
        float  *scaleB = (float  *)aligned_alloc(PACK_ALIGN, ((size_t)(NC + NR) * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN);

        for (int k_outer = 0; k_outer < K; k_outer += KC) {
            int kc = (K - k_outer < KC) ? (K - k_outer) : KC;
            float beta = (k_outer == 0) ? 0.0f : 1.0f;
            int kb = k_outer / KC;
            for (int j_outer = N_start; j_outer < N_end; j_outer += NC) {
                int nc = (N_end - j_outer < NC) ? (N_end - j_outer) : NC;
                const int8_t *packB_use;
                const float  *scaleB_use;
                if (j->B_blocks) {
                    // Pre-packed path: point at the matching block + scale slice.
                    // With N-split, j_outer can start mid-block — offset into
                    // the packed block to the correct NR-aligned tile.
                    int jb = j_outer / NC;
                    int nj_in_block = j_outer - jb * NC;   // multiple of NR
                    packB_use = j->B_blocks[kb * j->num_jb + jb]
                                + (size_t)nj_in_block * kc;
                    scaleB_use = j->B_scale + j_outer;
                } else {
                    pack_B_s8_repack(j->B_s8, j->ldb, j->B_scale, k_outer, j_outer, kc, nc,
                                     packB, scaleB);
                    packB_use = packB;
                    scaleB_use = scaleB;
                }
                for (int i_outer = 0; i_outer < M; i_outer += MC) {
                    int mc = (M - i_outer < MC) ? (M - i_outer) : MC;
                    pack_A_s8_quant(j->A, j->lda, j->m_start + i_outer, k_outer, mc, kc,
                                    packA, scaleA);
                    macro_kernel_s8(mc, nc, kc, j->alpha, beta,
                                    packA, scaleA, packB_use, scaleB_use,
                                    j->C + (j->m_start + i_outer) * j->ldc + j_outer,
                                    j->ldc);
                }
            }
        }
        free(packA); free(packB); free(scaleA); free(scaleB);
        return;
#elif HAS_VNNI_S8
        // x86 AVX-VNNI W8A8: uint8 A (offset +128) × int8 B via dpbusd, plus
        // per-column correction = 128 × sum_B to back out the offset.
        size_t pA_cap = (size_t)((MC + MR) * KC);
        size_t pB_cap = (size_t)((NC + NR) * KC);
        uint8_t *packA  = (uint8_t *)aligned_alloc(PACK_ALIGN, ((pA_cap + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        int8_t  *packB  = (int8_t  *)aligned_alloc(PACK_ALIGN, ((pB_cap + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        float   *scaleA = (float   *)aligned_alloc(PACK_ALIGN, ((size_t)(MC + MR) * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN);
        float   *scaleB = (float   *)aligned_alloc(PACK_ALIGN, ((size_t)(NC + NR) * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN);
        int32_t *corrB  = (int32_t *)aligned_alloc(PACK_ALIGN, ((size_t)(NC + NR) * sizeof(int32_t) + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN);

        for (int k_outer = 0; k_outer < K; k_outer += KC) {
            int kc = (K - k_outer < KC) ? (K - k_outer) : KC;
            float beta = (k_outer == 0) ? 0.0f : 1.0f;
            for (int j_outer = N_start; j_outer < N_end; j_outer += NC) {
                int nc = (N_end - j_outer < NC) ? (N_end - j_outer) : NC;
                pack_B_s8_repack_vnni(j->B_s8, j->ldb, j->B_scale, k_outer, j_outer, kc, nc,
                                      packB, scaleB, corrB);
                for (int i_outer = 0; i_outer < M; i_outer += MC) {
                    int mc = (M - i_outer < MC) ? (M - i_outer) : MC;
                    pack_A_s8_quant_vnni(j->A, j->lda, j->m_start + i_outer, k_outer, mc, kc,
                                         packA, scaleA);
                    macro_kernel_s8_vnni(mc, nc, kc, j->alpha, beta,
                                         packA, scaleA, packB, scaleB, corrB,
                                         j->C + (j->m_start + i_outer) * j->ldc + j_outer,
                                         j->ldc);
                }
            }
        }
        free(packA); free(packB); free(scaleA); free(scaleB); free(corrB);
        return;
#elif HAS_AVX2_S8
        // x86 AVX2 W8A8: tight (rows × kc) int8 pack + widen+madd micro-kernel.
        size_t pA_cap = (size_t)((MC + MR) * KC);
        size_t pB_cap = (size_t)((NC + NR) * KC);
        int8_t *packA  = (int8_t *)aligned_alloc(PACK_ALIGN, ((pA_cap + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        int8_t *packB  = (int8_t *)aligned_alloc(PACK_ALIGN, ((pB_cap + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        float  *scaleA = (float  *)aligned_alloc(PACK_ALIGN, ((size_t)(MC + MR) * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN);
        float  *scaleB = (float  *)aligned_alloc(PACK_ALIGN, ((size_t)(NC + NR) * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN);

        for (int k_outer = 0; k_outer < K; k_outer += KC) {
            int kc = (K - k_outer < KC) ? (K - k_outer) : KC;
            float beta = (k_outer == 0) ? 0.0f : 1.0f;
            for (int j_outer = N_start; j_outer < N_end; j_outer += NC) {
                int nc = (N_end - j_outer < NC) ? (N_end - j_outer) : NC;
                pack_B_s8_repack_avx2(j->B_s8, j->ldb, j->B_scale, k_outer, j_outer, kc, nc,
                                      packB, scaleB);
                for (int i_outer = 0; i_outer < M; i_outer += MC) {
                    int mc = (M - i_outer < MC) ? (M - i_outer) : MC;
                    pack_A_s8_quant_avx2(j->A, j->lda, j->m_start + i_outer, k_outer, mc, kc,
                                         packA, scaleA);
                    macro_kernel_s8_avx2(mc, nc, kc, j->alpha, beta,
                                         packA, scaleA, packB, scaleB,
                                         j->C + (j->m_start + i_outer) * j->ldc + j_outer,
                                         j->ldc);
                }
            }
        }
        free(packA); free(packB); free(scaleA); free(scaleB);
        return;
#else
        // x86 / old-ARM fallback: dequantize int8 B to fp32 at pack time, run
        // the fp32 micro-kernel. No dynamic activation quant → no accuracy
        // drift beyond the weight quantization itself.
        size_t pA_cap = (size_t)((MC + MR) * KC);
        size_t pB_cap = (size_t)((NC + NR) * KC);
        float *packA = (float *)aligned_alloc(PACK_ALIGN, ((pA_cap * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        float *packB = (float *)aligned_alloc(PACK_ALIGN, ((pB_cap * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
        for (int k_outer = 0; k_outer < K; k_outer += KC) {
            int kc = (K - k_outer < KC) ? (K - k_outer) : KC;
            float beta = (k_outer == 0) ? 0.0f : 1.0f;
            for (int j_outer = N_start; j_outer < N_end; j_outer += NC) {
                int nc = (N_end - j_outer < NC) ? (N_end - j_outer) : NC;
                pack_B_s8_dequant(j->B_s8, j->ldb, j->B_scale, k_outer, j_outer, kc, nc, packB);
                for (int i_outer = 0; i_outer < M; i_outer += MC) {
                    int mc = (M - i_outer < MC) ? (M - i_outer) : MC;
                    pack_A(j->A, j->lda, j->m_start + i_outer, k_outer, mc, kc, packA);
                    macro_kernel(mc, nc, kc, j->alpha, beta, packA, packB,
                                 j->C + (j->m_start + i_outer) * j->ldc + j_outer, j->ldc);
                }
            }
        }
        free(packA); free(packB);
        return;
#endif
    }

    // FP32 path: only reached when B is fp32 (sgemm_rm callers).
    size_t pA_cap = (size_t)((MC + MR) * KC);
    size_t pB_cap = (size_t)((NC + NR) * KC);
    float *packA = (float *)aligned_alloc(PACK_ALIGN, ((pA_cap * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
    float *packB = (float *)aligned_alloc(PACK_ALIGN, ((pB_cap * sizeof(float) + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);

    for (int k_outer = 0; k_outer < K; k_outer += KC) {
        int kc = (K - k_outer < KC) ? (K - k_outer) : KC;
        float beta = (k_outer == 0) ? 0.0f : 1.0f;
        for (int j_outer = N_start; j_outer < N_end; j_outer += NC) {
            int nc = (N_end - j_outer < NC) ? (N_end - j_outer) : NC;
            pack_B(j->B, j->ldb, k_outer, j_outer, kc, nc, j->trans_b, packB);

            for (int i_outer = 0; i_outer < M; i_outer += MC) {
                int mc = (M - i_outer < MC) ? (M - i_outer) : MC;
                pack_A(j->A, j->lda, j->m_start + i_outer, k_outer, mc, kc, packA);

                macro_kernel(mc, nc, kc, j->alpha, beta,
                             packA, packB,
                             j->C + (j->m_start + i_outer) * j->ldc + j_outer,
                             j->ldc);
            }
        }
    }
    free(packA);
    free(packB);
}

static void *worker_main(void *arg) {
    pool_t *p = (pool_t *)arg;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->shutdown && p->jobs_pending == 0) pthread_cond_wait(&p->cv_work, &p->mu);
        if (p->shutdown && p->jobs_pending == 0) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        int jid = --p->jobs_pending;
        pthread_mutex_unlock(&p->mu);

        run_job(&p->jobs[jid]);

        pthread_mutex_lock(&p->mu);
        p->active_threads--;
        if (p->active_threads == 0) pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
}

static pool_t *pool_create(int n) {
    if (n > MAX_THREADS) n = MAX_THREADS;
    pool_t *p = (pool_t *)calloc(1, sizeof(pool_t));
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    p->size = n;
    for (int i = 0; i < n; i++) pthread_create(&p->threads[i], NULL, worker_main, p);
    return p;
}

static void pool_destroy(void *arg) {
    pool_t *p = (pool_t *)arg;
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->shutdown = true;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->size; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    free(p);
}

// TLS registration. Key destructor runs when the calling thread exits so the
// per-thread pool is torn down cleanly.
static pthread_key_t g_key;
static pthread_once_t g_key_once = PTHREAD_ONCE_INIT;
static void key_init(void) { pthread_key_create(&g_key, pool_destroy); }

// Per-thread requested concurrency. 0/1 means "don't bother with a pool".
static __thread int tls_user_threads = 0;

void sgemm_set_threads(int n) { tls_user_threads = n; }

static int decide_threads(int M, int N, int K) {
    int n = tls_user_threads;
    if (n < 1) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;

    // Work-per-thread floor: a pool wake-up is ~3-10 μs; amortize it against a
    // GEMM big enough for each thread to do meaningful work. The per-head
    // attention ops (99×99×64 ≈ 625K MACs) fall below this and run single-threaded.
    long work = (long)M * (long)N * (long)K;
    const long MIN_WORK_PER_THREAD = 2L * 1024 * 1024;  // ~2M MACs
    long max_n = work / MIN_WORK_PER_THREAD;
    if (max_n < 1) max_n = 1;
    if (max_n < n) n = (int)max_n;

    // dispatch() splits on whichever of (M, N) is larger. Each thread must own
    // at least one full micro-tile of that axis, else we'd dispatch empty jobs.
    int split_dim  = (N >= M) ? N  : M;
    int split_unit = (N >= M) ? (NR * 2) : (MR * 2);
    if (split_dim < split_unit * n) {
        int n2 = split_dim / split_unit;
        if (n2 < 1) n2 = 1;
        if (n2 < n) n = n2;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------
// Internal dispatcher used by both sgemm_rm and sgemm_s8rm. The `template`
// job_t has everything filled in except m_start/m_end.
static void dispatch(int M, const job_t *tpl) {
    int N = tpl->N;
    int nthreads = decide_threads(M, N, tpl->K);
    if (nthreads <= 1) {
        job_t j = *tpl;
        j.m_start = 0;       j.m_end = M;
        j.n_start = 0;       j.n_end = N;
        run_job(&j);
        return;
    }

    pthread_once(&g_key_once, key_init);
    pool_t *p = (pool_t *)pthread_getspecific(g_key);
    if (!p || p->size < nthreads) {
        if (p) { pthread_setspecific(g_key, NULL); pool_destroy(p); }
        p = pool_create(nthreads);
        pthread_setspecific(g_key, p);
    }

    // Pick the partition axis: whichever is larger wins, because packing the
    // "other" dim is redundant work per thread. For typical transformer linears
    // (M=T≈99, N∈{1024,4096}) N dominates, so we split columns — each thread
    // packs its own disjoint B slice instead of re-packing the full B.
    int k = 0;
    if (N >= M) {
        int per = ((N + nthreads - 1) / nthreads + NR - 1) / NR * NR;
        for (int s = 0; s < N && k < nthreads; s += per) {
            int e = s + per; if (e > N) e = N;
            p->jobs[k] = *tpl;
            p->jobs[k].m_start = 0;  p->jobs[k].m_end = M;
            p->jobs[k].n_start = s;  p->jobs[k].n_end = e;
            k++;
        }
    } else {
        int per = ((M + nthreads - 1) / nthreads + MR - 1) / MR * MR;
        for (int s = 0; s < M && k < nthreads; s += per) {
            int e = s + per; if (e > M) e = M;
            p->jobs[k] = *tpl;
            p->jobs[k].m_start = s;  p->jobs[k].m_end = e;
            p->jobs[k].n_start = 0;  p->jobs[k].n_end = N;
            k++;
        }
    }

    pthread_mutex_lock(&p->mu);
    p->jobs_pending = k;
    p->active_threads = k;
    pthread_cond_broadcast(&p->cv_work);
    while (p->active_threads > 0) pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

void sgemm_rm(bool trans_b,
              int M, int N, int K,
              float alpha,
              const float *A, int lda,
              const float *B, int ldb,
              float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    job_t tpl = {
        .N = N, .K = K, .alpha = alpha,
        .A = A, .lda = lda,
        .B = B, .B_s8 = NULL, .B_scale = NULL, .ldb = ldb,
        .trans_b = trans_b,
        .C = C, .ldc = ldc,
    };
    dispatch(M, &tpl);
}

void sgemm_s8rm(int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const int8_t *B, const float *scale, int ldb,
                float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    // K must be a multiple of the active kernel's k-chunk (4 for SDOT, 8 for
    // SMMLA). All wav2vec2 linear K values satisfy this; trip loudly otherwise.
    if (K % S8_K_ALIGN != 0) {
        fprintf(stderr, "sgemm_s8rm: K=%d must be a multiple of %d\n", K, S8_K_ALIGN);
        abort();
    }
    job_t tpl = {
        .N = N, .K = K, .alpha = alpha,
        .A = A, .lda = lda,
        .B = NULL, .B_s8 = B, .B_scale = scale, .ldb = ldb,
        .trans_b = true,   // int8 B is always stored as N x K row-major
        .C = C, .ldc = ldc,
    };
    dispatch(M, &tpl);
}

// ---------------------------------------------------------------------------
// Pre-packed int8 B: pack once at model_load, reuse every forward.
// ---------------------------------------------------------------------------
struct sgemm_s8_prepack {
    int N, K;
    int num_kb, num_jb;
    int8_t **blocks;   // num_kb × num_jb pointers; each block is kc × nc int8
};

sgemm_s8_prepack_t *sgemm_s8_prepack(int N, int K, const int8_t *B, int ldb) {
    // Pre-packing is only useful when the runtime actually consumes the int8
    // block layout. Hosts without SDOT/SMMLA take the dequant-to-fp32 fallback
    // in run_job — they re-pack fp32 per forward, so returning NULL here routes
    // linear() through sgemm_s8rm (non-prepacked) and skips wasted work.
#if !HAS_SMMLA && !HAS_SDOT
    (void)B; (void)ldb;
    return NULL;
#else
    if (K % S8_K_ALIGN != 0) return NULL;

    sgemm_s8_prepack_t *p = (sgemm_s8_prepack_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N; p->K = K;
    p->num_kb = (K + KC - 1) / KC;
    p->num_jb = (N + NC - 1) / NC;
    int nblocks = p->num_kb * p->num_jb;
    p->blocks = (int8_t **)calloc(nblocks, sizeof(int8_t *));
    if (!p->blocks) { free(p); return NULL; }

    // pack_B_s8_repack writes scales into a scratch buffer we discard — at
    // runtime, scaleB is passed directly as `scale + j_outer` (no copy).
    float scaleB_scratch[NC + NR];
    // Also need a dummy scale array sized ≥ N because pack_B reads from it;
    // the resulting scratch values are ignored.
    float *dummy_scale = (float *)calloc((size_t)N + NR, sizeof(float));
    if (!dummy_scale) { sgemm_s8_prepack_free(p); return NULL; }

    for (int kb = 0; kb < p->num_kb; kb++) {
        int k_outer = kb * KC;
        int kc = (K - k_outer < KC) ? (K - k_outer) : KC;
        for (int jb = 0; jb < p->num_jb; jb++) {
            int j_outer = jb * NC;
            int nc = (N - j_outer < NC) ? (N - j_outer) : NC;
            // pack_B_s8_repack always writes full NR-column tiles (zero-padding
            // the tail when nc % NR != 0), so size the buffer to the padded
            // column count — not the ragged one, or the tail block overruns.
            int nc_padded = ((nc + NR - 1) / NR) * NR;
            size_t bytes = (size_t)kc * nc_padded;
            int8_t *buf = (int8_t *)aligned_alloc(PACK_ALIGN, ((bytes + PACK_ALIGN - 1) / PACK_ALIGN) * PACK_ALIGN);
            if (!buf) { free(dummy_scale); sgemm_s8_prepack_free(p); return NULL; }
            pack_B_s8_repack(B, ldb, dummy_scale, k_outer, j_outer, kc, nc,
                             buf, scaleB_scratch);
            p->blocks[kb * p->num_jb + jb] = buf;
        }
    }
    free(dummy_scale);
    return p;
#endif  // HAS_SMMLA || HAS_SDOT
}

void sgemm_s8_prepack_free(sgemm_s8_prepack_t *p) {
    if (!p) return;
    if (p->blocks) {
        int n = p->num_kb * p->num_jb;
        for (int i = 0; i < n; i++) free(p->blocks[i]);
        free(p->blocks);
    }
    free(p);
}

void sgemm_s8rm_prepacked(int M, int N, int K,
                          float alpha,
                          const float *A, int lda,
                          const sgemm_s8_prepack_t *Bp, const float *scale,
                          float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0 || !Bp) return;
    if (Bp->N != N || Bp->K != K) return;  // shape mismatch
    job_t tpl = {
        .N = N, .K = K, .alpha = alpha,
        .A = A, .lda = lda,
        .B = NULL, .B_s8 = NULL, .B_scale = scale, .ldb = K,
        .B_blocks = Bp->blocks, .num_kb = Bp->num_kb, .num_jb = Bp->num_jb,
        .trans_b = true,
        .C = C, .ldc = ldc,
    };
    dispatch(M, &tpl);
}
