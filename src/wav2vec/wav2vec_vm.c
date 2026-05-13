#include "wav2vec_vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>

#include "sgemm.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


// --- Debug: dump intermediate activations when WAV2VEC_DUMP=<dir> is set ---
static const char *g_dump_dir = NULL;
static void dump_bin(const char *rel, const float *data, size_t n) {
    if (!g_dump_dir) return;
    char path[1024]; snprintf(path, sizeof path, "%s/%s", g_dump_dir, rel);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

const int CONV_KERNELS[NUM_CONV_LAYERS] = {10, 3, 3, 3, 3, 2, 2};
const int CONV_STRIDES[NUM_CONV_LAYERS] = { 5, 2, 2, 2, 2, 2, 2};

// --------------------------------------------------------------------------
// Forward scratch: one owned by each caller thread, reused across calls.
// Sizes are derived from INPUT_SAMPLES (fixed 32000) → T=99 after the feature
// extractor, so all scratch shapes are compile-time constants.
// --------------------------------------------------------------------------
// T is computed statically from CONV_KERNELS/STRIDES applied to 32000 samples.
// (32000-10)/5+1=6399 → /2→3199 → /2→1599 → /2→799 → /2→399 → /2→199 → /2→99
#define T_OUT 99

struct forward_scratch {
    float *conv_a;        // feature extractor ping-pong buffer A
    float *conv_b;        // feature extractor ping-pong buffer B
    float *conv_col;      // im2col buffer shared by all conv1d calls
    float *x;             // main (T, HS) activation
    float *pos_scratch;   // positional conv scratch
    float *tmp_T_H;       // encoder tmp (T, HS)
    float *tmp_T_F;       // encoder tmp (T, FFN)
    float *attn;          // Q/K/V/S/C packed
};

#define CONV_SCRATCH_ELEMS (6500 * CONV_CHANNELS)

static size_t conv_col_max_elems(void) {
    size_t max = 0;
    int in_t = INPUT_SAMPLES;
    int in_c = 1;
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        int k = CONV_KERNELS[i], s = CONV_STRIDES[i];
        int out_t = (in_t - k) / s + 1;
        size_t col = (size_t)(in_c * k) * (size_t)out_t;
        if (col > max) max = col;
        in_c = CONV_CHANNELS;
        in_t = out_t;
    }
    // Positional conv1d: groups=16, K = (HS/groups)*kernel, out_t ≈ T_OUT+1
    size_t pos_col = (size_t)(HIDDEN_SIZE / POS_GROUPS) * POS_KERNEL * (size_t)(T_OUT + 2);
    if (pos_col > max) max = pos_col;
    return max;
}

static float *aligned_floats(size_t n) {
    size_t bytes = ((n * sizeof(float) + 63) / 64) * 64;
    float *p = (float *)aligned_alloc(64, bytes);
    if (!p) { perror("aligned_alloc(scratch)"); exit(1); }
    return p;
}

forward_scratch_t *forward_scratch_new(void) {
    forward_scratch_t *s = (forward_scratch_t *)calloc(1, sizeof *s);
    if (!s) { perror("calloc(scratch)"); exit(1); }
    s->conv_a      = aligned_floats(CONV_SCRATCH_ELEMS);
    s->conv_b      = aligned_floats(CONV_SCRATCH_ELEMS);
    s->conv_col    = aligned_floats(conv_col_max_elems());
    s->x           = aligned_floats((size_t)T_OUT * HIDDEN_SIZE);
    s->pos_scratch = aligned_floats(2 * (size_t)HIDDEN_SIZE * (T_OUT + 16));
    s->tmp_T_H     = aligned_floats((size_t)T_OUT * HIDDEN_SIZE);
    s->tmp_T_F     = aligned_floats((size_t)T_OUT * FFN_SIZE);
    s->attn        = aligned_floats(3 * (size_t)T_OUT * HIDDEN_SIZE
                                    + (size_t)NUM_HEADS * T_OUT * T_OUT
                                    + (size_t)T_OUT * HIDDEN_SIZE);
    return s;
}

void forward_scratch_free(forward_scratch_t *s) {
    if (!s) return;
    free(s->conv_a); free(s->conv_b); free(s->conv_col);
    free(s->x); free(s->pos_scratch);
    free(s->tmp_T_H); free(s->tmp_T_F); free(s->attn);
    free(s);
}

// --------------------------------------------------------------------------
// Weight loading
// --------------------------------------------------------------------------

// Load a file into a heap-allocated buffer; returns NULL on missing.
static void *slurp(const char *path, long *nbytes_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long nbytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = aligned_alloc(64, ((nbytes + 63) / 64) * 64);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, nbytes, f) != (size_t)nbytes) { fclose(f); free(p); return NULL; }
    fclose(f);
    if (nbytes_out) *nbytes_out = nbytes;
    return p;
}

// Load a dense fp32 blob. Errors if the file is missing — callers that expect
// a possibly-quantized weight should use load_qw() instead.
static float *load_blob(const char *dir, const char *rel, size_t *count_out) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, rel);
    long nbytes = 0;
    float *p = (float *)slurp(path, &nbytes);
    if (!p) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (nbytes <= 0 || nbytes % 4 != 0) {
        fprintf(stderr, "bad size %ld for %s\n", nbytes, path); exit(1);
    }
    if (count_out) *count_out = (size_t)nbytes / 4;
    return p;
}

// Load a potentially-quantized weight. Prefers <rel> (fp32) if present;
// otherwise expects <stem>.q8.bin + <stem>.scale.bin (produced by
// extract_weights.py --int8). `rows` and `cols` describe the PyTorch shape.
// Populates *qw with either f32 xor q8+scale. Appends raw allocations into
// m->owned so they get freed via model_free.
static void load_qw(model_t *m, const char *dir, const char *rel,
                    int rows, int cols, qw_t *qw) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, rel);
    long nbytes = 0;
    float *fp = (float *)slurp(path, &nbytes);
    if (fp) {
        if ((long)rows * cols * 4 != nbytes) {
            fprintf(stderr, "%s: expected %d*%d*4=%ld bytes, got %ld\n",
                    path, rows, cols, (long)rows * cols * 4, nbytes);
            exit(1);
        }
        qw->f32 = fp; qw->q8 = NULL; qw->scale = NULL;
        qw->rows = rows; qw->cols = cols;
        m->owned[m->num_owned++] = fp;
        return;
    }
    // Int8 fallback
    size_t rlen = strlen(rel);
    if (!(rlen > 4 && strcmp(rel + rlen - 4, ".bin") == 0)) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    char stem[512];
    size_t slen = rlen - 4;
    if (slen >= sizeof stem) { fprintf(stderr, "path too long\n"); exit(1); }
    memcpy(stem, rel, slen); stem[slen] = 0;
    char q8p[1024], scp[1024];
    snprintf(q8p, sizeof q8p, "%s/%s.q8.bin",    dir, stem);
    snprintf(scp, sizeof scp, "%s/%s.scale.bin", dir, stem);
    long qb = 0, sb = 0;
    int8_t *q8 = (int8_t *)slurp(q8p, &qb);
    float  *sc = (float  *)slurp(scp, &sb);
    if (!q8 || !sc) {
        fprintf(stderr, "open %s (and fallback %s.{q8,scale}.bin): %s\n",
                path, stem, strerror(errno));
        exit(1);
    }
    if (qb != (long)rows * cols) {
        fprintf(stderr, "%s: expected %d*%d=%ld int8 bytes, got %ld\n",
                q8p, rows, cols, (long)rows * cols, qb);
        exit(1);
    }
    if (sb != (long)rows * 4) {
        fprintf(stderr, "%s: expected %d float scales, got %ld bytes\n",
                scp, rows, sb);
        exit(1);
    }
    qw->f32 = NULL; qw->q8 = q8; qw->scale = sc;
    qw->rows = rows; qw->cols = cols;
    // Pre-pack into the sgemm micro-kernel layout so forward() skips pack_B.
    // Returns NULL (silently) if K isn't S8_K_ALIGN-aligned; linear() then
    // falls back to sgemm_s8rm which re-packs every call.
    qw->prepack = sgemm_s8_prepack(rows, cols, q8, cols);
    m->owned[m->num_owned++] = q8;
    m->owned[m->num_owned++] = sc;
}

#define LOAD(dst, rel)  do { (dst) = load_blob(weights_dir, (rel), NULL); m->owned[m->num_owned++] = (dst); } while (0)

model_t *model_load(const char *weights_dir) {
    model_t *m = (model_t *)calloc(1, sizeof(model_t));
    // Worst case (all quantizable weights are int8, each contributing q8+scale):
    //   7*4 (fe) + 3 fp32 + 2 (fp_proj q8+scale) + 2 (pos)
    //   + 2 (enc_norm) + 24*(4 ln + 6*(q8+scale) + 6 bias)
    //   + (classifier proj q8+scale + bias) + (out q8+scale + bias) ≈ 580.
    m->owned = (void **)calloc(1024, sizeof(void *));

    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        char rel[128];
        int in_c = (i == 0) ? 1 : CONV_CHANNELS;
        int cols = in_c * CONV_KERNELS[i];
        snprintf(rel, sizeof rel, "feature_extractor/conv%d.weight.bin", i);
        load_qw(m, weights_dir, rel, CONV_CHANNELS, cols, &m->fe_conv_w[i]);
        snprintf(rel, sizeof rel, "feature_extractor/conv%d.bias.bin", i);        LOAD(m->fe_conv_b[i], rel);
        snprintf(rel, sizeof rel, "feature_extractor/conv%d.norm_weight.bin", i); LOAD(m->fe_norm_w[i], rel);
        snprintf(rel, sizeof rel, "feature_extractor/conv%d.norm_bias.bin", i);   LOAD(m->fe_norm_b[i], rel);
    }

    LOAD(m->fp_norm_w, "feature_projection/norm_weight.bin");
    LOAD(m->fp_norm_b, "feature_projection/norm_bias.bin");
    load_qw(m, weights_dir, "feature_projection/proj.weight.bin",
            HIDDEN_SIZE, CONV_CHANNELS, &m->fp_proj_w);
    LOAD(m->fp_proj_b, "feature_projection/proj.bias.bin");

    LOAD(m->pos_w, "pos_conv/weight.bin");
    LOAD(m->pos_b, "pos_conv/bias.bin");

    LOAD(m->enc_norm_w, "encoder_norm/weight.bin");
    LOAD(m->enc_norm_b, "encoder_norm/bias.bin");

    for (int i = 0; i < NUM_ENCODER_LAYERS; i++) {
        char rel[128], dir[64];
        snprintf(dir, sizeof dir, "encoder/layer_%d", i);
        #define LL_F(field, name) do { \
            snprintf(rel, sizeof rel, "%s/%s", dir, name); \
            LOAD(m->layers[i].field, rel); \
        } while (0)
        #define LL_Q(field, name, R, C) do { \
            snprintf(rel, sizeof rel, "%s/%s", dir, name); \
            load_qw(m, weights_dir, rel, (R), (C), &m->layers[i].field); \
        } while (0)
        LL_F(ln1_w, "ln1.weight.bin");   LL_F(ln1_b, "ln1.bias.bin");
        LL_Q(q_w,   "attn_q.weight.bin",   HIDDEN_SIZE, HIDDEN_SIZE); LL_F(q_b, "attn_q.bias.bin");
        LL_Q(k_w,   "attn_k.weight.bin",   HIDDEN_SIZE, HIDDEN_SIZE); LL_F(k_b, "attn_k.bias.bin");
        LL_Q(v_w,   "attn_v.weight.bin",   HIDDEN_SIZE, HIDDEN_SIZE); LL_F(v_b, "attn_v.bias.bin");
        LL_Q(out_w, "attn_out.weight.bin", HIDDEN_SIZE, HIDDEN_SIZE); LL_F(out_b, "attn_out.bias.bin");
        LL_F(ln2_w, "ln2.weight.bin");   LL_F(ln2_b, "ln2.bias.bin");
        LL_Q(ffn_in_w,  "ffn_in.weight.bin",  FFN_SIZE,    HIDDEN_SIZE); LL_F(ffn_in_b,  "ffn_in.bias.bin");
        LL_Q(ffn_out_w, "ffn_out.weight.bin", HIDDEN_SIZE, FFN_SIZE);    LL_F(ffn_out_b, "ffn_out.bias.bin");
        #undef LL_F
        #undef LL_Q
    }

    load_qw(m, weights_dir, "classifier/projector.weight.bin",
            CLASSIFIER_PROJ, HIDDEN_SIZE, &m->proj_w);
    LOAD(m->proj_b, "classifier/projector.bias.bin");
    load_qw(m, weights_dir, "classifier/out.weight.bin",
            NUM_LABELS, CLASSIFIER_PROJ, &m->out_w);
    LOAD(m->out_b, "classifier/out.bias.bin");

    return m;
}

void model_free(model_t *m) {
    if (!m) return;
    // Free pre-pack handles (one per quantized weight); qw_t pointers are
    // embedded in model_t, so walk the fields directly.
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        sgemm_s8_prepack_free(m->fe_conv_w[i].prepack);
    }
    sgemm_s8_prepack_free(m->fp_proj_w.prepack);
    sgemm_s8_prepack_free(m->proj_w.prepack);
    sgemm_s8_prepack_free(m->out_w.prepack);
    for (int i = 0; i < NUM_ENCODER_LAYERS; i++) {
        sgemm_s8_prepack_free(m->layers[i].q_w.prepack);
        sgemm_s8_prepack_free(m->layers[i].k_w.prepack);
        sgemm_s8_prepack_free(m->layers[i].v_w.prepack);
        sgemm_s8_prepack_free(m->layers[i].out_w.prepack);
        sgemm_s8_prepack_free(m->layers[i].ffn_in_w.prepack);
        sgemm_s8_prepack_free(m->layers[i].ffn_out_w.prepack);
    }
    for (size_t i = 0; i < m->num_owned; i++) free(m->owned[i]);
    free(m->owned);
    free(m);
}

// --------------------------------------------------------------------------
// Core ops
// --------------------------------------------------------------------------

// y[M, N] = x[M, K] @ W^T + b    (W is [N, K] row-major, PyTorch convention)
void linear_f32(const float *x, const float *W, const float *b, int M, int K, int N, float *y) {
    sgemm_rm(/*trans_b=*/true, M, N, K, 1.0f, x, K, W, K, y, N);
    if (b) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) y[i*N + j] += b[j];
        }
    }
}

// Dispatches to fp32, int8, or pre-packed int8 GEMM based on qw_t fields.
void linear(const float *x, const qw_t *W, const float *b, int M, int K, int N, float *y) {
    if (W->f32) {
        sgemm_rm(/*trans_b=*/true, M, N, K, 1.0f, x, K, W->f32, K, y, N);
    } else if (W->prepack) {
        sgemm_s8rm_prepacked(M, N, K, 1.0f, x, K, W->prepack, W->scale, y, N);
    } else {
        sgemm_s8rm(M, N, K, 1.0f, x, K, W->q8, W->scale, K, y, N);
    }
    if (b) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) y[i*N + j] += b[j];
        }
    }
}

void layer_norm(float *x, int N, int C, const float *gamma, const float *beta, float eps) {
#if defined(__ARM_NEON)
    for (int i = 0; i < N; i++) {
        float *row = x + i * C;
        // Mean + mean-of-squares in one pass, each with 4 parallel accumulators
        // so the compiler can issue independent fmla chains.
        float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
        float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
        float32x4_t q0 = vdupq_n_f32(0), q1 = vdupq_n_f32(0);
        float32x4_t q2 = vdupq_n_f32(0), q3 = vdupq_n_f32(0);
        int j = 0;
        for (; j + 16 <= C; j += 16) {
            float32x4_t v0 = vld1q_f32(row + j + 0);
            float32x4_t v1 = vld1q_f32(row + j + 4);
            float32x4_t v2 = vld1q_f32(row + j + 8);
            float32x4_t v3 = vld1q_f32(row + j + 12);
            s0 = vaddq_f32(s0, v0); s1 = vaddq_f32(s1, v1);
            s2 = vaddq_f32(s2, v2); s3 = vaddq_f32(s3, v3);
            q0 = vfmaq_f32(q0, v0, v0); q1 = vfmaq_f32(q1, v1, v1);
            q2 = vfmaq_f32(q2, v2, v2); q3 = vfmaq_f32(q3, v3, v3);
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
        float sq  = vaddvq_f32(vaddq_f32(vaddq_f32(q0, q1), vaddq_f32(q2, q3)));
        for (; j < C; j++) { float v = row[j]; sum += v; sq += v * v; }
        float inv_C = 1.0f / (float)C;
        float mean  = sum * inv_C;
        float var   = sq * inv_C - mean * mean;
        float inv   = 1.0f / sqrtf(var + eps);
        // Second pass: (x - mean) * inv * gamma + beta, all fused.
        float32x4_t vmean = vdupq_n_f32(mean);
        float32x4_t vinv  = vdupq_n_f32(inv);
        j = 0;
        for (; j + 4 <= C; j += 4) {
            float32x4_t v = vld1q_f32(row + j);
            float32x4_t g = vld1q_f32(gamma + j);
            float32x4_t b = vld1q_f32(beta  + j);
            v = vsubq_f32(v, vmean);
            v = vmulq_f32(v, vinv);
            v = vfmaq_f32(b, v, g);
            vst1q_f32(row + j, v);
        }
        for (; j < C; j++) row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
    }
#else
    for (int i = 0; i < N; i++) {
        float *row = x + i * C;
        float mean = 0.0f;
        for (int j = 0; j < C; j++) mean += row[j];
        mean /= (float)C;
        float var = 0.0f;
        for (int j = 0; j < C; j++) { float d = row[j] - mean; var += d * d; }
        var /= (float)C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < C; j++) row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
    }
#endif
}

// tanh approx of GELU (matches transformers' "gelu" default for wav2vec2 -> no, wav2vec2 uses exact gelu)
// wav2vec2 uses `hidden_act=gelu` which is the exact formulation: x * 0.5 * (1 + erf(x / sqrt(2)))
void gelu(float *x, int n) {
    static const float INV_SQRT2 = 0.70710678118654752440f;
    for (int i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * INV_SQRT2));
}

void softmax_rows(float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float *row = x + i * cols;
        float mx = row[0];
        for (int j = 1; j < cols; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < cols; j++) row[j] *= inv;
    }
}

// --------------------------------------------------------------------------
// Conv1D. Layout depends on `groups`:
//
//  • groups == 1 (feature extractor): input (in_t × in_c) time-major, output
//    (out_t × out_c). This matches sgemm_s8rm's natural A-operand layout and
//    avoids the col/output transposes that channels-first would require. Both
//    fp32 (W->f32) and int8 (W->prepack) weights work in this layout.
//  • groups > 1 (pos_conv): input (in_c × in_t) channels-first, output
//    (out_c × out_t). Quantized weights are not supported for grouped convs.
//
// Weight reshapes as (out_c, K=in_c/groups * kernel) for both paths.
// --------------------------------------------------------------------------
static void conv1d(const float *x, int in_c, int in_t,
                   const qw_t *W, const float *bias,
                   int out_c, int kernel, int stride, int padding, int groups,
                   float *y, int *out_t_ptr, float *col) {
    int out_t = (in_t + 2 * padding - kernel) / stride + 1;
    int ic_per_g = in_c  / groups;
    int oc_per_g = out_c / groups;
    int K = ic_per_g * kernel;

    if (groups == 1) {
        // Time-major im2col: col[t*K + ic*kernel + k] = x[t*stride+k-pad, ic].
        for (int t = 0; t < out_t; t++) {
            float *crow = col + t * K;
            for (int ic = 0; ic < in_c; ic++) {
                float *dst = crow + ic * kernel;
                for (int k = 0; k < kernel; k++) {
                    int t_in = t * stride + k - padding;
                    dst[k] = ((unsigned)t_in < (unsigned)in_t) ? x[t_in * in_c + ic] : 0.0f;
                }
            }
        }
        // y (out_t × out_c) = col (out_t × K) @ W^T (K × out_c).
        if (W->prepack) {
            sgemm_s8rm_prepacked(out_t, out_c, K, 1.0f,
                                 col, K,
                                 W->prepack, W->scale,
                                 y, out_c);
        } else if (W->q8) {
            // int8 weights but no prepack (host lacks SDOT/SMMLA): run_job
            // dequantizes B to fp32 at pack time and uses the fp32 kernel.
            sgemm_s8rm(out_t, out_c, K, 1.0f,
                       col, K,
                       W->q8, W->scale, K,
                       y, out_c);
        } else {
            sgemm_rm(/*trans_b=*/true,
                     out_t, out_c, K,
                     1.0f, col, K,
                           W->f32, K,
                     y, out_c);
        }
        if (bias) {
            for (int t = 0; t < out_t; t++) {
                float *row = y + t * out_c;
                for (int oc = 0; oc < out_c; oc++) row[oc] += bias[oc];
            }
        }
        if (out_t_ptr) *out_t_ptr = out_t;
        return;
    }

    // Grouped path (fp32 only): y (oc_per_g × out_t) per group, channels-first.
    const float *W_fp = W->f32;
    for (int g = 0; g < groups; g++) {
        const float *xg = x + g * ic_per_g * in_t;
        const float *Wg = W_fp + g * oc_per_g * K;
        float *yg = y + g * oc_per_g * out_t;

        for (int ic = 0; ic < ic_per_g; ic++) {
            const float *xrow = xg + ic * in_t;
            for (int k = 0; k < kernel; k++) {
                float *crow = col + (ic * kernel + k) * out_t;
                int t_in_base = k - padding;
                for (int t = 0; t < out_t; t++) {
                    int t_in = t_in_base + t * stride;
                    crow[t] = ((unsigned)t_in < (unsigned)in_t) ? xrow[t_in] : 0.0f;
                }
            }
        }
        sgemm_rm(/*trans_b=*/false,
                 oc_per_g, out_t, K,
                 1.0f, Wg, K,
                       col, out_t,
                 yg, out_t);

        if (bias) {
            for (int oc = 0; oc < oc_per_g; oc++) {
                float bi = bias[g * oc_per_g + oc];
                float *row = yg + oc * out_t;
                for (int t = 0; t < out_t; t++) row[t] += bi;
            }
        }
    }
    if (out_t_ptr) *out_t_ptr = out_t;
}

// --------------------------------------------------------------------------
// Feature extractor: 7 Conv1D layers, each followed by LayerNorm (over channels) + GELU
// Input: (1, T=32000) [mono audio]
// Output: (T'=99, C=512) (time-major after transpose)
// --------------------------------------------------------------------------
static int feature_extractor_fwd(const model_t *m, const float *audio,
                                 float *scratch_a, float *scratch_b,
                                 float *col_scratch) {
    int in_t = INPUT_SAMPLES;
    // HF Wav2Vec2FeatureExtractor: zero-mean unit-variance normalize (per-utterance).
    {
        double mean = 0.0;
        for (int i = 0; i < in_t; i++) mean += audio[i];
        mean /= (double)in_t;
        double var = 0.0;
        for (int i = 0; i < in_t; i++) { double d = audio[i] - mean; var += d * d; }
        var /= (double)in_t;
        float inv = 1.0f / sqrtf((float)var + 1e-7f);
        for (int i = 0; i < in_t; i++) scratch_a[i] = ((float)(audio[i] - mean)) * inv;
    }

    // All fe convs have groups=1, so conv1d produces (out_t × out_c) time-major
    // directly (fp32 or int8, same layout). LN+GELU stay in that layout too —
    // no transposes between layers.
    int in_c = 1, out_t;
    float *in = scratch_a, *out = scratch_b;
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        conv1d(in, in_c, in_t,
               &m->fe_conv_w[i], m->fe_conv_b[i],
               CONV_CHANNELS, CONV_KERNELS[i], CONV_STRIDES[i], /*padding*/0, /*groups*/1,
               out, &out_t, col_scratch);
        if (i == 0) dump_bin("00b_conv0_preLN.bin", out, (size_t)out_t * CONV_CHANNELS);
        layer_norm(out, out_t, CONV_CHANNELS, m->fe_norm_w[i], m->fe_norm_b[i], LN_EPS);
        gelu(out, out_t * CONV_CHANNELS);
        in_c = CONV_CHANNELS;
        in_t = out_t;
        float *swap = in; in = out; out = swap;
    }
    // `in` now holds (T=99, 512) time-major. Downstream contract: result in scratch_a.
    if (in != scratch_a) memcpy(scratch_a, in, (size_t)in_t * in_c * sizeof(float));
    return in_t;
}

// --------------------------------------------------------------------------
// Positional convolutional embedding (on (B=1, C=1024, T) then GELU, crop to T)
// Input / output: (T, 1024) time-major. Transposes internally.
// --------------------------------------------------------------------------
static void pos_conv_embed_fwd(const model_t *m, float *x_tc, int T,
                               float *scratch, float *col_scratch) {
    // Transpose x_tc (T, C) -> scratch_cft (C, T)
    float *xct = scratch;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < HIDDEN_SIZE; c++) xct[c * T + t] = x_tc[t * HIDDEN_SIZE + c];
    }
    float *yct = scratch + HIDDEN_SIZE * T;
    int out_t_full;
    // pos_conv weight is fp32 (grouped convs aren't on the int8 path).
    qw_t pos_w_qw = { .f32 = m->pos_w };
    conv1d(xct, HIDDEN_SIZE, T,
           &pos_w_qw, m->pos_b,
           HIDDEN_SIZE, POS_KERNEL, /*stride*/1, /*padding*/POS_KERNEL / 2, POS_GROUPS,
           yct, &out_t_full, col_scratch);
    // Crop one time step to match input length (even-kernel conv produces T+1 with pad=kernel/2)
    int T_out = T;
    // GELU in-place (only over valid region)
    for (int c = 0; c < HIDDEN_SIZE; c++) {
        for (int t = 0; t < T_out; t++) {
            float v = yct[c * out_t_full + t];
            v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
            yct[c * out_t_full + t] = v;
        }
    }
    // Add to input: x_tc = x_tc + transpose(yct)
    for (int t = 0; t < T_out; t++) {
        for (int c = 0; c < HIDDEN_SIZE; c++)
            x_tc[t * HIDDEN_SIZE + c] += yct[c * out_t_full + t];
    }
}

// --------------------------------------------------------------------------
// Multi-head self-attention (pre-LN, no masking: input is always fixed length 99)
//   x: (T, 1024)
//   Wq, Wk, Wv, Wo: (1024, 1024)  (biases are 1024)
//   output overwrites x.
// --------------------------------------------------------------------------
static void attention_fwd(const model_t *m, int li, float *x, int T, float *scratch) {
    const int H = NUM_HEADS, D = HEAD_DIM, HS = HIDDEN_SIZE;
    // Layout inside scratch:
    //   Q: [T*HS]  K: [T*HS]  V: [T*HS]  scores: [H*T*T]  context: [T*HS]
    float *Q = scratch;
    float *K = Q + T * HS;
    float *V = K + T * HS;
    float *S = V + T * HS;
    float *C = S + H * T * T;

    linear(x, &m->layers[li].q_w, m->layers[li].q_b, T, HS, HS, Q);
    linear(x, &m->layers[li].k_w, m->layers[li].k_b, T, HS, HS, K);
    linear(x, &m->layers[li].v_w, m->layers[li].v_b, T, HS, HS, V);

    float scale = 1.0f / sqrtf((float)D);

    for (int h = 0; h < H; h++) {
        float *sh = S + h * T * T;
        sgemm_rm(/*trans_b=*/true,
                 T, T, D,
                 scale, Q + h * D, HS,
                        K + h * D, HS,
                 sh, T);
    }
    softmax_rows(S, H * T, T);

    for (int h = 0; h < H; h++) {
        float *sh = S + h * T * T;
        sgemm_rm(/*trans_b=*/false,
                 T, D, T,
                 1.0f, sh, T,
                       V + h * D, HS,
                 C + h * D, HS);
    }
    // Output projection: x = C @ Wo^T + bo
    linear(C, &m->layers[li].out_w, m->layers[li].out_b, T, HS, HS, x);
}

// --------------------------------------------------------------------------
// Encoder layer (pre-LN): residual attn + residual FFN
// --------------------------------------------------------------------------
static void encoder_layer_fwd(const model_t *m, int li, float *x, int T,
                              float *tmp_T_H, float *tmp_T_F, float *attn_scratch) {
    const int HS = HIDDEN_SIZE;
    // Attention branch
    memcpy(tmp_T_H, x, T * HS * sizeof(float));
    layer_norm(tmp_T_H, T, HS, m->layers[li].ln1_w, m->layers[li].ln1_b, LN_EPS);
    attention_fwd(m, li, tmp_T_H, T, attn_scratch);
    for (int i = 0; i < T * HS; i++) x[i] += tmp_T_H[i];

    // FFN branch
    memcpy(tmp_T_H, x, T * HS * sizeof(float));
    layer_norm(tmp_T_H, T, HS, m->layers[li].ln2_w, m->layers[li].ln2_b, LN_EPS);
    // intermediate: (T, HS) @ Win^T (FFN, HS) + bin  -> (T, FFN)
    linear(tmp_T_H, &m->layers[li].ffn_in_w, m->layers[li].ffn_in_b, T, HS, FFN_SIZE, tmp_T_F);
    gelu(tmp_T_F, T * FFN_SIZE);
    // output: (T, FFN) @ Wout^T (HS, FFN) + bout -> (T, HS)
    linear(tmp_T_F, &m->layers[li].ffn_out_w, m->layers[li].ffn_out_b, T, FFN_SIZE, HS, tmp_T_H);
    for (int i = 0; i < T * HS; i++) x[i] += tmp_T_H[i];
}

// --------------------------------------------------------------------------
// Full forward
// --------------------------------------------------------------------------
static double timing_ms(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

void model_forward(const model_t *m, forward_scratch_t *s, const float *audio,
                   float logits_out[NUM_LABELS], float probs_out[NUM_LABELS]) {
    g_dump_dir = getenv("WAV2VEC_DUMP");
    int do_prof = getenv("WAV2VEC_PROF") != NULL;
    struct timespec t0, t1;
    if (do_prof) clock_gettime(CLOCK_MONOTONIC, &t0);

    float *sa = s->conv_a;
    float *sb = s->conv_b;

    int T = feature_extractor_fwd(m, audio, sa, sb, s->conv_col);
    // sa now holds (T, CONV_CHANNELS=512) time-major
    dump_bin("01_fe_out.bin", sa, (size_t)T * CONV_CHANNELS);
    if (do_prof) { clock_gettime(CLOCK_MONOTONIC, &t1); fprintf(stderr, "  fe:       %7.1f ms\n", timing_ms(t0, t1)); t0 = t1; }

    // Feature projection: LayerNorm over 512, then Linear (512 -> 1024)
    // sa: (T, 512). Reuse.
    layer_norm(sa, T, CONV_CHANNELS, m->fp_norm_w, m->fp_norm_b, LN_EPS);
    dump_bin("02_fp_norm_out.bin", sa, (size_t)T * CONV_CHANNELS);
    float *x = s->x;
    linear(sa, &m->fp_proj_w, m->fp_proj_b, T, CONV_CHANNELS, HIDDEN_SIZE, x);
    dump_bin("03_fp_proj_out.bin", x, (size_t)T * HIDDEN_SIZE);
    // x: (T, 1024)

    // Positional conv embed (residual into x)
    pos_conv_embed_fwd(m, x, T, s->pos_scratch, s->conv_col);
    dump_bin("05_after_pos_add.bin", x, (size_t)T * HIDDEN_SIZE);

    // 24 encoder layers
    for (int i = 0; i < NUM_ENCODER_LAYERS; i++) {
        encoder_layer_fwd(m, i, x, T, s->tmp_T_H, s->tmp_T_F, s->attn);
        if (i == 0 || i == 23) {
            char rel[64]; snprintf(rel, sizeof rel, "06_layer_%d_out.bin", i);
            dump_bin(rel, x, (size_t)T * HIDDEN_SIZE);
        }
    }
    if (do_prof) { clock_gettime(CLOCK_MONOTONIC, &t1); fprintf(stderr, "  24 layers:%7.1f ms\n", timing_ms(t0, t1)); t0 = t1; }

    // Final post-encoder LayerNorm
    layer_norm(x, T, HIDDEN_SIZE, m->enc_norm_w, m->enc_norm_b, LN_EPS);
    dump_bin("07_after_enc_ln.bin", x, (size_t)T * HIDDEN_SIZE);

    // Mean pool over time: (T, 1024) -> (1024)
    float pooled[HIDDEN_SIZE] = {0};
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < HIDDEN_SIZE; c++) pooled[c] += x[t * HIDDEN_SIZE + c];
    }
    float inv_T = 1.0f / (float)T;
    for (int c = 0; c < HIDDEN_SIZE; c++) pooled[c] *= inv_T;
    dump_bin("08_pooled.bin", pooled, HIDDEN_SIZE);

    // Classifier: projector (1024 -> 256) then out (256 -> 2)
    float proj[CLASSIFIER_PROJ];
    linear(pooled, &m->proj_w, m->proj_b, 1, HIDDEN_SIZE, CLASSIFIER_PROJ, proj);
    dump_bin("09_projector_out.bin", proj, CLASSIFIER_PROJ);
    linear(proj,  &m->out_w,  m->out_b,  1, CLASSIFIER_PROJ, NUM_LABELS, logits_out);
    dump_bin("10_logits.bin", logits_out, NUM_LABELS);

    // Softmax for probs
    float mx = logits_out[0] > logits_out[1] ? logits_out[0] : logits_out[1];
    float e0 = expf(logits_out[0] - mx), e1 = expf(logits_out[1] - mx);
    float sum = e0 + e1;
    probs_out[0] = e0 / sum;
    probs_out[1] = e1 / sum;
}
