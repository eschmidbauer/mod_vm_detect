/*
 * fireredvad - Pure C streaming Voice Activity Detection + Audio Event
 *              Detection with hand-rolled DFSMN inference.
 *
 * Zero external dependencies beyond libc and libm.
 * Weights are loaded from a binary file produced by extract_weights.py.
 *
 * Architecture (both VAD and AED):
 *   Input projection:  feat[T,80] -> MatMul[80,256]+bias -> ReLU -> MatMul[256,128]+bias -> ReLU
 *   8 FSMN blocks:     each with depthwise conv1d lookback filter (kernel=20)
 *                       + FC expansion [128,256]+bias -> ReLU -> FC projection [256,128]
 *                       + residual connection (blocks 1-7)
 *   Output projection: MatMul[128,256]+bias -> ReLU -> MatMul[256,out]+bias -> Sigmoid
 *
 * VAD: streaming with caches (lookback only), output dim=1
 * AED: non-streaming with lookback+lookahead, output dim=3
 */

#include "fireredvad.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#elif defined(__SSE__)
#include <immintrin.h>
#define USE_SSE 1
#endif

/* ── Helpers ────────────────────────────────────────────────────────── */

static inline int max_i(int a, int b) { return a > b ? a : b; }
static inline float max_f(float a, float b) { return a > b ? a : b; }

/* ── Mel filterbank ─────────────────────────────────────────────────── */

#define LOW_FREQ 20.0f
#define HIGH_FREQ 8000.0f
#define PRE_EMPHASIS 0.97f

static float hertz_to_mel(float f) { return 1127.0f * logf(1.0f + f / 700.0f); }
static float mel_to_hertz(float m) { return 700.0f * (expf(m / 1127.0f) - 1.0f); }

static float g_window[FRAME_LENGTH];
static float g_mel_fb[NUM_MEL_BINS][NUM_FFT_BINS];

static void init_window(void)
{
    for (int i = 0; i < FRAME_LENGTH; i++)
    {
        float hamming = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (FRAME_LENGTH - 1));
        g_window[i] = powf(hamming, 0.85f);
    }
}

static void init_mel_filterbank(void)
{
    float mel_low = hertz_to_mel(LOW_FREQ);
    float mel_high = hertz_to_mel(HIGH_FREQ);
    float mel_points[NUM_MEL_BINS + 2];
    float bin_points[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++)
    {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (NUM_MEL_BINS + 1);
        bin_points[i] = mel_to_hertz(mel_points[i]) * FFT_SIZE / (float)SAMPLE_RATE;
    }
    memset(g_mel_fb, 0, sizeof(g_mel_fb));
    for (int m = 0; m < NUM_MEL_BINS; m++)
    {
        float left = bin_points[m], center = bin_points[m + 1], right = bin_points[m + 2];
        for (int k = 0; k < NUM_FFT_BINS; k++)
        {
            if (k >= left && k <= center && center > left)
                g_mel_fb[m][k] = (k - left) / (center - left);
            else if (k > center && k <= right && right > center)
                g_mel_fb[m][k] = (right - k) / (right - center);
        }
    }
}

void fireredvad_init(void)
{
    init_window();
    init_mel_filterbank();
}

/* ── FFT ────────────────────────────────────────────────────────────── */

static void fft(float *re, float *im, int N)
{
    int j = 0;
    for (int i = 0; i < N - 1; i++)
    {
        if (i < j)
        {
            float t;
            t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
        int m = N >> 1;
        while (m >= 1 && j >= m)
        {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    for (int step = 1; step < N; step <<= 1)
    {
        float angle = -(float)M_PI / step;
        float wRe = cosf(angle), wIm = sinf(angle);
        for (int group = 0; group < N; group += step << 1)
        {
            float curRe = 1.0f, curIm = 0.0f;
            for (int pair = 0; pair < step; pair++)
            {
                int a = group + pair, b = a + step;
                float tRe = curRe * re[b] - curIm * im[b];
                float tIm = curRe * im[b] + curIm * re[b];
                re[b] = re[a] - tRe;
                im[b] = im[a] - tIm;
                re[a] += tRe;
                im[a] += tIm;
                float nr = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = nr;
            }
        }
    }
}

static void power_spectrum(const float *frame, float *power)
{
    float re[FFT_SIZE], im[FFT_SIZE];
    memset(re, 0, sizeof(re));
    memset(im, 0, sizeof(im));
    memcpy(re, frame, FRAME_LENGTH * sizeof(float));
    fft(re, im, FFT_SIZE);
    for (int k = 0; k < NUM_FFT_BINS; k++)
        power[k] = re[k] * re[k] + im[k] * im[k];
}

/* ── Fbank extraction ───────────────────────────────────────────────── */

#define FBANK_STACK_MAX 8192 /* Stack buffer for combined remainder+pcm (covers typical 20ms streaming) */

int fireredvad_extract_fbank(const int16_t *pcm, int pcm_len,
                             int16_t *remainder, int *rem_len,
                             float *out, int max_frames)
{
    int total = *rem_len + pcm_len;
    int16_t stack_buf[FBANK_STACK_MAX];
    int16_t *combined = (total <= FBANK_STACK_MAX) ? stack_buf : (int16_t *)malloc(total * sizeof(int16_t));
    if (!combined)
        return 0;
    memcpy(combined, remainder, *rem_len * sizeof(int16_t));
    memcpy(combined + *rem_len, pcm, pcm_len * sizeof(int16_t));

    int num_frames = 0;
    if (total >= FRAME_LENGTH)
        num_frames = (total - FRAME_LENGTH) / FRAME_SHIFT + 1;
    if (num_frames > max_frames)
        num_frames = max_frames;

    float windowed[FRAME_LENGTH], power[NUM_FFT_BINS];
    for (int i = 0; i < num_frames; i++)
    {
        int start = i * FRAME_SHIFT;
        float prev = (start > 0) ? (float)combined[start - 1] : (float)combined[start];
        windowed[0] = ((float)combined[start] - PRE_EMPHASIS * prev) * g_window[0];
        for (int j = 1; j < FRAME_LENGTH; j++)
            windowed[j] = ((float)combined[start + j] - PRE_EMPHASIS * (float)combined[start + j - 1]) * g_window[j];
        power_spectrum(windowed, power);
        float *feat = out + i * NUM_MEL_BINS;
        for (int m = 0; m < NUM_MEL_BINS; m++)
        {
            float sum = 0.0f;
            for (int k = 0; k < NUM_FFT_BINS; k++)
                sum += g_mel_fb[m][k] * power[k];
            feat[m] = logf(max_f(sum, 1e-10f));
        }
    }

    int consumed = num_frames * FRAME_SHIFT;
    *rem_len = total - consumed;
    if (*rem_len > 0)
        memmove(remainder, combined + consumed, *rem_len * sizeof(int16_t));
    if (combined != stack_buf)
        free(combined);
    return num_frames;
}

float *fireredvad_extract_fbank_segment(const int16_t *pcm, int pcm_len, int *out_nf)
{
    int num_frames = 0;
    if (pcm_len >= FRAME_LENGTH)
        num_frames = (pcm_len - FRAME_LENGTH) / FRAME_SHIFT + 1;
    if (num_frames == 0)
    {
        *out_nf = 0;
        return NULL;
    }

    float *feat = (float *)malloc(num_frames * NUM_MEL_BINS * sizeof(float));
    float windowed[FRAME_LENGTH], power[NUM_FFT_BINS];
    for (int i = 0; i < num_frames; i++)
    {
        int start = i * FRAME_SHIFT;
        float prev = (start > 0) ? (float)pcm[start - 1] : (float)pcm[start];
        windowed[0] = ((float)pcm[start] - PRE_EMPHASIS * prev) * g_window[0];
        for (int j = 1; j < FRAME_LENGTH; j++)
            windowed[j] = ((float)pcm[start + j] - PRE_EMPHASIS * (float)pcm[start + j - 1]) * g_window[j];
        power_spectrum(windowed, power);
        float *f = feat + i * NUM_MEL_BINS;
        for (int m = 0; m < NUM_MEL_BINS; m++)
        {
            float sum = 0.0f;
            for (int k = 0; k < NUM_FFT_BINS; k++)
                sum += g_mel_fb[m][k] * power[k];
            f[m] = logf(max_f(sum, 1e-10f));
        }
    }
    *out_nf = num_frames;
    return feat;
}

/* ── CMVN ───────────────────────────────────────────────────────────── */

int fireredvad_load_cmvn(const char *path, Cmvn *cmvn)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Cannot open CMVN: %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    char *p = strstr(buf, "\"means\"");
    if (!p)
    {
        free(buf);
        return -1;
    }
    p = strchr(p, '[') + 1;
    for (int i = 0; i < NUM_MEL_BINS; i++)
    {
        cmvn->means[i] = strtof(p, &p);
        if (*p == ',')
            p++;
    }

    p = strstr(p, "\"inv_std\"");
    if (!p)
    {
        free(buf);
        return -1;
    }
    p = strchr(p, '[') + 1;
    for (int i = 0; i < NUM_MEL_BINS; i++)
    {
        cmvn->inv_std[i] = strtof(p, &p);
        if (*p == ',')
            p++;
    }

    free(buf);
    return 0;
}

void fireredvad_apply_cmvn(float *feat, int T, const Cmvn *cmvn)
{
    for (int t = 0; t < T; t++)
    {
        float *f = feat + t * NUM_MEL_BINS;
        for (int d = 0; d < NUM_MEL_BINS; d++)
            f[d] = (f[d] - cmvn->means[d]) * cmvn->inv_std[d];
    }
}

/* ── AED labels ─────────────────────────────────────────────────────── */

const char *AED_LABELS[AED_NUM_CLASSES] = {"speech", "music", "noise"};

/* ── Weight loading ─────────────────────────────────────────────────── */

int fireredvad_load_weights(const char *path, VadWeights *vad, AedWeights *aed)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "Cannot open weights: %s\n", path);
        return -1;
    }

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "FRVD", 4) != 0)
    {
        fprintf(stderr, "Bad magic in %s\n", path);
        fclose(f);
        return -1;
    }
    uint32_t ver;
    fread(&ver, 4, 1, f);

#define RD(ptr, n) fread(ptr, sizeof(float), n, f)

    /* VAD */
    RD(vad->inp_fc1_w, D_IN * D_HIDDEN);
    RD(vad->inp_fc1_b, D_HIDDEN);
    RD(vad->inp_fc2_w, D_HIDDEN * D_PROJ);
    RD(vad->inp_fc2_b, D_PROJ);
    RD(vad->fsmn0_lookback, D_PROJ * D_FILTER);

    for (int i = 0; i < 7; i++)
    {
        RD(vad->blocks[i].fc1_w, D_PROJ * D_HIDDEN);
        RD(vad->blocks[i].fc1_b, D_HIDDEN);
        RD(vad->blocks[i].fc2_w, D_HIDDEN * D_PROJ);
        RD(vad->blocks[i].lookback, D_PROJ * D_FILTER);
    }

    RD(vad->out_fc1_w, D_PROJ * D_HIDDEN);
    RD(vad->out_fc1_b, D_HIDDEN);
    RD(vad->out_fc2_w, D_HIDDEN * 1);
    RD(vad->out_fc2_b, 1);

    /* AED (skip if aed is NULL) */
    if (aed)
    {
        RD(aed->inp_fc1_w, D_IN * D_HIDDEN);
        RD(aed->inp_fc1_b, D_HIDDEN);
        RD(aed->inp_fc2_w, D_HIDDEN * D_PROJ);
        RD(aed->inp_fc2_b, D_PROJ);
        RD(aed->fsmn0_lookback, D_PROJ * D_FILTER);
        RD(aed->fsmn0_lookahead, D_PROJ * D_FILTER);

        for (int i = 0; i < 7; i++)
        {
            RD(aed->blocks[i].fc1_w, D_PROJ * D_HIDDEN);
            RD(aed->blocks[i].fc1_b, D_HIDDEN);
            RD(aed->blocks[i].fc2_w, D_HIDDEN * D_PROJ);
            RD(aed->blocks[i].lookback, D_PROJ * D_FILTER);
            RD(aed->blocks[i].lookahead, D_PROJ * D_FILTER);
        }

        RD(aed->out_fc1_w, D_PROJ * D_HIDDEN);
        RD(aed->out_fc1_b, D_HIDDEN);
        RD(aed->out_fc2_w, D_HIDDEN * AED_NUM_CLASSES);
        RD(aed->out_fc2_b, AED_NUM_CLASSES);
    }

#undef RD

    fclose(f);
    return 0;
}

/* ── Per-stream caches ──────────────────────────────────────────────── */

void fireredvad_caches_init(VadCaches *state)
{
    memset(state->caches, 0, sizeof(state->caches));
}

/* ── Tensor ops ─────────────────────────────────────────────────────── */

/*
 * MatMul: C[T,N] = A[T,M] * B[M,N]
 * With optional bias[N] added and optional ReLU activation.
 *
 * Uses ikj (row-accumulation) loop order for cache-friendly access to B.
 * K-unrolling by 4 reduces load/store traffic on C.
 * With -O2/-O3 and -march=native, the compiler auto-vectorizes the inner loop.
 */
static void matmul_bias_relu(const float *restrict A, const float *restrict B,
                             const float *restrict bias,
                             float *restrict C, int T, int M, int N, int do_relu)
{
    /* Initialize C with bias or zero */
    for (int i = 0; i < T; i++)
    {
        float *restrict c_row = C + (size_t)i * N;
        if (bias)
        {
            memcpy(c_row, bias, N * sizeof(float));
        }
        else
        {
            memset(c_row, 0, N * sizeof(float));
        }
    }

    /* ikj loop with k-unrolling: process 4 rows of B per iteration to reduce
     * load/store traffic on C. The inner SIMD loop processes 4 floats at once. */
    for (int i = 0; i < T; i++)
    {
        float *restrict c_row = C + (size_t)i * N;
        const float *a_row = A + (size_t)i * M;
        int k = 0;
        /* Process 4 rows of B at a time */
        for (; k + 3 < M; k += 4)
        {
            const float a0 = a_row[k], a1 = a_row[k + 1];
            const float a2 = a_row[k + 2], a3 = a_row[k + 3];
            const float *restrict b0 = B + (size_t)k * N;
            const float *restrict b1 = B + (size_t)(k + 1) * N;
            const float *restrict b2 = B + (size_t)(k + 2) * N;
            const float *restrict b3 = B + (size_t)(k + 3) * N;
            int j = 0;
#if defined(USE_NEON)
            float32x4_t va0 = vdupq_n_f32(a0), va1 = vdupq_n_f32(a1);
            float32x4_t va2 = vdupq_n_f32(a2), va3 = vdupq_n_f32(a3);
            for (; j + 3 < N; j += 4)
            {
                float32x4_t acc = vld1q_f32(c_row + j);
                acc = vmlaq_f32(acc, va0, vld1q_f32(b0 + j));
                acc = vmlaq_f32(acc, va1, vld1q_f32(b1 + j));
                acc = vmlaq_f32(acc, va2, vld1q_f32(b2 + j));
                acc = vmlaq_f32(acc, va3, vld1q_f32(b3 + j));
                vst1q_f32(c_row + j, acc);
            }
#elif defined(USE_SSE)
            __m128 va0 = _mm_set1_ps(a0), va1 = _mm_set1_ps(a1);
            __m128 va2 = _mm_set1_ps(a2), va3 = _mm_set1_ps(a3);
            for (; j + 3 < N; j += 4)
            {
                __m128 acc = _mm_loadu_ps(c_row + j);
                acc = _mm_add_ps(acc, _mm_mul_ps(va0, _mm_loadu_ps(b0 + j)));
                acc = _mm_add_ps(acc, _mm_mul_ps(va1, _mm_loadu_ps(b1 + j)));
                acc = _mm_add_ps(acc, _mm_mul_ps(va2, _mm_loadu_ps(b2 + j)));
                acc = _mm_add_ps(acc, _mm_mul_ps(va3, _mm_loadu_ps(b3 + j)));
                _mm_storeu_ps(c_row + j, acc);
            }
#endif
            for (; j < N; j++)
                c_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
        }
        /* Remaining rows of B */
        for (; k < M; k++)
        {
            const float a_ik = a_row[k];
            const float *restrict b_row = B + (size_t)k * N;
            for (int j = 0; j < N; j++)
                c_row[j] += a_ik * b_row[j];
        }
    }

    /* ReLU */
    if (do_relu)
    {
        int total = T * N;
        for (int i = 0; i < total; i++)
            if (C[i] < 0.0f)
                C[i] = 0.0f;
    }
}

/*
 * Depthwise Conv1d lookback filter (causal).
 * Input:  x[D_PROJ][T_in] where T_in = LOOKBACK + T
 * Filter: w[D_PROJ][D_FILTER] (D_FILTER = 20)
 * Output: out[D_PROJ][T] — only the valid (non-padded) portion
 */
static void depthwise_conv1d_lookback(const float *x, int T_in,
                                      const float *w,
                                      float *out, int T)
{
    (void)T_in;
    for (int ch = 0; ch < D_PROJ; ch++)
    {
        const float *x_ch = x + ch * T_in;
        const float *w_ch = w + ch * D_FILTER;
        float *o_ch = out + ch * T;
        for (int t = 0; t < T; t++)
        {
            const float *xp = x_ch + t;
#if defined(USE_NEON)
            /* D_FILTER=20: process as 4x4 + 4 = 20 */
            float32x4_t s0 = vmulq_f32(vld1q_f32(w_ch), vld1q_f32(xp));
            float32x4_t s1 = vmulq_f32(vld1q_f32(w_ch + 4), vld1q_f32(xp + 4));
            float32x4_t s2 = vmulq_f32(vld1q_f32(w_ch + 8), vld1q_f32(xp + 8));
            float32x4_t s3 = vmulq_f32(vld1q_f32(w_ch + 12), vld1q_f32(xp + 12));
            float32x4_t s4 = vmulq_f32(vld1q_f32(w_ch + 16), vld1q_f32(xp + 16));
            float32x4_t sum01 = vaddq_f32(s0, s1);
            float32x4_t sum23 = vaddq_f32(s2, s3);
            float32x4_t sum = vaddq_f32(vaddq_f32(sum01, sum23), s4);
            o_ch[t] = vaddvq_f32(sum);
#elif defined(USE_SSE)
            __m128 s0 = _mm_mul_ps(_mm_loadu_ps(w_ch), _mm_loadu_ps(xp));
            __m128 s1 = _mm_mul_ps(_mm_loadu_ps(w_ch + 4), _mm_loadu_ps(xp + 4));
            __m128 s2 = _mm_mul_ps(_mm_loadu_ps(w_ch + 8), _mm_loadu_ps(xp + 8));
            __m128 s3 = _mm_mul_ps(_mm_loadu_ps(w_ch + 12), _mm_loadu_ps(xp + 12));
            __m128 s4 = _mm_mul_ps(_mm_loadu_ps(w_ch + 16), _mm_loadu_ps(xp + 16));
            __m128 sum = _mm_add_ps(_mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3)), s4);
            /* horizontal sum */
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            _mm_store_ss(&o_ch[t], sum);
#else
            float sum = 0.0f;
            for (int k = 0; k < D_FILTER; k++)
                sum += w_ch[k] * xp[k];
            o_ch[t] = sum;
#endif
        }
    }
}

/*
 * Depthwise Conv1d lookahead filter (non-causal, for AED).
 * Input:  x[D_PROJ][T]
 * Filter: w[D_PROJ][D_FILTER]
 * Output: out[D_PROJ][T] — with lookahead (future context)
 */
static void depthwise_conv1d_lookahead(const float *x, int T,
                                       const float *w,
                                       float *out)
{
    for (int ch = 0; ch < D_PROJ; ch++)
    {
        const float *x_ch = x + ch * T;
        const float *w_ch = w + ch * D_FILTER;
        float *o_ch = out + ch * T;
        for (int t = 0; t < T; t++)
        {
            float sum = 0.0f;
            for (int k = 0; k < D_FILTER; k++)
            {
                int idx = t + 1 + k;
                if (idx < T)
                    sum += w_ch[k] * x_ch[idx];
            }
            o_ch[t] = sum;
        }
    }
}

/* Transpose [T, D_PROJ] -> [D_PROJ, T] */
static void transpose_td(const float *in, float *out, int T, int D)
{
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            out[d * T + t] = in[t * D + d];
}

/* Transpose [D_PROJ, T] -> [T, D_PROJ] */
static void transpose_dt(const float *in, float *out, int D, int T)
{
    for (int d = 0; d < D; d++)
        for (int t = 0; t < T; t++)
            out[t * D + d] = in[d * T + t];
}

/* Add: out[n] = a[n] + b[n] */
/* Callers legitimately alias out with a or b (in-place accumulation).
 * Element-wise add is safe under aliasing at the same index, so drop the
 * `restrict` qualifiers — the NEON/SSE body uses intrinsics and doesn't
 * depend on restrict for vectorization. */
static void vec_add(const float *a, const float *b, float *out, int n)
{
    int i = 0;
#if defined(USE_NEON)
    for (; i + 3 < n; i += 4)
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#elif defined(USE_SSE)
    for (; i + 3 < n; i += 4)
        _mm_storeu_ps(out + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
#endif
    for (; i < n; i++)
        out[i] = a[i] + b[i];
}

/* ── VAD workspace ──────────────────────────────────────────────────── */

int fireredvad_ws_init(VadWorkspace *ws, int max_T)
{
    ws->max_T = max_T;
    ws->hidden = (float *)malloc(max_T * D_HIDDEN * sizeof(float));
    ws->proj = (float *)malloc(max_T * D_PROJ * sizeof(float));
    ws->proj_t = (float *)malloc(D_PROJ * (LOOKBACK + max_T) * sizeof(float));
    ws->conv_out = (float *)malloc(D_PROJ * max_T * sizeof(float));
    ws->fsmn_out = (float *)malloc(D_PROJ * max_T * sizeof(float));
    ws->prev_res = (float *)malloc(max_T * D_PROJ * sizeof(float));
    ws->tmp_td = (float *)malloc(max_T * D_PROJ * sizeof(float));
    ws->proj_cf = (float *)malloc(D_PROJ * max_T * sizeof(float));
    if (!ws->hidden || !ws->proj || !ws->proj_t || !ws->conv_out ||
        !ws->fsmn_out || !ws->prev_res || !ws->tmp_td || !ws->proj_cf)
    {
        fireredvad_ws_free(ws);
        return -1;
    }
    return 0;
}

void fireredvad_ws_free(VadWorkspace *ws)
{
    free(ws->hidden);
    free(ws->proj);
    free(ws->proj_t);
    free(ws->conv_out);
    free(ws->fsmn_out);
    free(ws->prev_res);
    free(ws->tmp_td);
    free(ws->proj_cf);
}

/* ── VAD inference ──────────────────────────────────────────────────── */

void fireredvad_vad_infer(const VadWeights *w, VadCaches *state, const float *feat,
                          int T, float *probs_out, VadWorkspace *ws)
{
    float *hidden = ws->hidden, *proj = ws->proj, *proj_t = ws->proj_t;
    float *conv_out = ws->conv_out, *fsmn_out = ws->fsmn_out;
    float *prev_res = ws->prev_res, *tmp_td = ws->tmp_td, *proj_cf = ws->proj_cf;

    /* Input projection: feat[T,80] -> hidden[T,256] -> proj[T,128] */
    matmul_bias_relu(feat, w->inp_fc1_w, w->inp_fc1_b, hidden, T, D_IN, D_HIDDEN, 1);
    matmul_bias_relu(hidden, w->inp_fc2_w, w->inp_fc2_b, proj, T, D_HIDDEN, D_PROJ, 1);

    /* Block 0: FSMN on input projection */
    /* Transpose proj[T,128] -> proj_t[128, LOOKBACK+T] with cache prepended */
    for (int ch = 0; ch < D_PROJ; ch++)
    {
        memcpy(proj_t + ch * (LOOKBACK + T), state->caches[0][ch], LOOKBACK * sizeof(float));
        for (int t = 0; t < T; t++)
            proj_t[ch * (LOOKBACK + T) + LOOKBACK + t] = proj[t * D_PROJ + ch];
    }

    /* Update cache: last LOOKBACK values */
    for (int ch = 0; ch < D_PROJ; ch++)
    {
        float *col = proj_t + ch * (LOOKBACK + T);
        memcpy(state->caches[0][ch], col + T, LOOKBACK * sizeof(float));
    }

    /* Depthwise conv1d lookback */
    depthwise_conv1d_lookback(proj_t, LOOKBACK + T, w->fsmn0_lookback, conv_out, T);

    /* Residual: proj + conv_out (in channel-first format) */
    transpose_td(proj, proj_cf, T, D_PROJ);
    vec_add(proj_cf, conv_out, fsmn_out, D_PROJ * T);

    /* Transpose back to [T, D_PROJ] for FC layers */
    transpose_dt(fsmn_out, prev_res, D_PROJ, T);

    /* Blocks 1-7 */
    for (int b = 0; b < 7; b++)
    {
        const FsmnBlock *blk = &w->blocks[b];

        /* FC expansion + projection */
        matmul_bias_relu(prev_res, blk->fc1_w, blk->fc1_b, hidden, T, D_PROJ, D_HIDDEN, 1);
        matmul_bias_relu(hidden, blk->fc2_w, NULL, proj, T, D_HIDDEN, D_PROJ, 0);

        /* Transpose proj[T,128] -> proj_t[128, LOOKBACK+T] with cache */
        for (int ch = 0; ch < D_PROJ; ch++)
        {
            memcpy(proj_t + ch * (LOOKBACK + T), state->caches[b + 1][ch], LOOKBACK * sizeof(float));
            for (int t = 0; t < T; t++)
                proj_t[ch * (LOOKBACK + T) + LOOKBACK + t] = proj[t * D_PROJ + ch];
        }

        /* Update cache */
        for (int ch = 0; ch < D_PROJ; ch++)
        {
            float *col = proj_t + ch * (LOOKBACK + T);
            memcpy(state->caches[b + 1][ch], col + T, LOOKBACK * sizeof(float));
        }

        /* Conv1d lookback */
        depthwise_conv1d_lookback(proj_t, LOOKBACK + T, blk->lookback, conv_out, T);

        /* Residual: proj + conv_out */
        transpose_td(proj, proj_cf, T, D_PROJ);
        vec_add(proj_cf, conv_out, fsmn_out, D_PROJ * T);

        /* Transpose back + skip connection with previous block */
        transpose_dt(fsmn_out, tmp_td, D_PROJ, T);
        vec_add(tmp_td, prev_res, prev_res, T * D_PROJ);
    }

    /* Output projection: prev_res[T,128] -> hidden[T,256] -> out[T,1] -> sigmoid */
    matmul_bias_relu(prev_res, w->out_fc1_w, w->out_fc1_b, hidden, T, D_PROJ, D_HIDDEN, 1);
    for (int t = 0; t < T; t++)
    {
        float sum = w->out_fc2_b[0];
        for (int m = 0; m < D_HIDDEN; m++)
            sum += hidden[t * D_HIDDEN + m] * w->out_fc2_w[m];
        probs_out[t] = 1.0f / (1.0f + expf(-sum));
    }
}

/* ── AED inference ──────────────────────────────────────────────────── */

void fireredvad_aed_infer(const AedWeights *model, const float *feat, int T, float *probs_out)
{
    float *hidden = (float *)malloc(T * D_HIDDEN * sizeof(float));
    float *proj = (float *)malloc(T * D_PROJ * sizeof(float));
    float *x_cf = (float *)malloc(D_PROJ * T * sizeof(float));
    float *conv_lb = (float *)malloc(D_PROJ * T * sizeof(float));
    float *conv_la = (float *)malloc(D_PROJ * T * sizeof(float));
    float *fsmn_out = (float *)malloc(D_PROJ * T * sizeof(float));
    float *prev_res = (float *)malloc(T * D_PROJ * sizeof(float));
    float *tmp_td = (float *)malloc(T * D_PROJ * sizeof(float));
    float *padded = (float *)malloc(D_PROJ * (LOOKBACK + T) * sizeof(float));

    /* Input projection */
    matmul_bias_relu(feat, model->inp_fc1_w, model->inp_fc1_b, hidden, T, D_IN, D_HIDDEN, 1);
    matmul_bias_relu(hidden, model->inp_fc2_w, model->inp_fc2_b, proj, T, D_HIDDEN, D_PROJ, 1);

    /* Block 0: FSMN on input projection */
    transpose_td(proj, x_cf, T, D_PROJ);

    /* Lookback conv: zero-pad left by LOOKBACK */
    for (int ch = 0; ch < D_PROJ; ch++)
    {
        memset(padded + ch * (LOOKBACK + T), 0, LOOKBACK * sizeof(float));
        memcpy(padded + ch * (LOOKBACK + T) + LOOKBACK, x_cf + ch * T, T * sizeof(float));
    }
    depthwise_conv1d_lookback(padded, LOOKBACK + T, model->fsmn0_lookback, conv_lb, T);
    vec_add(x_cf, conv_lb, fsmn_out, D_PROJ * T);

    /* Lookahead conv */
    depthwise_conv1d_lookahead(x_cf, T, model->fsmn0_lookahead, conv_la);
    vec_add(fsmn_out, conv_la, fsmn_out, D_PROJ * T);

    transpose_dt(fsmn_out, prev_res, D_PROJ, T);

    /* Blocks 1-7 */
    for (int b = 0; b < 7; b++)
    {
        const AedFsmnBlock *blk = &model->blocks[b];

        matmul_bias_relu(prev_res, blk->fc1_w, blk->fc1_b, hidden, T, D_PROJ, D_HIDDEN, 1);
        matmul_bias_relu(hidden, blk->fc2_w, NULL, proj, T, D_HIDDEN, D_PROJ, 0);

        transpose_td(proj, x_cf, T, D_PROJ);

        /* Lookback */
        for (int ch = 0; ch < D_PROJ; ch++)
        {
            memset(padded + ch * (LOOKBACK + T), 0, LOOKBACK * sizeof(float));
            memcpy(padded + ch * (LOOKBACK + T) + LOOKBACK, x_cf + ch * T, T * sizeof(float));
        }
        depthwise_conv1d_lookback(padded, LOOKBACK + T, blk->lookback, conv_lb, T);
        vec_add(x_cf, conv_lb, fsmn_out, D_PROJ * T);

        /* Lookahead */
        depthwise_conv1d_lookahead(x_cf, T, blk->lookahead, conv_la);
        vec_add(fsmn_out, conv_la, fsmn_out, D_PROJ * T);

        /* Transpose + skip */
        transpose_dt(fsmn_out, tmp_td, D_PROJ, T);
        vec_add(tmp_td, prev_res, prev_res, T * D_PROJ);
    }

    /* Output: [T,128] -> [T,256] -> [T,3] -> sigmoid */
    matmul_bias_relu(prev_res, model->out_fc1_w, model->out_fc1_b, hidden, T, D_PROJ, D_HIDDEN, 1);
    matmul_bias_relu(hidden, model->out_fc2_w, model->out_fc2_b, probs_out, T, D_HIDDEN, AED_NUM_CLASSES, 0);
    for (int i = 0; i < T * AED_NUM_CLASSES; i++)
        probs_out[i] = 1.0f / (1.0f + expf(-probs_out[i]));

    free(padded);
    free(tmp_td);
    free(prev_res);
    free(fsmn_out);
    free(conv_la);
    free(conv_lb);
    free(x_cf);
    free(proj);
    free(hidden);
}

int fireredvad_aed_classify(const AedWeights *model, const float *feat, int T, float *avg_probs)
{
    if (T == 0)
        return -1;
    float *probs = (float *)malloc(T * AED_NUM_CLASSES * sizeof(float));
    fireredvad_aed_infer(model, feat, T, probs);

    for (int c = 0; c < AED_NUM_CLASSES; c++)
        avg_probs[c] = 0;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < AED_NUM_CLASSES; c++)
            avg_probs[c] += probs[t * AED_NUM_CLASSES + c];

    int best = 0;
    for (int c = 0; c < AED_NUM_CLASSES; c++)
    {
        avg_probs[c] /= T;
        if (avg_probs[c] > avg_probs[best])
            best = c;
    }

    free(probs);
    return best;
}

/* ── AED workspace ─────────────────────────────────────────────────── */

int fireredvad_aed_ws_init(AedWorkspace *ws, int max_T)
{
    ws->max_T = max_T;
    ws->hidden = (float *)malloc(max_T * D_HIDDEN * sizeof(float));
    ws->proj = (float *)malloc(max_T * D_PROJ * sizeof(float));
    ws->x_cf = (float *)malloc(D_PROJ * max_T * sizeof(float));
    ws->conv_lb = (float *)malloc(D_PROJ * max_T * sizeof(float));
    ws->conv_la = (float *)malloc(D_PROJ * max_T * sizeof(float));
    ws->fsmn_out = (float *)malloc(D_PROJ * max_T * sizeof(float));
    ws->prev_res = (float *)malloc(max_T * D_PROJ * sizeof(float));
    ws->tmp_td = (float *)malloc(max_T * D_PROJ * sizeof(float));
    ws->padded = (float *)malloc(D_PROJ * (LOOKBACK + max_T) * sizeof(float));
    ws->feat = (float *)malloc(max_T * NUM_MEL_BINS * sizeof(float));
    ws->probs = (float *)malloc(max_T * AED_NUM_CLASSES * sizeof(float));
    if (!ws->hidden || !ws->proj || !ws->x_cf || !ws->conv_lb || !ws->conv_la ||
        !ws->fsmn_out || !ws->prev_res || !ws->tmp_td || !ws->padded ||
        !ws->feat || !ws->probs)
    {
        fireredvad_aed_ws_free(ws);
        return -1;
    }
    return 0;
}

void fireredvad_aed_ws_free(AedWorkspace *ws)
{
    free(ws->hidden);
    free(ws->proj);
    free(ws->x_cf);
    free(ws->conv_lb);
    free(ws->conv_la);
    free(ws->fsmn_out);
    free(ws->prev_res);
    free(ws->tmp_td);
    free(ws->padded);
    free(ws->feat);
    free(ws->probs);
    memset(ws, 0, sizeof(*ws));
}

/* ── Non-allocating fbank for complete segments ────────────────────── */

int fireredvad_extract_fbank_segment_buf(const int16_t *pcm, int pcm_len,
                                          float *out, int max_frames)
{
    int num_frames = 0;
    if (pcm_len >= FRAME_LENGTH)
        num_frames = (pcm_len - FRAME_LENGTH) / FRAME_SHIFT + 1;
    if (num_frames == 0)
        return 0;
    if (num_frames > max_frames)
        num_frames = max_frames;

    float windowed[FRAME_LENGTH], power[NUM_FFT_BINS];
    for (int i = 0; i < num_frames; i++)
    {
        int start = i * FRAME_SHIFT;
        float prev = (start > 0) ? (float)pcm[start - 1] : (float)pcm[start];
        windowed[0] = ((float)pcm[start] - PRE_EMPHASIS * prev) * g_window[0];
        for (int j = 1; j < FRAME_LENGTH; j++)
            windowed[j] = ((float)pcm[start + j] - PRE_EMPHASIS * (float)pcm[start + j - 1]) * g_window[j];
        power_spectrum(windowed, power);
        float *f = out + i * NUM_MEL_BINS;
        for (int m = 0; m < NUM_MEL_BINS; m++)
        {
            float sum = 0.0f;
            for (int k = 0; k < NUM_FFT_BINS; k++)
                sum += g_mel_fb[m][k] * power[k];
            f[m] = logf(max_f(sum, 1e-10f));
        }
    }
    return num_frames;
}

/* ── AED inference with pre-allocated workspace ────────────────────── */

void fireredvad_aed_infer_ws(const AedWeights *model, const float *feat, int T,
                              float *probs_out, AedWorkspace *ws)
{
    float *hidden = ws->hidden, *proj = ws->proj;
    float *x_cf = ws->x_cf, *conv_lb = ws->conv_lb, *conv_la = ws->conv_la;
    float *fsmn_out = ws->fsmn_out, *prev_res = ws->prev_res;
    float *tmp_td = ws->tmp_td, *padded = ws->padded;

    /* Input projection */
    matmul_bias_relu(feat, model->inp_fc1_w, model->inp_fc1_b, hidden, T, D_IN, D_HIDDEN, 1);
    matmul_bias_relu(hidden, model->inp_fc2_w, model->inp_fc2_b, proj, T, D_HIDDEN, D_PROJ, 1);

    /* Block 0: FSMN on input projection */
    transpose_td(proj, x_cf, T, D_PROJ);

    /* Lookback conv: zero-pad left by LOOKBACK */
    for (int ch = 0; ch < D_PROJ; ch++)
    {
        memset(padded + ch * (LOOKBACK + T), 0, LOOKBACK * sizeof(float));
        memcpy(padded + ch * (LOOKBACK + T) + LOOKBACK, x_cf + ch * T, T * sizeof(float));
    }
    depthwise_conv1d_lookback(padded, LOOKBACK + T, model->fsmn0_lookback, conv_lb, T);
    vec_add(x_cf, conv_lb, fsmn_out, D_PROJ * T);

    /* Lookahead conv */
    depthwise_conv1d_lookahead(x_cf, T, model->fsmn0_lookahead, conv_la);
    vec_add(fsmn_out, conv_la, fsmn_out, D_PROJ * T);

    transpose_dt(fsmn_out, prev_res, D_PROJ, T);

    /* Blocks 1-7 */
    for (int b = 0; b < 7; b++)
    {
        const AedFsmnBlock *blk = &model->blocks[b];

        matmul_bias_relu(prev_res, blk->fc1_w, blk->fc1_b, hidden, T, D_PROJ, D_HIDDEN, 1);
        matmul_bias_relu(hidden, blk->fc2_w, NULL, proj, T, D_HIDDEN, D_PROJ, 0);

        transpose_td(proj, x_cf, T, D_PROJ);

        /* Lookback */
        for (int ch = 0; ch < D_PROJ; ch++)
        {
            memset(padded + ch * (LOOKBACK + T), 0, LOOKBACK * sizeof(float));
            memcpy(padded + ch * (LOOKBACK + T) + LOOKBACK, x_cf + ch * T, T * sizeof(float));
        }
        depthwise_conv1d_lookback(padded, LOOKBACK + T, blk->lookback, conv_lb, T);
        vec_add(x_cf, conv_lb, fsmn_out, D_PROJ * T);

        /* Lookahead */
        depthwise_conv1d_lookahead(x_cf, T, blk->lookahead, conv_la);
        vec_add(fsmn_out, conv_la, fsmn_out, D_PROJ * T);

        /* Transpose + skip */
        transpose_dt(fsmn_out, tmp_td, D_PROJ, T);
        vec_add(tmp_td, prev_res, prev_res, T * D_PROJ);
    }

    /* Output: [T,128] -> [T,256] -> [T,3] -> sigmoid */
    matmul_bias_relu(prev_res, model->out_fc1_w, model->out_fc1_b, hidden, T, D_PROJ, D_HIDDEN, 1);
    matmul_bias_relu(hidden, model->out_fc2_w, model->out_fc2_b, probs_out, T, D_HIDDEN, AED_NUM_CLASSES, 0);
    for (int i = 0; i < T * AED_NUM_CLASSES; i++)
        probs_out[i] = 1.0f / (1.0f + expf(-probs_out[i]));
}

int fireredvad_aed_classify_ws(const AedWeights *model, const float *feat,
                                int T, float *avg_probs, AedWorkspace *ws)
{
    if (T == 0)
        return -1;
    fireredvad_aed_infer_ws(model, feat, T, ws->probs, ws);

    for (int c = 0; c < AED_NUM_CLASSES; c++)
        avg_probs[c] = 0;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < AED_NUM_CLASSES; c++)
            avg_probs[c] += ws->probs[t * AED_NUM_CLASSES + c];

    int best = 0;
    for (int c = 0; c < AED_NUM_CLASSES; c++)
    {
        avg_probs[c] /= T;
        if (avg_probs[c] > avg_probs[best])
            best = c;
    }
    return best;
}

/* ── VAD state machine ──────────────────────────────────────────────── */

void fireredvad_sm_init(VadSmStateMachine *sm)
{
    memset(sm, 0, sizeof(*sm));
    sm->last_speech_start_frame = -1;
    sm->last_speech_end_frame = -1;
    sm->pad_start_frame = max_i(SMOOTH_WINDOW_SIZE, PAD_START_FRAME);
    sm->speech_threshold = SPEECH_THRESHOLD;
    sm->min_speech_frames = MIN_SPEECH_FRAME;
    sm->min_silence_frames = MIN_SILENCE_FRAME;
    sm->max_speech_frames = MAX_SPEECH_FRAME;
}

int fireredvad_sm_process_frame(VadSmStateMachine *sm, float raw_prob, VadEvent *evt)
{
    sm->frame_cnt++;
    if (sm->smooth_len < SMOOTH_WINDOW_SIZE)
    {
        sm->smooth_window[sm->smooth_len] = raw_prob;
        sm->smooth_sum += raw_prob;
        sm->smooth_len++;
    }
    else
    {
        sm->smooth_sum -= sm->smooth_window[sm->smooth_head];
        sm->smooth_window[sm->smooth_head] = raw_prob;
        sm->smooth_sum += raw_prob;
        sm->smooth_head = (sm->smooth_head + 1) % SMOOTH_WINDOW_SIZE;
    }
    float smoothed = sm->smooth_sum / sm->smooth_len;
    int is_speech = smoothed >= sm->speech_threshold;
    int has_event = 0;

    if (sm->hit_max_speech)
    {
        evt->type = "speech_start";
        evt->start_frame = sm->frame_cnt;
        evt->end_frame = sm->frame_cnt;
        sm->last_speech_start_frame = sm->frame_cnt;
        sm->hit_max_speech = 0;
        has_event = 1;
    }

    switch (sm->state)
    {
    case STATE_SILENCE:
        if (is_speech)
        {
            sm->state = STATE_POSSIBLE_SPEECH;
            sm->speech_cnt = 1;
        }
        else
        {
            sm->silence_cnt++;
            sm->speech_cnt = 0;
        }
        break;
    case STATE_POSSIBLE_SPEECH:
        if (is_speech)
        {
            sm->speech_cnt++;
            if (sm->speech_cnt >= sm->min_speech_frames)
            {
                sm->state = STATE_SPEECH;
                int start = sm->frame_cnt - sm->speech_cnt + 1 - sm->pad_start_frame;
                if (start < 1)
                    start = 1;
                if (start <= sm->last_speech_end_frame)
                    start = sm->last_speech_end_frame + 1;
                sm->last_speech_start_frame = start;
                sm->silence_cnt = 0;
                evt->type = "speech_start";
                evt->start_frame = start;
                evt->end_frame = sm->frame_cnt;
                has_event = 1;
            }
        }
        else
        {
            sm->state = STATE_SILENCE;
            sm->silence_cnt = 1;
            sm->speech_cnt = 0;
        }
        break;
    case STATE_SPEECH:
        sm->speech_cnt++;
        if (is_speech)
        {
            sm->silence_cnt = 0;
            if (sm->speech_cnt >= sm->max_speech_frames)
            {
                sm->hit_max_speech = 1;
                sm->speech_cnt = 0;
                evt->type = "speech_end";
                evt->start_frame = sm->last_speech_start_frame;
                evt->end_frame = sm->frame_cnt;
                sm->last_speech_end_frame = sm->frame_cnt;
                sm->last_speech_start_frame = -1;
                has_event = 1;
            }
        }
        else
        {
            sm->state = STATE_POSSIBLE_SILENCE;
            sm->silence_cnt = 1;
        }
        break;
    case STATE_POSSIBLE_SILENCE:
        sm->speech_cnt++;
        if (is_speech)
        {
            sm->state = STATE_SPEECH;
            sm->silence_cnt = 0;
            if (sm->speech_cnt >= sm->max_speech_frames)
            {
                sm->hit_max_speech = 1;
                sm->speech_cnt = 0;
                evt->type = "speech_end";
                evt->start_frame = sm->last_speech_start_frame;
                evt->end_frame = sm->frame_cnt;
                sm->last_speech_end_frame = sm->frame_cnt;
                sm->last_speech_start_frame = -1;
                has_event = 1;
            }
        }
        else
        {
            sm->silence_cnt++;
            if (sm->silence_cnt >= sm->min_silence_frames)
            {
                sm->state = STATE_SILENCE;
                evt->type = "speech_end";
                evt->start_frame = sm->last_speech_start_frame;
                evt->end_frame = sm->frame_cnt;
                sm->last_speech_end_frame = sm->frame_cnt;
                sm->last_speech_start_frame = -1;
                sm->speech_cnt = 0;
                has_event = 1;
            }
        }
        break;
    }
    return has_event;
}

int fireredvad_sm_flush(VadSmStateMachine *sm, VadEvent *evt)
{
    if (sm->state == STATE_SPEECH || sm->state == STATE_POSSIBLE_SILENCE)
    {
        evt->type = "speech_end";
        evt->start_frame = sm->last_speech_start_frame;
        evt->end_frame = sm->frame_cnt;
        sm->state = STATE_SILENCE;
        sm->last_speech_end_frame = sm->frame_cnt;
        sm->last_speech_start_frame = -1;
        sm->speech_cnt = 0;
        return 1;
    }
    return 0;
}

/* ── WAV I/O ────────────────────────────────────────────────────────── */

int fireredvad_wav_read(const char *path, WavFile *wav)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "Cannot open WAV: %s\n", path);
        return -1;
    }
    char riff[4];
    uint32_t file_size;
    char wave[4];
    fread(riff, 1, 4, f);
    fread(&file_size, 4, 1, f);
    fread(wave, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0)
    {
        fprintf(stderr, "Not a WAV: %s\n", path);
        fclose(f);
        return -1;
    }
    int16_t audio_fmt = 0, num_channels = 0, bits_per_sample = 0;
    int32_t sample_rate_read = 0, data_size = 0;
    int found_fmt = 0, found_data = 0;
    while (!found_data && !feof(f))
    {
        char cid[4];
        uint32_t csz;
        if (fread(cid, 1, 4, f) != 4)
            break;
        if (fread(&csz, 4, 1, f) != 1)
            break;
        if (memcmp(cid, "fmt ", 4) == 0)
        {
            fread(&audio_fmt, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sample_rate_read, 4, 1, f);
            fseek(f, 6, SEEK_CUR);
            fread(&bits_per_sample, 2, 1, f);
            if (csz > 16)
                fseek(f, csz - 16, SEEK_CUR);
            found_fmt = 1;
        }
        else if (memcmp(cid, "data", 4) == 0)
        {
            data_size = (int32_t)csz;
            found_data = 1;
        }
        else
        {
            fseek(f, csz, SEEK_CUR);
        }
    }
    if (!found_fmt || !found_data || audio_fmt != 1 || bits_per_sample != 16)
    {
        fprintf(stderr, "Unsupported WAV format in %s\n", path);
        fclose(f);
        return -1;
    }
    int ns = data_size / (num_channels * 2);
    int16_t *data = (int16_t *)malloc(ns * sizeof(int16_t));
    if (num_channels == 1)
    {
        fread(data, 2, ns, f);
    }
    else
    {
        int16_t *buf = (int16_t *)malloc(num_channels * 2);
        for (int i = 0; i < ns; i++)
        {
            fread(buf, 2, num_channels, f);
            data[i] = buf[0];
        }
        free(buf);
    }
    fclose(f);
    if (sample_rate_read != SAMPLE_RATE)
        fprintf(stderr, "Warning: sample rate %d != %d\n", sample_rate_read, SAMPLE_RATE);
    wav->data = data;
    wav->num_samples = ns;
    wav->sample_rate = sample_rate_read;
    return 0;
}

int fireredvad_wav_write(const char *path, const int16_t *data, int ns)
{
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "Cannot write: %s\n", path);
        return -1;
    }
    int32_t dsz = ns * 2, fsz = 36 + dsz;
    int32_t fmt = 16;
    int16_t af = 1, ch = 1;
    int32_t sr = SAMPLE_RATE, br = sr * 2;
    int16_t ba = 2, bps = 16;
    fwrite("RIFF", 1, 4, f);
    fwrite(&fsz, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    fwrite(&fmt, 4, 1, f);
    fwrite(&af, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&br, 4, 1, f);
    fwrite(&ba, 2, 1, f);
    fwrite(&bps, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&dsz, 4, 1, f);
    fwrite(data, 2, ns, f);
    fclose(f);
    return 0;
}
