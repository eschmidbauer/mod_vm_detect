#ifndef WAV2VEC_VM_H
#define WAV2VEC_VM_H

#include <stdint.h>
#include <stddef.h>
#include "sgemm.h"

#define NUM_CONV_LAYERS 7
#define NUM_ENCODER_LAYERS 24
#define HIDDEN_SIZE 1024
#define FFN_SIZE 4096
#define NUM_HEADS 16
#define HEAD_DIM 64
#define CONV_CHANNELS 512
#define CLASSIFIER_PROJ 256
#define NUM_LABELS 2
#define POS_KERNEL 128
#define POS_GROUPS 16
#define POS_IN_PER_GROUP (HIDDEN_SIZE / POS_GROUPS)
#define LN_EPS 1e-5f
#define INPUT_SAMPLES 32000
#define SAMPLE_RATE 16000

extern const int CONV_KERNELS[NUM_CONV_LAYERS];
extern const int CONV_STRIDES[NUM_CONV_LAYERS];

// Quantizable weight: populated as either fp32 or int8+scale, never both.
// Shape is rows x cols, with the PyTorch convention for nn.Linear: rows=out, cols=in.
// When q8 != NULL, scale points to `rows` floats (per-output-channel scale).
typedef struct {
    const float  *f32;   // fp32 weights, row-major (rows x cols), or NULL if int8
    const int8_t *q8;    // int8 weights, row-major (rows x cols), or NULL if fp32
    const float  *scale; // per-row scale (length == rows), only valid when q8 != NULL
    // Optional: pre-packed B in the sgemm micro-kernel layout. When present,
    // linear() routes through sgemm_s8rm_prepacked and skips pack_B entirely.
    sgemm_s8_prepack_t *prepack;
    int rows, cols;
} qw_t;

typedef struct {
    // Feature extractor (7 Conv1D layers with LayerNorm + GELU).
    // conv0 stays fp32 (K=10 isn't int8-kernel aligned); conv1..6 may be int8.
    qw_t   fe_conv_w[NUM_CONV_LAYERS];
    float *fe_conv_b[NUM_CONV_LAYERS];
    float *fe_norm_w[NUM_CONV_LAYERS];
    float *fe_norm_b[NUM_CONV_LAYERS];

    // Feature projection
    float *fp_norm_w, *fp_norm_b;      // [512]
    qw_t   fp_proj_w;                  // [1024, 512]
    float *fp_proj_b;                  // [1024]

    // Positional convolutional embedding (Conv1D w/ weight norm collapsed, GELU)
    float *pos_w;                      // [1024, 64, 128]  (out_ch, in_ch/groups, kernel)
    float *pos_b;                      // [1024]

    // 24 transformer encoder layers (pre-LayerNorm)
    struct {
        float *ln1_w, *ln1_b;          // [1024]
        qw_t   q_w; float *q_b;        // [1024, 1024] / [1024]
        qw_t   k_w; float *k_b;
        qw_t   v_w; float *v_b;
        qw_t   out_w; float *out_b;    // [1024, 1024] / [1024]
        float *ln2_w, *ln2_b;          // [1024]
        qw_t   ffn_in_w;  float *ffn_in_b;   // [4096, 1024] / [4096]
        qw_t   ffn_out_w; float *ffn_out_b;  // [1024, 4096] / [1024]
    } layers[NUM_ENCODER_LAYERS];

    // Post-encoder LayerNorm + classification head
    float *enc_norm_w, *enc_norm_b;    // [1024]
    qw_t   proj_w; float *proj_b;      // [256, 1024] / [256]  (kept fp32; tiny)
    qw_t   out_w;  float *out_b;       // [2, 256]   / [2]

    // Owned memory block (one free per loaded file, heterogeneous pointer types)
    void **owned;
    size_t num_owned;
} model_t;

typedef struct {
    const char *name;
    int correct;
    float voicemail_prob;
} pred_t;

// Per-forward scratch buffers. Allocated once per caller thread (model_t is
// shared read-only; scratch is NOT — do not share one across threads).
typedef struct forward_scratch forward_scratch_t;
forward_scratch_t *forward_scratch_new(void);
void               forward_scratch_free(forward_scratch_t *s);

// Public API
model_t *model_load(const char *weights_dir);
void     model_free(model_t *m);
// audio: length INPUT_SAMPLES float32, 16kHz mono. Writes logits[2] and probs[2].
void     model_forward(const model_t *m, forward_scratch_t *s,
                       const float *audio,
                       float logits_out[NUM_LABELS], float probs_out[NUM_LABELS]);

// Low-level ops (exposed for testability)
void linear(const float *x, const qw_t *W, const float *b, int M, int K, int N, float *y);
void linear_f32(const float *x, const float *W, const float *b, int M, int K, int N, float *y);
void layer_norm(float *x, int N, int C, const float *gamma, const float *beta, float eps);
void gelu(float *x, int n);
void softmax_rows(float *x, int rows, int cols);

#endif // WAV2VEC_VM_H
