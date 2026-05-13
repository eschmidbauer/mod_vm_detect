/*
 * fireredvad.h - Public API for FireRedVAD
 *
 * Streaming Voice Activity Detection + Audio Event Detection
 * with hand-rolled DFSMN inference. Zero external dependencies.
 */

#ifndef FIREREDVAD_H
#define FIREREDVAD_H

#include <stdint.h>

/* ── Audio constants ── */

#define SAMPLE_RATE 16000
#define FRAME_LENGTH 400
#define FRAME_SHIFT 160
#define FFT_SIZE 512
#define NUM_FFT_BINS (FFT_SIZE / 2 + 1)
#define NUM_MEL_BINS 80
#define FRAMES_PER_SECOND 100

/* ── DFSMN dimensions ── */

#define D_IN 80
#define D_HIDDEN 256
#define D_PROJ 128
#define D_FILTER 20
#define N_BLOCKS 8
#define LOOKBACK 19

/* ── AED ── */

#define AED_NUM_CLASSES 3
extern const char *AED_LABELS[AED_NUM_CLASSES];

/* ── VAD state machine constants ── */

#define SMOOTH_WINDOW_SIZE 5
#define SPEECH_THRESHOLD 0.4f
#define PAD_START_FRAME 5
#define MIN_SPEECH_FRAME 8
#define MAX_SPEECH_FRAME 2000
#define MIN_SILENCE_FRAME 50

/* ── Types ── */

typedef struct {
    float means[NUM_MEL_BINS];
    float inv_std[NUM_MEL_BINS];
} Cmvn;

typedef struct {
    float fc1_w[D_PROJ * D_HIDDEN];
    float fc1_b[D_HIDDEN];
    float fc2_w[D_HIDDEN * D_PROJ];
    float lookback[D_PROJ * D_FILTER];
} FsmnBlock;

typedef struct {
    float inp_fc1_w[D_IN * D_HIDDEN];
    float inp_fc1_b[D_HIDDEN];
    float inp_fc2_w[D_HIDDEN * D_PROJ];
    float inp_fc2_b[D_PROJ];
    float fsmn0_lookback[D_PROJ * D_FILTER];
    FsmnBlock blocks[7];
    float out_fc1_w[D_PROJ * D_HIDDEN];
    float out_fc1_b[D_HIDDEN];
    float out_fc2_w[D_HIDDEN * 1];
    float out_fc2_b[1];
} VadWeights;

typedef struct {
    float caches[N_BLOCKS][D_PROJ][LOOKBACK];
} VadCaches;

typedef struct {
    float fc1_w[D_PROJ * D_HIDDEN];
    float fc1_b[D_HIDDEN];
    float fc2_w[D_HIDDEN * D_PROJ];
    float lookback[D_PROJ * D_FILTER];
    float lookahead[D_PROJ * D_FILTER];
} AedFsmnBlock;

typedef struct {
    float inp_fc1_w[D_IN * D_HIDDEN];
    float inp_fc1_b[D_HIDDEN];
    float inp_fc2_w[D_HIDDEN * D_PROJ];
    float inp_fc2_b[D_PROJ];
    float fsmn0_lookback[D_PROJ * D_FILTER];
    float fsmn0_lookahead[D_PROJ * D_FILTER];
    AedFsmnBlock blocks[7];
    float out_fc1_w[D_PROJ * D_HIDDEN];
    float out_fc1_b[D_HIDDEN];
    float out_fc2_w[D_HIDDEN * AED_NUM_CLASSES];
    float out_fc2_b[AED_NUM_CLASSES];
} AedWeights;

typedef struct {
    float *hidden;
    float *proj;
    float *proj_t;
    float *conv_out;
    float *fsmn_out;
    float *prev_res;
    float *tmp_td;
    float *proj_cf;
    int max_T;
} VadWorkspace;

typedef struct {
    float *hidden;
    float *proj;
    float *x_cf;
    float *conv_lb;
    float *conv_la;
    float *fsmn_out;
    float *prev_res;
    float *tmp_td;
    float *padded;
    float *feat;      /* max_T * NUM_MEL_BINS — for fbank extraction */
    float *probs;     /* max_T * AED_NUM_CLASSES — for inference output */
    int max_T;
} AedWorkspace;

typedef enum {
    STATE_SILENCE,
    STATE_POSSIBLE_SPEECH,
    STATE_SPEECH,
    STATE_POSSIBLE_SILENCE
} VadSmState;

typedef struct {
    const char *type;
    int start_frame;
    int end_frame;
} VadEvent;

typedef struct {
    VadSmState state;
    int frame_cnt;
    float smooth_window[SMOOTH_WINDOW_SIZE];
    int smooth_len, smooth_head;
    float smooth_sum;
    int speech_cnt, silence_cnt;
    int hit_max_speech;
    int last_speech_start_frame, last_speech_end_frame;
    int pad_start_frame;

    /* Configurable thresholds (set after fireredvad_sm_init to override defaults) */
    float speech_threshold;  /* smoothed prob threshold (default SPEECH_THRESHOLD) */
    int min_speech_frames;   /* min speech frames to confirm (default MIN_SPEECH_FRAME) */
    int min_silence_frames;  /* min silence frames to end segment (default MIN_SILENCE_FRAME) */
    int max_speech_frames;   /* force segment split (default MAX_SPEECH_FRAME) */
} VadSmStateMachine;

typedef struct {
    int16_t *data;
    int num_samples;
    int sample_rate;
} WavFile;

/* ── API ── */

/** Call once before any other function. Initializes window + mel filterbank. */
void fireredvad_init(void);

/** Load CMVN stats from JSON. Returns 0 on success. */
int fireredvad_load_cmvn(const char *path, Cmvn *cmvn);

/** Apply CMVN normalization in-place to feat[T][NUM_MEL_BINS]. */
void fireredvad_apply_cmvn(float *feat, int T, const Cmvn *cmvn);

/** Load VAD + AED weights from binary file. Returns 0 on success. */
int fireredvad_load_weights(const char *path, VadWeights *vad, AedWeights *aed);

/** Zero-initialize per-stream caches (for reuse without realloc). */
void fireredvad_caches_init(VadCaches *state);

/** Allocate workspace for up to max_T frames per chunk. Returns 0 on success. */
int fireredvad_ws_init(VadWorkspace *ws, int max_T);

/** Free workspace buffers. */
void fireredvad_ws_free(VadWorkspace *ws);

/** Streaming fbank extraction. Returns number of frames produced. */
int fireredvad_extract_fbank(const int16_t *pcm, int pcm_len,
                             int16_t *remainder, int *rem_len,
                             float *out, int max_frames);

/** Non-streaming fbank for a complete segment. Caller must free() result. */
float *fireredvad_extract_fbank_segment(const int16_t *pcm, int pcm_len,
                                        int *out_nf);

/** Run streaming VAD. Writes T speech probabilities to probs_out. */
void fireredvad_vad_infer(const VadWeights *w, VadCaches *state,
                          const float *feat, int T, float *probs_out,
                          VadWorkspace *ws);

/** Run non-streaming AED. Writes T*AED_NUM_CLASSES probabilities to probs_out. */
void fireredvad_aed_infer(const AedWeights *model, const float *feat,
                          int T, float *probs_out);

/** Classify a segment. Returns best class index, fills avg_probs[3]. */
int fireredvad_aed_classify(const AedWeights *model, const float *feat,
                            int T, float *avg_probs);

/** Allocate AED workspace for up to max_T frames. Returns 0 on success. */
int fireredvad_aed_ws_init(AedWorkspace *ws, int max_T);

/** Free AED workspace buffers. */
void fireredvad_aed_ws_free(AedWorkspace *ws);

/** Non-streaming fbank into pre-allocated buffer. Returns number of frames. */
int fireredvad_extract_fbank_segment_buf(const int16_t *pcm, int pcm_len,
                                          float *out, int max_frames);

/** AED inference using pre-allocated workspace (no heap allocation). */
void fireredvad_aed_infer_ws(const AedWeights *model, const float *feat,
                              int T, float *probs_out, AedWorkspace *ws);

/** AED classify using pre-allocated workspace (no heap allocation). */
int fireredvad_aed_classify_ws(const AedWeights *model, const float *feat,
                                int T, float *avg_probs, AedWorkspace *ws);

/** Initialize state machine. */
void fireredvad_sm_init(VadSmStateMachine *sm);

/** Feed one frame. Returns 1 if evt was filled. */
int fireredvad_sm_process_frame(VadSmStateMachine *sm, float raw_prob,
                                VadEvent *evt);

/** Flush pending speech at end of stream. Returns 1 if evt was filled. */
int fireredvad_sm_flush(VadSmStateMachine *sm, VadEvent *evt);

/** Read a 16-bit PCM WAV file. Caller must free(wav->data). */
int fireredvad_wav_read(const char *path, WavFile *wav);

/** Write a 16-bit mono 16kHz WAV file. */
int fireredvad_wav_write(const char *path, const int16_t *data, int ns);

#endif /* FIREREDVAD_H */
