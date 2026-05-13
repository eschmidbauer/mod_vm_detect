#ifndef MOD_VM_DETECT_H
#define MOD_VM_DETECT_H

#include <switch.h>
#include <speex/speex_resampler.h>

#include "fireredvad.h"
/* fireredvad.h and wav2vec_vm.h both define SAMPLE_RATE=16000; identical value,
 * but undef before second include to silence -Wmacro-redefined on strict builds. */
#ifdef SAMPLE_RATE
#undef SAMPLE_RATE
#endif
#include "wav2vec_vm.h"

#define VM_DETECT_BUG_NAME        "vm_detect"
#define VM_DETECT_MAX_SESSION_ID  256
#define VM_DETECT_MAX_CHUNK_FRAMES 10
#define VM_DETECT_INITIAL_PCM_CAP (SAMPLE_RATE * 2)   /* 2 s initial */
#define VM_DETECT_PCM_BUF_MAX_CAP (SAMPLE_RATE * 120) /* 120 s cap   */

#define VM_DETECT_EVENT_RESULT    "vm_detect::result"

#define VM_DETECT_LABEL_HUMAN     "human"
#define VM_DETECT_LABEL_VOICEMAIL "voicemail"
#define VM_DETECT_LABEL_UNCERTAIN "uncertain"

typedef enum {
    VM_WINDOW_FIRST = 0,
    VM_WINDOW_LAST,
    VM_WINDOW_AVERAGE
} vm_window_policy_t;

struct vm_detect_private {
    char session_id[VM_DETECT_MAX_SESSION_ID];

    /* Resampler (native codec rate → 16 kHz) */
    SpeexResamplerState *read_resampler;
    int channels;
    int native_rate;

    /* Streaming VAD state */
    VadCaches *vad_caches;
    VadWorkspace vad_ws;
    int16_t *fbank_remainder;
    int fbank_rem_len;
    float *feat_buf;
    float *prob_buf;
    VadSmStateMachine vad_sm;

    /* AED (audio event detection: speech / music / noise) — non-streaming,
     * runs once per detected speech segment to gate / annotate the wav2vec
     * classification. */
    AedWorkspace aed_ws;
    int aed_ws_max_T;

    /* Raw 16 kHz PCM ring for segment extraction */
    int16_t *pcm_buf;
    size_t pcm_buf_size;
    size_t pcm_buf_cap;
    size_t pcm_samples_discarded;

    size_t vad_frame_count;
    switch_mutex_t *mutex;
    switch_time_t vad_start_time;

    /* Last classification (exposed via vm_detect API + channel vars) */
    switch_mutex_t *result_mutex;
    char last_label[16];
    float last_p_human;
    float last_p_voicemail;
    switch_time_t last_latency_us;
    switch_time_t last_result_time;
    int result_count;
};

typedef struct vm_detect_private private_t;

#endif /* MOD_VM_DETECT_H */
