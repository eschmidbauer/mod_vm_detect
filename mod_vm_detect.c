#include "mod_vm_detect.h"

#include "sgemm.h"

SWITCH_MODULE_LOAD_FUNCTION(mod_vm_detect_load);
SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_vm_detect_shutdown);
SWITCH_MODULE_DEFINITION(mod_vm_detect, mod_vm_detect_load, mod_vm_detect_shutdown, NULL);

static struct {
    switch_memory_pool_t *pool;
    switch_mutex_t *mutex;

    /* FireRed VAD + AED (shared, read-only) */
    VadWeights *vad_weights;
    AedWeights *aed_weights;
    Cmvn cmvn;
    char fireredvad_weights_path[512];
    char fireredvad_cmvn_path[512];

    /* If AED says non-speech (music or noise) at >= aed_nonspeech_threshold
     * the segment is reported as "uncertain" without running wav2vec. 0
     * disables the gate entirely (still log AED class for visibility). */
    float aed_nonspeech_threshold;

    /* wav2vec2 voicemail classifier (shared, read-only) */
    model_t *vm_model;
    char weights_path[512];

    /* Shared forward scratch + mutex. One instance for the whole process:
     * 30 MB × N channels is prohibitive; speech-end events are rare so
     * contention is negligible (see PLAN.md). */
    forward_scratch_t *fwd_scratch;
    switch_mutex_t *fwd_mutex;

    vm_window_policy_t window_policy;
    float min_confidence;
    int intra_op_threads;
    switch_bool_t emit_on_speech_end;
} globals;

/* ── Config ── */

static vm_window_policy_t parse_window_policy(const char *s, vm_window_policy_t fallback)
{
    if (zstr(s)) return fallback;
    if (!strcasecmp(s, "first"))   return VM_WINDOW_FIRST;
    if (!strcasecmp(s, "last"))    return VM_WINDOW_LAST;
    if (!strcasecmp(s, "average")) return VM_WINDOW_AVERAGE;
    return fallback;
}

static const char *window_policy_str(vm_window_policy_t p)
{
    switch (p) {
    case VM_WINDOW_LAST:    return "last";
    case VM_WINDOW_AVERAGE: return "average";
    case VM_WINDOW_FIRST:
    default:                return "first";
    }
}

static switch_status_t load_config(void)
{
    switch_xml_t cfg, xml, settings, param;

    snprintf(globals.weights_path, sizeof(globals.weights_path),
             "%s%svm_detect%sweights-int8",
             SWITCH_GLOBAL_dirs.data_dir, SWITCH_PATH_SEPARATOR, SWITCH_PATH_SEPARATOR);
    snprintf(globals.fireredvad_weights_path, sizeof(globals.fireredvad_weights_path),
             "%s%svm_detect%sfireredvad.bin",
             SWITCH_GLOBAL_dirs.data_dir, SWITCH_PATH_SEPARATOR, SWITCH_PATH_SEPARATOR);
    snprintf(globals.fireredvad_cmvn_path, sizeof(globals.fireredvad_cmvn_path),
             "%s%svm_detect%sfireredvad.json",
             SWITCH_GLOBAL_dirs.data_dir, SWITCH_PATH_SEPARATOR, SWITCH_PATH_SEPARATOR);

    globals.window_policy = VM_WINDOW_FIRST;
    globals.min_confidence = 0.0f;
    globals.intra_op_threads = 1;
    globals.emit_on_speech_end = SWITCH_TRUE;
    globals.aed_nonspeech_threshold = 0.0f;

    if (!(xml = switch_xml_open_cfg("vm_detect.conf", &cfg, NULL))) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_WARNING,
                          "mod_vm_detect: no vm_detect.conf found, using defaults\n");
        return SWITCH_STATUS_SUCCESS;
    }

    if ((settings = switch_xml_child(cfg, "settings"))) {
        for (param = switch_xml_child(settings, "param"); param; param = param->next) {
            const char *name = switch_xml_attr_soft(param, "name");
            const char *value = switch_xml_attr_soft(param, "value");

            if (!strcasecmp(name, "weights-path") && !zstr(value)) {
                switch_copy_string(globals.weights_path, value, sizeof(globals.weights_path));
            } else if (!strcasecmp(name, "fireredvad-weights-path") && !zstr(value)) {
                switch_copy_string(globals.fireredvad_weights_path, value,
                                   sizeof(globals.fireredvad_weights_path));
            } else if (!strcasecmp(name, "fireredvad-cmvn-path") && !zstr(value)) {
                switch_copy_string(globals.fireredvad_cmvn_path, value,
                                   sizeof(globals.fireredvad_cmvn_path));
            } else if (!strcasecmp(name, "window-policy") && !zstr(value)) {
                globals.window_policy = parse_window_policy(value, VM_WINDOW_FIRST);
            } else if (!strcasecmp(name, "min-confidence") && !zstr(value)) {
                globals.min_confidence = (float)atof(value);
            } else if (!strcasecmp(name, "intra-op-threads") && !zstr(value)) {
                globals.intra_op_threads = atoi(value);
                if (globals.intra_op_threads < 1) globals.intra_op_threads = 1;
            } else if (!strcasecmp(name, "emit-on-speech-end") && !zstr(value)) {
                globals.emit_on_speech_end = switch_true(value) ? SWITCH_TRUE : SWITCH_FALSE;
            } else if (!strcasecmp(name, "aed-nonspeech-threshold") && !zstr(value)) {
                globals.aed_nonspeech_threshold = (float)atof(value);
                if (globals.aed_nonspeech_threshold < 0.0f) globals.aed_nonspeech_threshold = 0.0f;
                if (globals.aed_nonspeech_threshold > 1.0f) globals.aed_nonspeech_threshold = 1.0f;
            }
        }
    }

    switch_xml_free(xml);
    return SWITCH_STATUS_SUCCESS;
}

/* ── PCM buffer helpers ── */

static void pcm_buf_discard(private_t *s, size_t num_samples);

static int pcm_buf_ensure(private_t *s, size_t need)
{
    if (need <= s->pcm_buf_cap) return 0;
    size_t newcap = s->pcm_buf_cap ? s->pcm_buf_cap * 2 : VM_DETECT_INITIAL_PCM_CAP;
    while (newcap < need) newcap *= 2;
    int16_t *nb = (int16_t *)realloc(s->pcm_buf, sizeof(int16_t) * newcap);
    if (!nb) return -1;
    s->pcm_buf = nb;
    s->pcm_buf_cap = newcap;
    return 0;
}

static void pcm_buf_append(private_t *s, const int16_t *samples, size_t num_samples)
{
    if (s->pcm_buf_size + num_samples > VM_DETECT_PCM_BUF_MAX_CAP) {
        size_t discard = s->pcm_buf_size / 2;
        if (discard > 0) pcm_buf_discard(s, discard);
    }
    if (pcm_buf_ensure(s, s->pcm_buf_size + num_samples) != 0) return;
    memcpy(s->pcm_buf + s->pcm_buf_size, samples, sizeof(int16_t) * num_samples);
    s->pcm_buf_size += num_samples;
}

static void pcm_buf_discard(private_t *s, size_t num_samples)
{
    if (num_samples >= s->pcm_buf_size) {
        s->pcm_samples_discarded += s->pcm_buf_size;
        s->pcm_buf_size = 0;
        return;
    }
    memmove(s->pcm_buf, s->pcm_buf + num_samples,
            sizeof(int16_t) * (s->pcm_buf_size - num_samples));
    s->pcm_buf_size -= num_samples;
    s->pcm_samples_discarded += num_samples;
}

/* ── Classification ── */

static const char *pick_label(float p_human, float p_voicemail)
{
    float top = p_human > p_voicemail ? p_human : p_voicemail;
    if (globals.min_confidence > 0.0f && top < globals.min_confidence)
        return VM_DETECT_LABEL_UNCERTAIN;
    return p_voicemail > p_human ? VM_DETECT_LABEL_VOICEMAIL : VM_DETECT_LABEL_HUMAN;
}

/* Fill `INPUT_SAMPLES` float32 from a slice of int16 PCM, zero-padding the tail. */
static void i16_window_to_f32(const int16_t *src, size_t src_len, float *dst)
{
    size_t copy = src_len < (size_t)INPUT_SAMPLES ? src_len : (size_t)INPUT_SAMPLES;
    for (size_t i = 0; i < copy; i++) dst[i] = (float)src[i] / 32768.0f;
    if (copy < (size_t)INPUT_SAMPLES)
        memset(dst + copy, 0, (INPUT_SAMPLES - copy) * sizeof(float));
}

static switch_status_t classify_segment(const int16_t *seg, size_t seg_samples,
                                        float *out_p_human, float *out_p_voicemail,
                                        switch_time_t *out_latency_us)
{
    if (!globals.vm_model || !globals.fwd_scratch) return SWITCH_STATUS_FALSE;

    /* fp32 input buffers are small (128 KB each at 32000 samples) but we want
     * them off the stack to be safe on small FreeSWITCH thread stacks. */
    float *window_a = (float *)malloc(INPUT_SAMPLES * sizeof(float));
    float *window_b = NULL;
    if (!window_a) return SWITCH_STATUS_FALSE;

    int do_avg = (globals.window_policy == VM_WINDOW_AVERAGE &&
                  seg_samples > (size_t)INPUT_SAMPLES);

    if (globals.window_policy == VM_WINDOW_LAST && seg_samples > (size_t)INPUT_SAMPLES) {
        i16_window_to_f32(seg + (seg_samples - INPUT_SAMPLES), INPUT_SAMPLES, window_a);
    } else {
        /* first / average(first-half) / <=2s: take from the start */
        i16_window_to_f32(seg, seg_samples, window_a);
    }

    if (do_avg) {
        window_b = (float *)malloc(INPUT_SAMPLES * sizeof(float));
        if (window_b) {
            i16_window_to_f32(seg + (seg_samples - INPUT_SAMPLES), INPUT_SAMPLES, window_b);
        }
    }

    switch_time_t t0 = switch_micro_time_now();
    float logits[NUM_LABELS];
    float probs_a[NUM_LABELS];
    float probs_b[NUM_LABELS] = {0.0f, 0.0f};

    switch_mutex_lock(globals.fwd_mutex);
    sgemm_set_threads(globals.intra_op_threads);
    model_forward(globals.vm_model, globals.fwd_scratch, window_a, logits, probs_a);
    if (window_b) {
        model_forward(globals.vm_model, globals.fwd_scratch, window_b, logits, probs_b);
    }
    switch_mutex_unlock(globals.fwd_mutex);
    switch_time_t t1 = switch_micro_time_now();

    float p_human, p_voicemail;
    if (window_b) {
        p_human     = 0.5f * (probs_a[0] + probs_b[0]);
        p_voicemail = 0.5f * (probs_a[1] + probs_b[1]);
    } else {
        p_human     = probs_a[0];
        p_voicemail = probs_a[1];
    }

    free(window_a);
    free(window_b);

    *out_p_human = p_human;
    *out_p_voicemail = p_voicemail;
    *out_latency_us = t1 - t0;
    return SWITCH_STATUS_SUCCESS;
}

/* ── Event emission ── */

static void fire_result_event(private_t *s, switch_core_session_t *session,
                              const char *label, float p_human, float p_voicemail,
                              size_t segment_samples, switch_time_t latency_us)
{
    if (globals.emit_on_speech_end != SWITCH_TRUE) return;

    switch_event_t *event;
    if (switch_event_create_subclass(&event, SWITCH_EVENT_CUSTOM, VM_DETECT_EVENT_RESULT)
            != SWITCH_STATUS_SUCCESS)
        return;

    switch_event_add_header_string(event, SWITCH_STACK_BOTTOM, "Unique-ID", s->session_id);
    switch_event_add_header_string(event, SWITCH_STACK_BOTTOM, "VM-Detect-Label", label);
    switch_event_add_header(event, SWITCH_STACK_BOTTOM, "VM-Detect-P-Human", "%.4f", p_human);
    switch_event_add_header(event, SWITCH_STACK_BOTTOM, "VM-Detect-P-Voicemail", "%.4f", p_voicemail);
    switch_event_add_header(event, SWITCH_STACK_BOTTOM, "VM-Detect-Segment-Ms", "%u",
                            (unsigned)(segment_samples * 1000 / SAMPLE_RATE));
    switch_event_add_header(event, SWITCH_STACK_BOTTOM, "VM-Detect-Latency-Ms", "%u",
                            (unsigned)(latency_us / 1000));
    switch_event_add_header_string(event, SWITCH_STACK_BOTTOM, "VM-Detect-Window",
                                   window_policy_str(globals.window_policy));
    switch_event_fire(&event);
}

static void set_channel_vars(switch_core_session_t *session,
                             const char *label, float p_human, float p_voicemail)
{
    switch_channel_t *channel = switch_core_session_get_channel(session);
    if (!channel) return;
    char buf[16];
    switch_channel_set_variable(channel, "vm_detect_label", label);
    switch_snprintf(buf, sizeof(buf), "%.4f", p_human);
    switch_channel_set_variable(channel, "vm_detect_p_human", buf);
    switch_snprintf(buf, sizeof(buf), "%.4f", p_voicemail);
    switch_channel_set_variable(channel, "vm_detect_p_voicemail", buf);
}

/* ── Speech segment extraction + classification ── */

static void handle_speech_end_event(private_t *s, switch_core_session_t *session, const VadEvent *evt)
{
    int abs_start = (evt->start_frame - 1) * FRAME_SHIFT;
    int abs_end   = evt->end_frame * FRAME_SHIFT;
    if (abs_start < 0) abs_start = 0;

    int buf_start = abs_start - (int)s->pcm_samples_discarded;
    int buf_end   = abs_end   - (int)s->pcm_samples_discarded;
    if (buf_start < 0) buf_start = 0;
    if (buf_end > (int)s->pcm_buf_size) buf_end = (int)s->pcm_buf_size;

    /* CLI-compatible windowing: when policy=FIRST, anchor the segment at the
     * start of received audio (not at VAD-detected speech_start). This matches
     * what `prep_audio.py` + the CLI feed the model — the leading silence /
     * breath that VAD would otherwise strip is preserved, and the wav2vec
     * window the module sees is the same one a CLI run on the same input
     * would see. */
    if (globals.window_policy == VM_WINDOW_FIRST) {
        buf_start = 0;
    }

    int seg_len = buf_end - buf_start;
    if (seg_len <= 0) {
        if (buf_end > 0) pcm_buf_discard(s, (size_t)buf_end);
        return;
    }

    size_t seg_samples = (size_t)seg_len;
    int16_t *seg = (int16_t *)malloc(seg_samples * sizeof(int16_t));
    if (!seg) {
        if (buf_end > 0) pcm_buf_discard(s, (size_t)buf_end);
        return;
    }
    memcpy(seg, s->pcm_buf + buf_start, seg_samples * sizeof(int16_t));

    /* AED on the segment (speech / music / noise). Runs alongside wav2vec for
     * visibility, and optionally gates the label when non-speech dominates. */
    int aed_class = -1;
    float aed_probs[AED_NUM_CLASSES] = {0};
    if (globals.aed_weights && s->aed_ws_max_T > 0) {
        int aed_T = (int)(seg_samples / FRAME_SHIFT);
        if (aed_T > s->aed_ws_max_T) aed_T = s->aed_ws_max_T;
        if (aed_T > 0) {
            int got = fireredvad_extract_fbank_segment_buf(seg, (int)seg_samples,
                                                           s->aed_ws.feat, aed_T);
            if (got > 0) {
                fireredvad_apply_cmvn(s->aed_ws.feat, got, &globals.cmvn);
                aed_class = fireredvad_aed_classify_ws(globals.aed_weights,
                                                       s->aed_ws.feat, got,
                                                       aed_probs, &s->aed_ws);
            }
        }
    }

    float p_human = 0.0f, p_voicemail = 0.0f;
    switch_time_t latency = 0;
    switch_status_t rc = classify_segment(seg, seg_samples, &p_human, &p_voicemail, &latency);
    free(seg);

    if (buf_end > 0) pcm_buf_discard(s, (size_t)buf_end);

    if (rc != SWITCH_STATUS_SUCCESS) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_WARNING,
                          "vm_detect: classification failed for segment (%zu samples)\n", seg_samples);
        return;
    }

    const char *label = pick_label(p_human, p_voicemail);

    /* AED gate: if non-speech (music + noise) dominates and threshold is set,
     * downgrade the wav2vec verdict to "uncertain". */
    float p_nonspeech = aed_probs[1] + aed_probs[2];
    if (aed_class >= 0 && globals.aed_nonspeech_threshold > 0.0f &&
        p_nonspeech >= globals.aed_nonspeech_threshold) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                          "vm_detect: AED p_nonspeech=%.3f >= %.2f → forcing label=uncertain "
                          "(was %s)\n", p_nonspeech, globals.aed_nonspeech_threshold, label);
        label = VM_DETECT_LABEL_UNCERTAIN;
    }

    float start_s = (float)(evt->start_frame - 1) / FRAMES_PER_SECOND;
    float end_s   = (float)evt->end_frame / FRAMES_PER_SECOND;
    if (aed_class >= 0) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                          "vm_detect: %s (P_human=%.4f P_vm=%.4f) "
                          "AED=%s [speech=%.3f music=%.3f noise=%.3f] "
                          "seg %.3f-%.3fs (%zu samp) in %ums\n",
                          label, p_human, p_voicemail,
                          AED_LABELS[aed_class], aed_probs[0], aed_probs[1], aed_probs[2],
                          start_s, end_s, seg_samples, (unsigned)(latency / 1000));
    } else {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                          "vm_detect: %s (P_human=%.4f P_vm=%.4f) seg %.3f-%.3fs (%zu samp) in %ums\n",
                          label, p_human, p_voicemail, start_s, end_s, seg_samples,
                          (unsigned)(latency / 1000));
    }

    switch_mutex_lock(s->result_mutex);
    switch_copy_string(s->last_label, label, sizeof(s->last_label));
    s->last_p_human = p_human;
    s->last_p_voicemail = p_voicemail;
    s->last_latency_us = latency;
    s->last_result_time = switch_micro_time_now();
    s->result_count++;
    switch_mutex_unlock(s->result_mutex);

    set_channel_vars(session, label, p_human, p_voicemail);
    fire_result_event(s, session, label, p_human, p_voicemail, seg_samples, latency);
}

/* ── VAD frame pipeline ── */

static void vad_process(private_t *s, switch_core_session_t *session,
                        int16_t *samples, size_t num_samples)
{
    /* One-shot log on first audio chunk so we can confirm RTP is reaching the bug */
    if (s->pcm_samples_discarded == 0 && s->pcm_buf_size == 0) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                          "vm_detect: first audio chunk received (%zu samples @ 16kHz)\n",
                          num_samples);
    }

    pcm_buf_append(s, samples, num_samples);

    int nf = fireredvad_extract_fbank(samples, (int)num_samples,
                                      s->fbank_remainder, &s->fbank_rem_len,
                                      s->feat_buf, VM_DETECT_MAX_CHUNK_FRAMES);
    if (nf <= 0) return;

    fireredvad_apply_cmvn(s->feat_buf, nf, &globals.cmvn);
    fireredvad_vad_infer(globals.vad_weights, s->vad_caches,
                         s->feat_buf, nf, s->prob_buf, &s->vad_ws);

    /* Per-chunk prob stats for the heartbeat below */
    float chunk_max = 0.0f, chunk_sum = 0.0f;
    int chunk_above = 0;
    for (int i = 0; i < nf; i++) {
        float p = s->prob_buf[i];
        chunk_sum += p;
        if (p > chunk_max) chunk_max = p;
        if (p >= SPEECH_THRESHOLD) chunk_above++;
    }

    for (int i = 0; i < nf; i++) {
        float prob = s->prob_buf[i];
        VadEvent evt;
        if (fireredvad_sm_process_frame(&s->vad_sm, prob, &evt)) {
            if (strcmp(evt.type, "speech_start") == 0) {
                s->vad_start_time = switch_micro_time_now();
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                                  "vm_detect: speech_start at frame %d (%.3fs) prob=%.3f\n",
                                  evt.start_frame,
                                  (float)(evt.start_frame - 1) / FRAMES_PER_SECOND,
                                  prob);
            } else if (strcmp(evt.type, "speech_end") == 0) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                                  "vm_detect: speech_end at frame %d (%.3fs)\n",
                                  evt.end_frame,
                                  (float)evt.end_frame / FRAMES_PER_SECOND);
                handle_speech_end_event(s, session, &evt);
            }
        }
        s->vad_frame_count++;

        /* Heartbeat once per second of audio so you can see VAD is alive
         * even when no event fires. Logs avg/max smoothed-input prob and
         * how many frames in the second crossed SPEECH_THRESHOLD. */
        if (s->vad_frame_count % FRAMES_PER_SECOND == 0) {
            float avg = nf > 0 ? chunk_sum / (float)nf : 0.0f;
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                              "vm_detect: vad heartbeat t=%.1fs avg=%.3f max=%.3f thr=%.2f above=%d/%d state=%d\n",
                              (float)s->vad_frame_count / FRAMES_PER_SECOND,
                              avg, chunk_max, SPEECH_THRESHOLD,
                              chunk_above, nf, (int)s->vad_sm.state);
        }
    }
}

/* ── Media bug callback ── */

static switch_bool_t capture_callback(switch_media_bug_t *bug, void *user_data, switch_abc_type_t type)
{
    switch_core_session_t *session = switch_core_media_bug_get_session(bug);
    private_t *tech_pvt = (private_t *)user_data;

    switch (type) {
    case SWITCH_ABC_TYPE_INIT: {
        int err;
        int target_rate = SAMPLE_RATE;
        switch_codec_t *read_codec = switch_core_session_get_read_codec(session);
        tech_pvt->native_rate = read_codec->implementation->actual_samples_per_second;
        tech_pvt->channels = read_codec->implementation->number_of_channels;
        if (tech_pvt->channels <= 0) tech_pvt->channels = 1;

        switch_mutex_init(&tech_pvt->mutex, SWITCH_MUTEX_NESTED, switch_core_session_get_pool(session));
        switch_mutex_init(&tech_pvt->result_mutex, SWITCH_MUTEX_NESTED, switch_core_session_get_pool(session));

        tech_pvt->vad_caches = (VadCaches *)calloc(1, sizeof(VadCaches));
        if (!tech_pvt->vad_caches) return SWITCH_FALSE;
        fireredvad_caches_init(tech_pvt->vad_caches);

        if (fireredvad_ws_init(&tech_pvt->vad_ws, VM_DETECT_MAX_CHUNK_FRAMES) != 0) {
            free(tech_pvt->vad_caches);
            tech_pvt->vad_caches = NULL;
            return SWITCH_FALSE;
        }

        tech_pvt->fbank_remainder = (int16_t *)calloc(FRAME_LENGTH + SAMPLE_RATE, sizeof(int16_t));
        tech_pvt->feat_buf = (float *)malloc(VM_DETECT_MAX_CHUNK_FRAMES * NUM_MEL_BINS * sizeof(float));
        tech_pvt->prob_buf = (float *)malloc(VM_DETECT_MAX_CHUNK_FRAMES * sizeof(float));
        if (!tech_pvt->fbank_remainder || !tech_pvt->feat_buf || !tech_pvt->prob_buf) {
            free(tech_pvt->vad_caches); tech_pvt->vad_caches = NULL;
            fireredvad_ws_free(&tech_pvt->vad_ws);
            free(tech_pvt->fbank_remainder); tech_pvt->fbank_remainder = NULL;
            free(tech_pvt->feat_buf); tech_pvt->feat_buf = NULL;
            free(tech_pvt->prob_buf); tech_pvt->prob_buf = NULL;
            return SWITCH_FALSE;
        }
        tech_pvt->fbank_rem_len = 0;

        fireredvad_sm_init(&tech_pvt->vad_sm);

        /* AED workspace sized for typical speech segments (5 s @ 100 fps).
         * Longer segments are truncated for AED — wav2vec still classifies
         * the full window per the configured policy. */
        tech_pvt->aed_ws_max_T = 500;
        if (fireredvad_aed_ws_init(&tech_pvt->aed_ws, tech_pvt->aed_ws_max_T) != 0) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_WARNING,
                              "vm_detect: AED workspace alloc failed; AED disabled for this session\n");
            tech_pvt->aed_ws_max_T = 0;
        }

        tech_pvt->pcm_buf_cap = VM_DETECT_INITIAL_PCM_CAP;
        tech_pvt->pcm_buf = (int16_t *)malloc(sizeof(int16_t) * tech_pvt->pcm_buf_cap);
        if (!tech_pvt->pcm_buf) {
            free(tech_pvt->vad_caches); tech_pvt->vad_caches = NULL;
            fireredvad_ws_free(&tech_pvt->vad_ws);
            free(tech_pvt->fbank_remainder); tech_pvt->fbank_remainder = NULL;
            free(tech_pvt->feat_buf); tech_pvt->feat_buf = NULL;
            free(tech_pvt->prob_buf); tech_pvt->prob_buf = NULL;
            return SWITCH_FALSE;
        }
        tech_pvt->pcm_buf_size = 0;
        tech_pvt->pcm_samples_discarded = 0;

        if (tech_pvt->native_rate != target_rate) {
            /* Use max-quality (10) speex resampler. The default
             * SWITCH_RESAMPLE_QUALITY is 2, which is fine for human ears but
             * leaves enough aliasing in the low band to perturb the wav2vec
             * features. Bumping to 10 minimizes aliasing artifacts; it can't
             * recover the missing 4-8 kHz content of an 8 kHz call but it
             * cleans up what content survives. */
            tech_pvt->read_resampler = speex_resampler_init(tech_pvt->channels,
                                                            tech_pvt->native_rate,
                                                            target_rate,
                                                            10, &err);
            if (err != 0) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                                  "vm_detect: resampler init failed: %s\n",
                                  speex_resampler_strerror(err));
                return SWITCH_FALSE;
            }
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
                          "vm_detect: attached (native=%d ch=%d policy=%s)\n",
                          tech_pvt->native_rate, tech_pvt->channels,
                          window_policy_str(globals.window_policy));
        break;
    }
    case SWITCH_ABC_TYPE_CLOSE: {
        /* Flush any unterminated speech. Fall back to classifying the entire
         * ringbuffer if VAD never reached a closeable state — supports short
         * bursts that hang up before any speech_end transition. */
        VadEvent evt;
        int flushed = fireredvad_sm_flush(&tech_pvt->vad_sm, &evt);
        if (flushed) {
            handle_speech_end_event(tech_pvt, session, &evt);
        } else if (tech_pvt->pcm_buf_size >= (size_t)(SAMPLE_RATE / 2)) {
            evt.type = "speech_end";
            evt.start_frame = (int)(tech_pvt->pcm_samples_discarded / FRAME_SHIFT) + 1;
            evt.end_frame   = (int)((tech_pvt->pcm_samples_discarded
                                     + tech_pvt->pcm_buf_size) / FRAME_SHIFT);
            if (evt.end_frame < evt.start_frame) evt.end_frame = evt.start_frame;
            handle_speech_end_event(tech_pvt, session, &evt);
        }
        if (tech_pvt->read_resampler) {
            speex_resampler_destroy(tech_pvt->read_resampler);
            tech_pvt->read_resampler = NULL;
        }
        free(tech_pvt->vad_caches); tech_pvt->vad_caches = NULL;
        fireredvad_ws_free(&tech_pvt->vad_ws);
        if (tech_pvt->aed_ws_max_T > 0) {
            fireredvad_aed_ws_free(&tech_pvt->aed_ws);
            tech_pvt->aed_ws_max_T = 0;
        }
        free(tech_pvt->fbank_remainder); tech_pvt->fbank_remainder = NULL;
        free(tech_pvt->feat_buf); tech_pvt->feat_buf = NULL;
        free(tech_pvt->prob_buf); tech_pvt->prob_buf = NULL;
        free(tech_pvt->pcm_buf); tech_pvt->pcm_buf = NULL;
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "vm_detect: detached\n");
        break;
    }
    case SWITCH_ABC_TYPE_READ: {
        uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
        switch_frame_t frame = {0};
        frame.data = data;
        frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;

        while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
            if (frame.datalen == 0) continue;

            int16_t mono_stack[SWITCH_RECOMMENDED_BUFFER_SIZE / sizeof(int16_t)];

            if (tech_pvt->read_resampler == NULL) {
                int16_t *pcm = (int16_t *)frame.data;
                int total_samples = frame.datalen / sizeof(int16_t);
                if (tech_pvt->channels == 1) {
                    vad_process(tech_pvt, session, pcm, total_samples);
                } else {
                    int frames = total_samples / tech_pvt->channels;
                    for (int i = 0; i < frames; i++) {
                        int32_t sum = 0;
                        for (int c = 0; c < tech_pvt->channels; c++)
                            sum += pcm[i * tech_pvt->channels + c];
                        mono_stack[i] = (int16_t)(sum / tech_pvt->channels);
                    }
                    vad_process(tech_pvt, session, mono_stack, frames);
                }
            } else {
                spx_uint32_t in_len = frame.datalen / sizeof(spx_int16_t);
                spx_uint32_t out_max = (spx_uint32_t)((double)in_len * SAMPLE_RATE /
                                                      tech_pvt->native_rate) + 64;
                if (out_max > SWITCH_RECOMMENDED_BUFFER_SIZE / sizeof(spx_int16_t))
                    out_max = SWITCH_RECOMMENDED_BUFFER_SIZE / sizeof(spx_int16_t);
                spx_uint32_t out_len = out_max;
                spx_int16_t out[SWITCH_RECOMMENDED_BUFFER_SIZE / sizeof(spx_int16_t)];

                if (tech_pvt->channels == 1) {
                    speex_resampler_process_int(tech_pvt->read_resampler, 0,
                                                (const spx_int16_t *)frame.data, &in_len,
                                                out, &out_len);
                    vad_process(tech_pvt, session, out, out_len);
                } else {
                    speex_resampler_process_interleaved_int(tech_pvt->read_resampler,
                                                            (const spx_int16_t *)frame.data,
                                                            &in_len, out, &out_len);
                    for (spx_uint32_t i = 0; i < out_len; i++) {
                        int32_t sum = 0;
                        for (int c = 0; c < tech_pvt->channels; c++)
                            sum += out[i * tech_pvt->channels + c];
                        mono_stack[i] = (int16_t)(sum / tech_pvt->channels);
                    }
                    vad_process(tech_pvt, session, mono_stack, out_len);
                }
            }
        }
        break;
    }
    case SWITCH_ABC_TYPE_WRITE:
    default:
        break;
    }

    return SWITCH_TRUE;
}

/* ── Application helpers ── */

static switch_status_t start_capture(switch_core_session_t *session)
{
    switch_channel_t *channel = switch_core_session_get_channel(session);
    switch_media_bug_t *bug;
    switch_status_t status;

    if (switch_channel_get_private(channel, VM_DETECT_BUG_NAME)) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_WARNING,
                          "vm_detect: already attached\n");
        return SWITCH_STATUS_FALSE;
    }

    if (switch_channel_answer(channel) != SWITCH_STATUS_SUCCESS) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                          "vm_detect: channel must be answered before start\n");
        return SWITCH_STATUS_FALSE;
    }

    private_t *tech_pvt = (private_t *)switch_core_session_alloc(session, sizeof(private_t));
    if (!tech_pvt) return SWITCH_STATUS_MEMERR;
    memset(tech_pvt, 0, sizeof(private_t));
    switch_copy_string(tech_pvt->session_id, switch_core_session_get_uuid(session),
                       sizeof(tech_pvt->session_id));

    if ((status = switch_core_media_bug_add(session, VM_DETECT_BUG_NAME, NULL,
                                            capture_callback, tech_pvt, 0,
                                            SMBF_READ_STREAM, &bug))
            != SWITCH_STATUS_SUCCESS) {
        return status;
    }
    switch_channel_set_private(channel, VM_DETECT_BUG_NAME, bug);
    return SWITCH_STATUS_SUCCESS;
}

static switch_status_t stop_capture(switch_core_session_t *session)
{
    switch_channel_t *channel = switch_core_session_get_channel(session);
    switch_media_bug_t *bug = (switch_media_bug_t *)switch_channel_get_private(channel, VM_DETECT_BUG_NAME);
    if (!bug) return SWITCH_STATUS_FALSE;
    switch_channel_set_private(channel, VM_DETECT_BUG_NAME, NULL);
    switch_core_media_bug_remove(session, &bug);
    return SWITCH_STATUS_SUCCESS;
}

#define VM_DETECT_START_APP_SYNTAX ""
SWITCH_STANDARD_APP(vm_detect_start_function)
{
    (void)data;
    if (start_capture(session) != SWITCH_STATUS_SUCCESS) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                          "vm_detect_start: failed to attach\n");
    }
}

#define VM_DETECT_STOP_APP_SYNTAX ""
SWITCH_STANDARD_APP(vm_detect_stop_function)
{
    (void)data;
    stop_capture(session);
}

/* ── Synchronous detect: start → block until result or timeout → stop ── */
#define VM_DETECT_SYNC_APP_SYNTAX "[timeout_ms]"
SWITCH_STANDARD_APP(vm_detect_sync_function)
{
    int timeout_ms = 5000; /* default 5 s */
    if (!zstr(data)) {
        timeout_ms = atoi(data);
        if (timeout_ms <= 0) timeout_ms = 5000;
    }

    switch_channel_t *channel = switch_core_session_get_channel(session);

    if (start_capture(session) != SWITCH_STATUS_SUCCESS) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                          "vm_detect: failed to attach\n");
        switch_channel_set_variable(channel, "vm_detect_sync_status", "attach_failed");
        return;
    }

    switch_media_bug_t *bug = (switch_media_bug_t *)switch_channel_get_private(channel, VM_DETECT_BUG_NAME);
    if (!bug) {
        switch_channel_set_variable(channel, "vm_detect_sync_status", "attach_failed");
        stop_capture(session);
        return;
    }

    private_t *tech_pvt = (private_t *)switch_core_media_bug_get_user_data(bug);
    if (!tech_pvt) {
        switch_channel_set_variable(channel, "vm_detect_sync_status", "attach_failed");
        stop_capture(session);
        return;
    }

    int elapsed = 0;
    int have_result = 0;
    while (switch_channel_ready(channel) && elapsed < timeout_ms) {
        switch_mutex_lock(tech_pvt->result_mutex);
        have_result = tech_pvt->result_count > 0;
        switch_mutex_unlock(tech_pvt->result_mutex);
        if (have_result) break;
        /* switch_ivr_sleep drives the read codec / media bug pipeline while
         * waiting; switch_yield does not, so frames pile up in the RTP socket
         * and the bug callback never fires. */
        switch_ivr_sleep(session, 100, SWITCH_TRUE, NULL);
        elapsed += 100;
    }

    /* Detach before the final check so the bug close path (flush + short-burst
     * fallback) has a chance to land a result before we report timeout. */
    stop_capture(session);

    if (!have_result) {
        switch_mutex_lock(tech_pvt->result_mutex);
        have_result = tech_pvt->result_count > 0;
        switch_mutex_unlock(tech_pvt->result_mutex);
    }

    switch_channel_set_variable(channel, "vm_detect_sync_status",
                                have_result ? "detected" : "timeout");

    const char *label = switch_channel_get_variable(channel, "vm_detect_label");
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                      "vm_detect: sync result=%s label=%s elapsed=%dms timeout=%dms\n",
                      have_result ? "detected" : "timeout",
                      label ? label : "(none)", elapsed, timeout_ms);
}

#define VM_DETECT_API_SYNTAX "<uuid> [start | stop | status]"
SWITCH_STANDARD_API(vm_detect_api_function)
{
    char *mycmd = NULL, *argv[3] = {0};
    int argc = 0;
    switch_status_t status = SWITCH_STATUS_FALSE;

    if (!zstr(cmd) && (mycmd = strdup(cmd))) {
        argc = switch_separate_string(mycmd, ' ', argv, (sizeof(argv) / sizeof(argv[0])));
    }
    if (zstr(cmd) || argc < 2) {
        stream->write_function(stream, "-USAGE: %s\n", VM_DETECT_API_SYNTAX);
        goto done;
    }

    switch_core_session_t *lsession = switch_core_session_locate(argv[0]);
    if (!lsession) {
        stream->write_function(stream, "-ERR session %s not found\n", argv[0]);
        goto done;
    }

    if (!strcasecmp(argv[1], "start")) {
        status = start_capture(lsession);
        stream->write_function(stream, "%s\n",
                               status == SWITCH_STATUS_SUCCESS ? "+OK" : "-ERR start failed");
    } else if (!strcasecmp(argv[1], "stop")) {
        status = stop_capture(lsession);
        stream->write_function(stream, "%s\n",
                               status == SWITCH_STATUS_SUCCESS ? "+OK" : "-ERR not attached");
    } else if (!strcasecmp(argv[1], "status")) {
        switch_channel_t *ch = switch_core_session_get_channel(lsession);
        switch_media_bug_t *bug = (switch_media_bug_t *)switch_channel_get_private(ch, VM_DETECT_BUG_NAME);
        if (!bug) {
            stream->write_function(stream, "-ERR not attached\n");
        } else {
            private_t *s = (private_t *)switch_core_media_bug_get_user_data(bug);
            switch_mutex_lock(s->result_mutex);
            if (s->result_count == 0) {
                stream->write_function(stream, "+OK attached results=0\n");
            } else {
                stream->write_function(stream,
                                       "+OK label=%s p_human=%.4f p_voicemail=%.4f "
                                       "latency_ms=%u results=%d\n",
                                       s->last_label, s->last_p_human, s->last_p_voicemail,
                                       (unsigned)(s->last_latency_us / 1000), s->result_count);
            }
            switch_mutex_unlock(s->result_mutex);
            status = SWITCH_STATUS_SUCCESS;
        }
    } else {
        stream->write_function(stream, "-USAGE: %s\n", VM_DETECT_API_SYNTAX);
    }

    switch_core_session_rwunlock(lsession);

done:
    switch_safe_free(mycmd);
    return SWITCH_STATUS_SUCCESS;
}

/* ── Module load / shutdown ── */

SWITCH_MODULE_LOAD_FUNCTION(mod_vm_detect_load)
{
    switch_api_interface_t *api_interface;
    switch_application_interface_t *app_interface;
    *module_interface = switch_loadable_module_create_module_interface(pool, modname);

    globals.pool = pool;
    switch_mutex_init(&globals.mutex, SWITCH_MUTEX_NESTED, globals.pool);
    switch_mutex_init(&globals.fwd_mutex, SWITCH_MUTEX_NESTED, globals.pool);

    load_config();

    fireredvad_init();

    if (fireredvad_load_cmvn(globals.fireredvad_cmvn_path, &globals.cmvn) != 0) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
                          "mod_vm_detect: failed to load CMVN from %s\n", globals.fireredvad_cmvn_path);
        return SWITCH_STATUS_TERM;
    }

    globals.vad_weights = (VadWeights *)calloc(1, sizeof(VadWeights));
    globals.aed_weights = (AedWeights *)calloc(1, sizeof(AedWeights));
    if (!globals.vad_weights || !globals.aed_weights) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
                          "mod_vm_detect: failed to allocate VAD/AED weights\n");
        free(globals.vad_weights); globals.vad_weights = NULL;
        free(globals.aed_weights); globals.aed_weights = NULL;
        return SWITCH_STATUS_TERM;
    }

    if (fireredvad_load_weights(globals.fireredvad_weights_path,
                                globals.vad_weights, globals.aed_weights) != 0) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
                          "mod_vm_detect: failed to load FireRed VAD/AED weights from %s\n",
                          globals.fireredvad_weights_path);
        free(globals.vad_weights); globals.vad_weights = NULL;
        free(globals.aed_weights); globals.aed_weights = NULL;
        return SWITCH_STATUS_TERM;
    }
    switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO,
                      "mod_vm_detect: loaded FireRed VAD+AED weights from %s\n",
                      globals.fireredvad_weights_path);

    globals.vm_model = model_load(globals.weights_path);
    if (!globals.vm_model) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
                          "mod_vm_detect: failed to load wav2vec2 voicemail model from %s\n",
                          globals.weights_path);
        free(globals.vad_weights); globals.vad_weights = NULL;
        free(globals.aed_weights); globals.aed_weights = NULL;
        return SWITCH_STATUS_TERM;
    }

    globals.fwd_scratch = forward_scratch_new();
    if (!globals.fwd_scratch) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
                          "mod_vm_detect: failed to allocate forward scratch\n");
        model_free(globals.vm_model); globals.vm_model = NULL;
        free(globals.vad_weights); globals.vad_weights = NULL;
        free(globals.aed_weights); globals.aed_weights = NULL;
        return SWITCH_STATUS_TERM;
    }
    switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO,
                      "mod_vm_detect: loaded voicemail classifier from %s (policy=%s, min_conf=%.2f)\n",
                      globals.weights_path,
                      window_policy_str(globals.window_policy),
                      globals.min_confidence);

    if (switch_event_reserve_subclass(VM_DETECT_EVENT_RESULT) != SWITCH_STATUS_SUCCESS) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
                          "mod_vm_detect: could not reserve event subclass\n");
        return SWITCH_STATUS_TERM;
    }

    SWITCH_ADD_APP(app_interface, "vm_detect_start", "Attach voicemail detector",
                   "Attach voicemail detector to current channel",
                   vm_detect_start_function, VM_DETECT_START_APP_SYNTAX, SAF_NONE);
    SWITCH_ADD_APP(app_interface, "vm_detect_stop", "Detach voicemail detector",
                   "Detach voicemail detector from current channel",
                   vm_detect_stop_function, VM_DETECT_STOP_APP_SYNTAX, SAF_NONE);
    /* Note: the name "vm_detect" is registered both as a dialplan APP
     * (synchronous detection on the current channel) and as an FS API
     * (uuid-based start/stop/status from fs_cli/ESL). FreeSWITCH keeps
     * these in separate interface registries, so the overlap is legal
     * — just different invocation contexts. */
    SWITCH_ADD_APP(app_interface, "vm_detect", "Synchronous voicemail detection",
                   "Start detector, block until result or timeout, stop. Args: [timeout_ms]",
                   vm_detect_sync_function, VM_DETECT_SYNC_APP_SYNTAX, SAF_NONE);
    SWITCH_ADD_API(api_interface, "vm_detect", "vm_detect API",
                   vm_detect_api_function, VM_DETECT_API_SYNTAX);
    switch_console_set_complete("add vm_detect ::console::list_uuid start");
    switch_console_set_complete("add vm_detect ::console::list_uuid stop");
    switch_console_set_complete("add vm_detect ::console::list_uuid status");

    return SWITCH_STATUS_SUCCESS;
}

SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_vm_detect_shutdown)
{
    switch_mutex_lock(globals.mutex);
    switch_event_free_subclass(VM_DETECT_EVENT_RESULT);

    if (globals.fwd_scratch) {
        forward_scratch_free(globals.fwd_scratch);
        globals.fwd_scratch = NULL;
    }
    if (globals.vm_model) {
        model_free(globals.vm_model);
        globals.vm_model = NULL;
    }
    if (globals.vad_weights) {
        free(globals.vad_weights);
        globals.vad_weights = NULL;
    }
    if (globals.aed_weights) {
        free(globals.aed_weights);
        globals.aed_weights = NULL;
    }
    switch_mutex_unlock(globals.mutex);
    switch_mutex_destroy(globals.mutex);
    return SWITCH_STATUS_SUCCESS;
}
