// vm_detect: voicemail classifier over 32000 float32 PCM samples @ 16 kHz mono.
//
// Usage:
//   ./vm_detect <weights_dir> [--workers N] [audio.f32 ...]
//
// With no file args, reads one sample from stdin (model loaded, one forward, exit).
// With file args, loads the model once and runs a forward per file.
//   --workers N enables concurrent processing via N pthreads sharing the model.
//
// When concurrent, each line is prefixed with a worker id and a wall-clock
// timestamp (ms since the pool launched) so the overlap is visible.

#include "wav2vec_vm.h"
#include "sgemm.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const model_t *model;
    int id;
    double t0_ms;                 // shared zero-time for all workers

    // Shared work queue: (files, count) read-only; next is the atomic cursor.
    const char **files;
    int count;
    pthread_mutex_t *mu;
    int *next;
    int intra_threads;
} worker_t;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// vm_detect reads raw float32 PCM. If someone feeds it an .mp3/.wav, the bytes
// decode to junk floats (often NaN/Inf), which silently propagate through the
// net and yield nan probabilities. Catch that here instead.
//
// We bit-test the IEEE-754 exponent because the build uses -ffast-math
// (-ffinite-math-only), under which `x != x` folds to false.
static int audio_has_nonfinite(const float *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned u;
        memcpy(&u, &a[i], sizeof(u));
        if ((u & 0x7F800000u) == 0x7F800000u) return 1;  // NaN or +/-Inf
    }
    return 0;
}

static int read_audio(const char *path, float *audio) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); fclose(f); return -1; }
    long bytes = ftell(f);
    rewind(f);
    const long expect = (long)INPUT_SAMPLES * (long)sizeof(float);
    if (bytes != expect) {
        fprintf(stderr,
                "%s: expected %ld bytes (%d float32 samples @ 16 kHz mono), "
                "got %ld — did you forget `python prep_audio.py %s`?\n",
                path, expect, INPUT_SAMPLES, bytes, path);
        fclose(f);
        return -1;
    }
    size_t n = fread(audio, sizeof(float), INPUT_SAMPLES, f);
    fclose(f);
    if (n != INPUT_SAMPLES) {
        fprintf(stderr, "%s: short read (%zu/%d samples)\n", path, n, INPUT_SAMPLES);
        return -1;
    }
    if (audio_has_nonfinite(audio, INPUT_SAMPLES)) {
        fprintf(stderr,
                "%s: input contains NaN/Inf — file is not valid float32 PCM "
                "(run prep_audio.py to produce a .f32)\n", path);
        return -1;
    }
    return 0;
}

static void run_one(const model_t *m, forward_scratch_t *s,
                    const char *path, int worker_id,
                    double t0_ms, int concurrent) {
    float audio[INPUT_SAMPLES];
    if (read_audio(path, audio) < 0) return;

    double t_start = now_ms() - t0_ms;
    float logits[NUM_LABELS], probs[NUM_LABELS];
    model_forward(m, s, audio, logits, probs);
    double t_end = now_ms() - t0_ms;
    double fwd_ms = t_end - t_start;

    const char *label = probs[1] > probs[0] ? "voicemail" : "human";
    if (concurrent) {
        // Line 1 at start of work, line 2 at end. printf is line-buffered on TTY
        // and fully buffered on pipes; a single printf call is atomic on POSIX
        // for small outputs, so merging both events into one line keeps it clean.
        printf("[w%d] %7.1f->%7.1f ms  (%6.1f ms)  %-50s  %s  (vm=%.3f)\n",
               worker_id, t_start, t_end, fwd_ms, path, label, probs[1]);
    } else {
        printf("%-50s  %s  (human=%.3f, voicemail=%.3f)  [%.1f ms]\n",
               path, label, probs[0], probs[1], fwd_ms);
    }
    fflush(stdout);
}

static void *worker_main(void *arg) {
    worker_t *w = (worker_t *)arg;
    sgemm_set_threads(w->intra_threads);
    forward_scratch_t *s = forward_scratch_new();
    for (;;) {
        pthread_mutex_lock(w->mu);
        int idx = (*w->next)++;
        pthread_mutex_unlock(w->mu);
        if (idx >= w->count) break;
        run_one(w->model, s, w->files[idx], w->id, w->t0_ms, /*concurrent*/1);
    }
    forward_scratch_free(s);
    return NULL;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s <weights_dir> [--workers N] [--intra N] [audio.f32 ...]\n"
            "  audio: 32000 float32 samples @ 16 kHz mono (--help for preprocessing)\n"
            "  --workers N: process files concurrently using N pthreads (default 1)\n"
            "  --intra   N: intra-op SGEMM threads per worker (default 1)\n"
            "  no files given -> reads one sample from stdin\n",
            prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    const char *weights_dir = argv[1];
    int workers = 1;
    int intra = 1;
    if (getenv("SGEMM_THREADS")) intra = atoi(getenv("SGEMM_THREADS"));

    // Parse flags and collect positional file args
    const char **files = (const char **)calloc(argc, sizeof(char *));
    int nfiles = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--workers") && i + 1 < argc) { workers = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--intra") && i + 1 < argc) { intra = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else if (argv[i][0] == '-') { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
        else files[nfiles++] = argv[i];
    }

    // Load the model exactly once.
    double lt0 = now_ms();
    model_t *m = model_load(weights_dir);
    double load_ms = now_ms() - lt0;
    fprintf(stderr, "loaded %s in %.1f ms\n", weights_dir, load_ms);

    // No files -> single stdin sample (legacy behaviour)
    if (nfiles == 0) {
        sgemm_set_threads(intra);
        float audio[INPUT_SAMPLES];
        size_t n = fread(audio, sizeof(float), INPUT_SAMPLES, stdin);
        if (n != INPUT_SAMPLES) {
            fprintf(stderr, "stdin: expected %d float32 samples, got %zu\n", INPUT_SAMPLES, n);
            free(files); model_free(m); return 1;
        }
        if (audio_has_nonfinite(audio, INPUT_SAMPLES)) {
            fprintf(stderr, "stdin: input contains NaN/Inf — not valid float32 PCM\n");
            free(files); model_free(m); return 1;
        }
        forward_scratch_t *s = forward_scratch_new();
        float logits[NUM_LABELS], probs[NUM_LABELS];
        double t0 = now_ms();
        model_forward(m, s, audio, logits, probs);
        double fwd_ms = now_ms() - t0;
        const char *label = probs[1] > probs[0] ? "voicemail" : "human";
        printf("prediction: %s  (human=%.3f, voicemail=%.3f)\n", label, probs[0], probs[1]);
        fprintf(stderr, "forward: %.1f ms\n", fwd_ms);
        forward_scratch_free(s);
        free(files); model_free(m); return 0;
    }

    if (workers <= 1) {
        // Sequential path over all files
        sgemm_set_threads(intra);
        forward_scratch_t *s = forward_scratch_new();
        double t0 = now_ms();
        for (int i = 0; i < nfiles; i++) run_one(m, s, files[i], 0, t0, 0);
        fprintf(stderr, "processed %d file(s) in %.1f ms\n", nfiles, now_ms() - t0);
        forward_scratch_free(s);
    } else {
        // Concurrent path: N pthreads pull from a shared queue over `files`.
        if (workers > nfiles) workers = nfiles;
        fprintf(stderr, "running %d file(s) across %d workers (intra=%d each)\n",
                nfiles, workers, intra);

        pthread_t *th = (pthread_t *)calloc(workers, sizeof(pthread_t));
        worker_t *ctx = (worker_t *)calloc(workers, sizeof(worker_t));
        pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
        int next = 0;
        double t0 = now_ms();

        for (int i = 0; i < workers; i++) {
            ctx[i] = (worker_t){m, i, t0, files, nfiles, &mu, &next, intra};
            pthread_create(&th[i], NULL, worker_main, &ctx[i]);
        }
        for (int i = 0; i < workers; i++) pthread_join(th[i], NULL);

        double wall_ms = now_ms() - t0;
        double tput = 1000.0 * nfiles / wall_ms;
        fprintf(stderr, "processed %d file(s) in %.1f ms  (%.2f fwd/s)\n", nfiles, wall_ms, tput);
        free(th); free(ctx);
    }

    free(files);
    model_free(m);
    return 0;
}
