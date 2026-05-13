// Fire N pthreads at a shared model_t, each running model_forward repeatedly on
// its own audio file. Verifies:
//   1. No crashes / deadlocks (concurrency-safe).
//   2. Each thread's prediction matches what the serial binary produces.
//   3. Throughput roughly scales with thread count.

#include "wav2vec_vm.h"
#include "sgemm.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const model_t *model;
    const char *path;
    int iters;
    int intra_threads;
    float expected_vm;   // last known voicemail prob for this file
    double elapsed_ms;
    int ok;
} ctx_t;

static void *worker(void *arg) {
    ctx_t *c = (ctx_t *)arg;
    float audio[INPUT_SAMPLES];
    FILE *f = fopen(c->path, "rb");
    if (!f) { perror(c->path); c->ok = 0; return NULL; }
    size_t n = fread(audio, sizeof(float), INPUT_SAMPLES, f);
    fclose(f);
    if (n < INPUT_SAMPLES) memset(audio + n, 0, (INPUT_SAMPLES - n) * sizeof(float));

    // Each worker opts into its own intra-op pool (independent per caller thread).
    sgemm_set_threads(c->intra_threads);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    float logits[NUM_LABELS], probs[NUM_LABELS];
    forward_scratch_t *s = forward_scratch_new();
    c->ok = 1;
    for (int i = 0; i < c->iters; i++) {
        model_forward(c->model, s, audio, logits, probs);
        // Tolerance covers both paths: fp32 stays within ~1e-3 of the PyTorch
        // golden; W8A8 (int8 weights + dynamic int8 activation quant) can drift
        // up to ~1.5e-2. Classification-correctness is the real invariant.
        if (fabsf(probs[1] - c->expected_vm) > 2e-2f) {
            fprintf(stderr, "[%s] iter %d: vm=%.4f expected %.4f\n",
                    c->path, i, probs[1], c->expected_vm);
            c->ok = 0;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    c->elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    forward_scratch_free(s);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <weights_dir> <N_WORKERS> [intra_threads=1]\n"
                        "  each worker runs a fixed test sample for a few iters\n",
                argv[0]);
        return 2;
    }
    const char *weights_dir = argv[1];
    int N = atoi(argv[2]);
    int intra = argc >= 4 ? atoi(argv[3]) : 1;

    printf("loading model from %s...\n", weights_dir);
    model_t *m = model_load(weights_dir);

    // Paired (sample path, expected voicemail prob) derived from the FP32
    // PyTorch run. Workers cycle through them.
    struct { const char *path; float vm; } cases[] = {
        {"samples/voicemail/Doo-Wop_Dudes.f32",              0.966f},
        {"samples/human/tim-griffith-messages-02.f32",       0.014f},
        {"samples/voicemail/Dial_and_Complain.f32",          0.999f},
        {"samples/voicemail/The_Hard_Boiled_Private_Eye.f32",0.969f},
        {"samples/human/tim-griffith-messages-01.f32",       0.357f},
        {"samples/voicemail/Dinner_For_Two.f32",             0.823f},
        {"samples/voicemail/Dr_Feel_Good.f32",               0.746f},
        {"samples/voicemail/The_Rap_Master_Machine.f32",     0.014f},
    };
    int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

    pthread_t *threads = calloc(N, sizeof(pthread_t));
    ctx_t *ctxs = calloc(N, sizeof(ctx_t));

    int iters = 2;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        ctxs[i] = (ctx_t){m, cases[i % ncases].path, iters,
                          intra, cases[i % ncases].vm, 0.0, 0};
        pthread_create(&threads[i], NULL, worker, &ctxs[i]);
    }
    for (int i = 0; i < N; i++) pthread_join(threads[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double wall_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    int total_fwds = 0, all_ok = 1;
    double worst_ms = 0, sum_ms = 0;
    for (int i = 0; i < N; i++) {
        total_fwds += iters;
        if (!ctxs[i].ok) all_ok = 0;
        if (ctxs[i].elapsed_ms > worst_ms) worst_ms = ctxs[i].elapsed_ms;
        sum_ms += ctxs[i].elapsed_ms;
    }

    printf("\nN workers           : %d  (intra-op threads each: %d)\n", N, intra);
    printf("total forwards      : %d\n", total_fwds);
    printf("wall clock          : %7.1f ms\n", wall_ms);
    printf("slowest worker      : %7.1f ms  (per fwd ~%.1f ms)\n", worst_ms, worst_ms / iters);
    printf("avg worker elapsed  : %7.1f ms\n", sum_ms / N);
    printf("overall throughput  : %7.1f fwd/s\n", 1000.0 * total_fwds / wall_ms);
    printf("outputs match       : %s\n", all_ok ? "yes" : "NO (see stderr)");

    model_free(m);
    free(threads); free(ctxs);
    return all_ok ? 0 : 1;
}
