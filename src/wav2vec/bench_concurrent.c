// Throughput / latency benchmark for concurrent model_forward.
//
// Each "worker" pthread runs N model_forward calls on a shared model_t, each
// call timed independently. At the end we aggregate per-forward latencies
// across all workers and report p50, p95, p99, plus aggregate throughput.
//
// Usage:
//   ./bench_concurrent <weights_dir>          [sweep]  -- runs a default sweep
//   ./bench_concurrent <weights_dir> <W> <I> <ITERS>   -- one config

#include "wav2vec_vm.h"
#include "sgemm.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *CASES[] = {
    "samples/voicemail/Doo-Wop_Dudes.f32",
    "samples/human/tim-griffith-messages-02.f32",
    "samples/voicemail/Dial_and_Complain.f32",
    "samples/voicemail/The_Hard_Boiled_Private_Eye.f32",
    "samples/human/tim-griffith-messages-01.f32",
    "samples/voicemail/Dinner_For_Two.f32",
    "samples/voicemail/Dr_Feel_Good.f32",
    "samples/voicemail/The_Rap_Master_Machine.f32",
};
#define NCASES (sizeof(CASES) / sizeof(CASES[0]))

typedef struct {
    const model_t *model;
    int worker_id;
    int iters;
    int intra_threads;
    double *lat_ms;   // size == iters
} ctx_t;

static double monotonic_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void *worker(void *arg) {
    ctx_t *c = (ctx_t *)arg;
    sgemm_set_threads(c->intra_threads);

    // Each worker gets one audio buffer round-robin from the case list.
    float audio[INPUT_SAMPLES];
    const char *path = CASES[c->worker_id % NCASES];
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    size_t n = fread(audio, sizeof(float), INPUT_SAMPLES, f);
    fclose(f);
    if (n < INPUT_SAMPLES) memset(audio + n, 0, (INPUT_SAMPLES - n) * sizeof(float));

    float logits[NUM_LABELS], probs[NUM_LABELS];
    forward_scratch_t *s = forward_scratch_new();
    // One warmup forward — first call pays pool creation + cold cache.
    model_forward(c->model, s, audio, logits, probs);

    for (int i = 0; i < c->iters; i++) {
        double t0 = monotonic_ms();
        model_forward(c->model, s, audio, logits, probs);
        c->lat_ms[i] = monotonic_ms() - t0;
    }
    forward_scratch_free(s);
    return NULL;
}

static int cmp_double(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

static double pct(double *sorted, int n, double p) {
    if (n == 0) return 0.0;
    double r = p * (n - 1);
    int lo = (int)r, hi = lo + 1;
    if (hi >= n) hi = n - 1;
    double f = r - lo;
    return sorted[lo] * (1 - f) + sorted[hi] * f;
}

static void run_config(const model_t *m, int W, int intra, int iters,
                       int print_header) {
    pthread_t *th = calloc(W, sizeof(pthread_t));
    ctx_t *ctxs = calloc(W, sizeof(ctx_t));
    double *all_lat = malloc((size_t)W * iters * sizeof(double));

    for (int i = 0; i < W; i++) {
        ctxs[i] = (ctx_t){m, i, iters, intra, all_lat + (size_t)i * iters};
    }

    double t0 = monotonic_ms();
    for (int i = 0; i < W; i++) pthread_create(&th[i], NULL, worker, &ctxs[i]);
    for (int i = 0; i < W; i++) pthread_join(th[i], NULL);
    double wall_ms = monotonic_ms() - t0;

    int total = W * iters;
    qsort(all_lat, total, sizeof(double), cmp_double);
    double p50 = pct(all_lat, total, 0.50);
    double p95 = pct(all_lat, total, 0.95);
    double p99 = pct(all_lat, total, 0.99);
    double mn = all_lat[0], mx = all_lat[total - 1];
    double tput = 1000.0 * total / wall_ms;

    if (print_header) {
        printf("\n%3s %5s  %6s  %8s %8s %8s %8s %8s  %8s\n",
               "W", "intra", "iters", "p50", "p95", "p99", "min", "max", "fwd/s");
        printf("----------------------------------------------------------------------------------\n");
    }
    printf("%3d %5d  %6d  %8.1f %8.1f %8.1f %8.1f %8.1f  %8.2f\n",
           W, intra, total, p50, p95, p99, mn, mx, tput);

    free(th); free(ctxs); free(all_lat);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <weights_dir> [W intra iters | sweep]\n", argv[0]);
        return 2;
    }
    const char *weights_dir = argv[1];
    printf("loading %s...\n", weights_dir);
    model_t *m = model_load(weights_dir);

    if (argc >= 5) {
        int W = atoi(argv[2]), intra = atoi(argv[3]), iters = atoi(argv[4]);
        run_config(m, W, intra, iters, 1);
    } else {
        // Default sweep. Tuned to finish in a couple of minutes on Apple Silicon.
        int iters_main = 10;
        int print_hdr = 1;

        printf("\n=== throughput sweep: W workers x 1 intra-op thread each ===\n");
        int ws[] = {1, 2, 4, 6, 8, 12, 16};
        for (size_t i = 0; i < sizeof(ws)/sizeof(ws[0]); i++) {
            run_config(m, ws[i], 1, iters_main, print_hdr);
            print_hdr = 0;
        }

        print_hdr = 1;
        printf("\n=== latency sweep: 1 worker x I intra-op threads ===\n");
        int is[] = {1, 2, 4, 8};
        for (size_t i = 0; i < sizeof(is)/sizeof(is[0]); i++) {
            run_config(m, 1, is[i], iters_main, print_hdr);
            print_hdr = 0;
        }

        print_hdr = 1;
        printf("\n=== hybrid: W x I where W*I stays near core count ===\n");
        int wi[][2] = {{2, 4}, {4, 2}, {2, 2}, {3, 3}, {4, 4}};
        for (size_t i = 0; i < sizeof(wi)/sizeof(wi[0]); i++) {
            run_config(m, wi[i][0], wi[i][1], iters_main, print_hdr);
            print_hdr = 0;
        }
    }

    model_free(m);
    return 0;
}
