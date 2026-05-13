# mod_vm_detect

FreeSWITCH module that classifies speech segments as **human** or **voicemail**
using a fine-tuned wav2vec2 model. Segmentation is driven by FireRed streaming
VAD; classification runs on the resulting 2 s speech window.

See [PLAN.md](PLAN.md) for the design rationale.

## Pipeline

```text
native codec (8/16/48 kHz) ─┐
                            ├─► FireRed VAD (streaming 10 ms frames)
       speex resampler      │         │
                            └──► 16 kHz L16 mono ─► ring buffer
                                                    │
                                speech_end ────────►│
                                                    ▼
                                wav2vec2 (W8A8) forward (~250–500 ms)
                                                    │
                                                    ▼
                                        CUSTOM vm_detect::result event
```

## Build

Dependencies:

- FreeSWITCH (with `freeswitch.pc` visible to `pkg-config`)
- `libspeexdsp`
- CMake ≥ 3.15, a C11 compiler

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Optional: build the vendored wav2vec test + bench binaries
cmake -B build -DBUILD_WAV2VEC_TESTS=ON
cmake --build build -j
./build/test_concurrent weights/weights-int8 4
```

`-DENABLE_LOCAL=ON` adds `/usr/local/freeswitch/lib/pkgconfig` to the
`PKG_CONFIG_PATH` for FreeSWITCH installs from source.

CMake auto-detects SIMD features at configure time:

- **ARM**: `dotprod` baseline, opts into `i8mm` (SMMLA) if present (macOS
  sysctl / Linux `/proc/cpuinfo`).
- **x86-64**: `AVX2 + FMA` on Haswell+, `AVX-VNNI` when present.

## Install

```sh
sudo cmake --install build
```

Installs:

- `mod_vm_detect.so` → FreeSWITCH modules dir
- `vm_detect.conf.xml` → `$CONFDIR/autoload_configs/`

The module does **not** install weights. Copy them manually:

```sh
sudo mkdir -p /usr/local/freeswitch/share/freeswitch/vm_detect
sudo cp -r weights/weights-int8            /usr/local/freeswitch/share/freeswitch/vm_detect/
sudo cp weights/fireredvad/fireredvad.bin  /usr/local/freeswitch/share/freeswitch/vm_detect/
sudo cp weights/fireredvad/fireredvad.json /usr/local/freeswitch/share/freeswitch/vm_detect/
```

Adjust `weights-path`, `fireredvad-weights-path`, `fireredvad-cmvn-path` in
`vm_detect.conf.xml` if your data dir differs.

Load the module:

```text
load mod_vm_detect
```

## Dialplan usage

Basic AMD (answering-machine-detect) replacement — attach the detector on
answer, bridge to the carrier, listen on the event bus:

```xml
<extension name="amd_example">
  <condition field="destination_number" expression="^amd_(.+)$">
    <action application="answer"/>
    <action application="vm_detect_start"/>
    <action application="bridge" data="sofia/gateway/pstn/$1"/>
  </condition>
</extension>
```

When a speech segment is classified the module:

- fires a `CUSTOM vm_detect::result` event
- sets channel variables `vm_detect_label`, `vm_detect_p_human`, `vm_detect_p_voicemail`
- updates the `vm_detect <uuid> status` API output

Event headers:

```text
Event-Name:            CUSTOM
Event-Subclass:        vm_detect::result
Unique-ID:             <session uuid>
VM-Detect-Label:       human | voicemail | uncertain
VM-Detect-P-Human:     0.034
VM-Detect-P-Voicemail: 0.966
VM-Detect-Segment-Ms:  2340
VM-Detect-Latency-Ms:  512
VM-Detect-Window:      first
```

## API surface

| Form | Signature | Purpose |
| --- | --- | --- |
| application | `vm_detect_start` | Attach detector to current channel |
| application | `vm_detect_stop` | Detach detector |
| api | `vm_detect <uuid> start\|stop\|status` | Control + query last result |

`vm_detect <uuid> status` output:

```text
+OK label=voicemail p_human=0.034 p_voicemail=0.966 latency_ms=512 results=3
```

## Config

All params in `vm_detect.conf.xml`:

| Param | Default | Purpose |
| --- | --- | --- |
| `weights-path` | `$DATA/vm_detect/weights-int8` | wav2vec2 classifier weights dir |
| `fireredvad-weights-path` | `$DATA/vm_detect/fireredvad.bin` | VAD weights |
| `fireredvad-cmvn-path` | `$DATA/vm_detect/fireredvad.json` | VAD CMVN stats |
| `window-policy` | `first` | `first`, `last`, or `average` of 2 s windows for segments > 2 s |
| `min-confidence` | `0.0` | If `max(probs) < this`, label is `uncertain` |
| `intra-op-threads` | `1` | Per-forward sgemm threads (keep at 1 for max concurrency) |
| `emit-on-speech-end` | `true` | Emit `vm_detect::result` automatically |

## Weights

Two paths, depending on whether you want the pre-built binary blobs or to
regenerate them from the HuggingFace checkpoint.

### Included in this repo

- `weights/weights-int8/` — W8A8 (int8 weights + dynamic int8 activations), ~327 MB.
- `weights/weights-fp32/` — fp32 reference, ~1.2 GB (useful for accuracy sanity).
- `weights/fireredvad/fireredvad.bin`, `fireredvad.json` — FireRed streaming VAD.

### Regenerate from HF

```sh
cd tools
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python extract_weights.py --out ../weights/weights-int8 --quantize int8
```

See `tools/extract_weights.py --help` for the fp32 path and other options.

## Testing the inference engine

The vendored wav2vec engine ships with a concurrency test and a micro-bench:

```sh
cmake -B build -DBUILD_WAV2VEC_TESTS=ON
cmake --build build -j
./build/test_concurrent weights/weights-int8 4     # 4 workers, correctness check
./build/bench_concurrent weights/weights-int8 8    # throughput probe
```

Sample .f32 files for the concurrency test live under
`tools/samples/{human,voicemail}/`.

## License

See [LICENSE](LICENSE). The vendored FireRed VAD code and wav2vec engine
retain their upstream licenses — see `fireredvad.c`, `src/wav2vec/*.c`.
