"""
Preprocess audio files (mp3/wav/flac/...) into the raw float32 format that
c/vm_detect expects: 32,000 samples at 16 kHz mono, zero-padded if shorter.

Usage:
  python prep_audio.py <input1> [<input2> ...]
      Writes each input with a .f32 extension next to the original.

  python prep_audio.py --stdout <input>
      Writes one sample to stdout (pipe into vm_detect).
"""

import sys
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 2


def load_clip(path: str) -> np.ndarray:
    a, _ = librosa.load(path, sr=SAMPLE_RATE)
    a = a[:NUM_SAMPLES]
    if len(a) < NUM_SAMPLES:
        a = np.pad(a, (0, NUM_SAMPLES - len(a)))
    return a.astype(np.float32)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    if args[0] == "--stdout":
        if len(args) != 2:
            print("--stdout takes exactly one input file", file=sys.stderr)
            sys.exit(2)
        clip = load_clip(args[1])
        sys.stdout.buffer.write(clip.tobytes())
        return

    for src in args:
        dst = Path(src).with_suffix(".f32")
        load_clip(src).tofile(dst)
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
