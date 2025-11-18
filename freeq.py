import numpy as np
import wave

SR = 48000  # 48 kHz
NYQUIST = SR / 2


# ------------------------------------------------------
# Basic waveforms
# ------------------------------------------------------
def sine_wave(freq, t):
    return np.sin(2 * np.pi * freq * t)

def square_wave(freq, t):
    return np.sign(np.sin(2 * np.pi * freq * t))

def saw_wave(freq, t):
    return 2 * ((freq * t) % 1.0) - 1.0

def tri_wave(freq, t):
    ph = (freq * t) % 1.0
    return 2 * np.abs(2 * ph - 1) - 1


WAVES = {
    "sine": sine_wave,
    "square": square_wave,
    "saw": saw_wave,
    "triangle": tri_wave,
}


# ------------------------------------------------------
# N-th positive going zero crossing
# ------------------------------------------------------
def nth_positive_zero_crossing(x, n):
    z = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(z) < n:
        return None
    return int(z[n - 1])


# ------------------------------------------------------
# Create the mono Fibonacci splice signal
# ------------------------------------------------------
def build_fib_splice_mono(sr=SR, seed=2):
    rng = np.random.default_rng(seed)

    fib = [1, 1]
    segments = []

    while True:
        f1, f2 = fib[-2], fib[-1]

        if max(f1, f2) > NYQUIST:
            break

        n_zero = f2

        # choose two different waveforms
        keys = list(WAVES.keys())
        wa = rng.choice(keys)
        wb = rng.choice([k for k in keys if k != wa])

        # enough time to include n_zero crossings
        min_f = min(f1, f2)
        cycles = n_zero + 2
        dur = cycles / min_f
        n = int(dur * sr)
        t = np.arange(n) / sr

        xa = WAVES[wa](f1, t)
        xb = WAVES[wb](f2, t)

        cut = nth_positive_zero_crossing(xa, n_zero)
        if cut is None:
            cut = n // 2

        seg = np.concatenate([xa[:cut], xb[cut:]])
        segments.append(seg)

        fib.append(f1 + f2)

    return np.concatenate(segments)


# ------------------------------------------------------
# Build stereo cumulative Fibonacci drift (Option C)
# ------------------------------------------------------
def build_stereo_fib_drift(mono, sr=SR):
    L = len(mono)
    fib = [1, 1]
    stereo_segments = []

    print("Mono length:", L, "samples")

    while True:
        offset = fib[-1]

        right = np.roll(mono, offset % L)
        stereo = np.column_stack([mono, right])
        stereo_segments.append(stereo)

        fib.append(fib[-1] + fib[-2])

        # stop when channels re-align
        if fib[-1] % L == 0:
            print("Re-alignment reached at Fibonacci =", fib[-1])
            break

        if fib[-1] > 10**9:
            print("Stopping early (very large Fibonacci).")
            break

    return np.concatenate(stereo_segments, axis=0)


# ------------------------------------------------------
# WAV writer
# ------------------------------------------------------
def write_wav_stereo(name, audio, sr=SR):
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(name, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    print("Building mono Fibonacci splice signal...")
    mono = build_fib_splice_mono()

    print("Building stereo drift...")
    stereo = build_stereo_fib_drift(mono)

    out = "fib_stereo_drift.wav"
    write_wav_stereo(out, stereo)

    print("Wrote:", out)
    print("Duration (sec):", len(stereo) / SR)
