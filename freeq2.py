import numpy as np
import wave
import math

SR = 48000  # sample rate


# ---------------------------------------------------
# Basic wave + FM helpers
# ---------------------------------------------------

def sine_wave(freq, t):
    return np.sin(2 * np.pi * freq * t)


def fm_sine(fc, fm, index, t):
    """
    Classic 1-operator FM:
      phase(t) = 2π fc t + I sin(2π fm t)
    """
    return np.sin(2 * np.pi * fc * t + index * np.sin(2 * np.pi * fm * t))


def nth_positive_zero_crossing(x, n):
    """
    Return index of N-th positive-going zero crossing:
      x[i] < 0 and x[i+1] >= 0
    """
    x = np.asarray(x)
    z = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(z) < n:
        return None
    return int(z[n - 1])


# ---------------------------------------------------
# WAV writers
# ---------------------------------------------------

def write_wav_mono(path, audio, sr=SR):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def write_wav_stereo(path, audio, sr=SR):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------
# Preview 1:
# Per-segment FM: fc=f1, fm=f2, splice plain→FM at Nth zero-crossing
# ---------------------------------------------------

def build_preview1():
    """
    Use a few low Fibonacci pairs:
      (1,1), (1,2), (2,3), (3,5)
    For each:
      - carrier_plain = sine(f1)
      - carrier_fm    = FM(fc=f1, fm=f2, index=2)
      - splice at N = f2 positive zero-crossing of carrier_plain
    Concatenate all segments.
    """
    fib_pairs = [(1, 1), (1, 2), (2, 3), (3, 5)]
    segments = []

    for (f1, f2) in fib_pairs:
        n_zero = f2
        min_f = f1
        cycles = n_zero + 2
        dur = cycles / min_f     # seconds
        n_samples = int(math.ceil(dur * SR))
        t = np.arange(n_samples) / SR

        carrier_plain = sine_wave(f1, t)
        carrier_fm = fm_sine(f1, f2, index=2.0, t=t)

        cut = nth_positive_zero_crossing(carrier_plain, n_zero)
        if cut is None:
            cut = n_samples // 2

        seg = np.concatenate([carrier_plain[:cut], carrier_fm[cut:]])
        segments.append(seg)

        print(f"Segment fc={f1}Hz, fm={f2}Hz, N={n_zero}, cut={cut}, len={len(seg)}")

    return np.concatenate(segments)


# ---------------------------------------------------
# Preview 3:
# Stereo FM: Left = plain, Right = FM
# ---------------------------------------------------

def build_preview3():
    """
    Simple stereo FM demo:
      Left  = plain 220 Hz sine
      Right = FM sine (fc=220 Hz, fm=330 Hz, index=3)
      Duration = 5 seconds
    """
    dur = 5.0
    n_samples = int(SR * dur)
    t = np.arange(n_samples) / SR

    fc = 220.0
    fm = 330.0

    left = sine_wave(fc, t)
    right = fm_sine(fc, fm, index=3.0, t=t)

    stereo = np.stack([left, right], axis=-1)
    return stereo


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":
    # Preview 1: mono, Fibonacci-based plain→FM splices
    print("Building Preview 1 (mono, Fibonacci plain→FM)...")
    audio1 = build_preview1()
    write_wav_mono("fm_preview1_plain_to_fm_fib.wav", audio1)
    print("Wrote fm_preview1_plain_to_fm_fib.wav  |  duration ~", len(audio1) / SR, "seconds")

    # Preview 3: stereo, left plain / right FM
    print("Building Preview 3 (stereo, plain vs FM)...")
    audio3 = build_preview3()
    write_wav_stereo("fm_preview3_stereo_plain_vs_fm.wav", audio3)
    print("Wrote fm_preview3_stereo_plain_vs_fm.wav  |  duration ~", audio3.shape[0] / SR, "seconds")
