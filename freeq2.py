import numpy as np
import wave
import math

SR = 48000        # sample rate
NYQUIST = SR / 2  # 24000 Hz


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
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("audio must be shape (n_samples, 2)")
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------
# FULL LENGTH – Mode 1:
# Mono Fibonacci FM splice: plain → FM at N-th zero crossing
# ---------------------------------------------------

def build_full_mode1_plain_to_fm(index=2.0):
    """
    For each Fibonacci pair (f1, f2) where max(f1,f2) <= Nyquist:
      - carrier_plain = sine(f1)
      - carrier_fm    = FM(fc=f1, fm=f2, I=index)
      - splice at N = f2 positive zero-crossing of carrier_plain
    Concatenate all segments.
    """
    fib = [1, 1]
    segments = []
    step = 0

    while True:
        f1, f2 = fib[-2], fib[-1]
        if max(f1, f2) > NYQUIST:
            break

        n_zero = f2
        min_f = f1 if f1 > 0 else 1.0
        cycles = n_zero + 2
        dur = cycles / min_f
        n_samples = int(math.ceil(dur * SR))
        t = np.arange(n_samples) / SR

        carrier_plain = sine_wave(f1, t)
        carrier_fm = fm_sine(f1, f2, index=index, t=t)

        cut = nth_positive_zero_crossing(carrier_plain, n_zero)
        if cut is None:
            cut = n_samples // 2

        seg = np.concatenate([carrier_plain[:cut], carrier_fm[cut:]])
        segments.append(seg)

        step += 1
        print(
            f"[Mode1] Step {step:3d}: fc={f1:8.1f} Hz, fm={f2:8.1f} Hz, "
            f"N={n_zero}, cut_idx={cut}, seg_len={len(seg)}"
        )

        fib.append(f1 + f2)

    return np.concatenate(segments)


# ---------------------------------------------------
# FULL LENGTH – Mode 3:
# Stereo Fibonacci FM: Left=plain, Right=FM
# ---------------------------------------------------

def build_full_mode3_stereo_plain_vs_fm(index=3.0):
    """
    For each Fibonacci pair (f1, f2) where max(f1,f2) <= Nyquist:
      - Left  = plain sine at fc = f1
      - Right = FM sine  at fc = f1, fm = f2, I = index
      - Same duration logic as Mode 1 (n_zero + 2 cycles of slower freq)

    All segments concatenated into one stereo file.
    """
    fib = [1, 1]
    stereo_segments = []
    step = 0

    while True:
        f1, f2 = fib[-2], fib[-1]
        if max(f1, f2) > NYQUIST:
            break

        n_zero = f2
        min_f = f1 if f1 > 0 else 1.0
        cycles = n_zero + 2
        dur = cycles / min_f
        n_samples = int(math.ceil(dur * SR))
        t = np.arange(n_samples) / SR

        left = sine_wave(f1, t)
        right = fm_sine(f1, f2, index=index, t=t)

        stereo = np.stack([left, right], axis=-1)
        stereo_segments.append(stereo)

        step += 1
        print(
            f"[Mode3] Step {step:3d}: fc={f1:8.1f} Hz, fm={f2:8.1f} Hz, "
            f"N={n_zero}, seg_len={n_samples}"
        )

        fib.append(f1 + f2)

    return np.concatenate(stereo_segments, axis=0)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":
    # Mode 1: mono full-length Fibonacci FM splice
    print("Building full-length Mode 1 (mono plain→FM, Fibonacci)…")
    audio1 = build_full_mode1_plain_to_fm(index=2.0)
    write_wav_mono("fm_full1_plain_to_fm_fib.wav", audio1)
    print("Wrote fm_full1_plain_to_fm_fib.wav  |  duration ~", len(audio1) / SR, "seconds")

    # Mode 3: stereo full-length Fibonacci FM (plain vs FM)
    print("Building full-length Mode 3 (stereo plain vs FM, Fibonacci)…")
    audio3 = build_full_mode3_stereo_plain_vs_fm(index=3.0)
    write_wav_stereo("fm_full3_stereo_plain_vs_fm.wav", audio3)
    print("Wrote fm_full3_stereo_plain_vs_fm.wav  |  duration ~", audio3.shape[0] / SR, "seconds")

