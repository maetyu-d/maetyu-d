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
# AD envelope
# ---------------------------------------------------

def ad_envelope(n_samples, sr=SR, attack=0.01, decay=0.1):
    """
    Simple AD envelope:
      0 -> 1 over 'attack' seconds
      1 -> 0 over 'decay' seconds
      then stays at 0.
    """
    env = np.zeros(n_samples, dtype=np.float32)

    a_samps = int(attack * sr)
    d_samps = int(decay * sr)

    # Attack
    if a_samps > 0:
        a_len = min(a_samps, n_samples)
        env[:a_len] = np.linspace(0.0, 1.0, a_len, endpoint=False)

    # Decay
    start_d = a_samps
    end_d = a_samps + d_samps
    if start_d < n_samples:
        end_d = min(end_d, n_samples)
        if end_d > start_d:
            env[start_d:end_d] = np.linspace(1.0, 0.0, end_d - start_d, endpoint=False)

    return env


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
# with AD envelope per segment
# ---------------------------------------------------

def build_full_mode1_plain_to_fm(index=2.0, attack=0.01, decay=0.1):
    """
    For each Fibonacci pair (f1, f2) where max(f1,f2) <= Nyquist:
      - carrier_plain = sine(f1)
      - carrier_fm    = FM(fc=f1, fm=f2, I=index)
      - splice at N = f2 positive zero-crossing of carrier_plain
      - apply AD envelope to the segment
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

        env = ad_envelope(len(seg), sr=SR, attack=attack, decay=decay)
        seg *= env

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
# with AD envelope per segment
# ---------------------------------------------------

def build_full_mode3_stereo_plain_vs_fm(index=3.0, attack=0.01, decay=0.1):
    """
    For each Fibonacci pair (f1, f2) where max(f1,f2) <= Nyquist:
      - Left  = plain sine at fc = f1
      - Right = FM sine  at fc = f1, fm = f2, I = index
      - duration = (n_zero + 2) cycles of fc
      - apply AD envelope to both channels
    Concatenate all segments.
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

        env = ad_envelope(n_samples, sr=SR, attack=attack, decay=decay)
        stereo *= env[:, None]

        stereo_segments.append(stereo)

        step += 1
        print(
            f"[Mode3] Step {step:3d}: fc={f1:8.1f} Hz, fm={f2:8.1f} Hz, "
            f"N={n_zero}, seg_len={n_samples}"
        )

        fib.append(f1 + f2)

    return np.concatenate(stereo_segments, axis=0)


# ---------------------------------------------------
# Utility: tile / repeat to target length
# ---------------------------------------------------

def tile_to_length(audio, target_len):
    """
    Repeat (tile) 1D or 2D audio along time axis until at least target_len,
    then trim to exactly target_len.
    """
    audio = np.asarray(audio)
    n = audio.shape[0]
    if n == 0:
        raise ValueError("audio length is zero")
    reps = target_len // n + 1
    if audio.ndim == 1:
        tiled = np.tile(audio, reps)
    else:
        tiled = np.tile(audio, (reps, 1))
    return tiled[:target_len]


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":
    target_minutes = 27.0
    target_samples = int(round(target_minutes * 60.0 * SR))
    print(f"Target: {target_minutes} minutes @ {SR} Hz = {target_samples} samples")

    # --- build raw pieces ---
    print("Building full-length Mode 1 (mono plain→FM, Fibonacci, AD env)…")
    audio1 = build_full_mode1_plain_to_fm(index=2.0, attack=0.01, decay=0.1)
    write_wav_mono("fm_full1_plain_to_fm_fib_AD_raw.wav", audio1)
    print("Raw Mode1 length (samples):", len(audio1),
          "=> seconds:", len(audio1) / SR)

    print("Building full-length Mode 3 (stereo plain vs FM, Fibonacci, AD env)…")
    audio3 = build_full_mode3_stereo_plain_vs_fm(index=3.0, attack=0.01, decay=0.1)
    write_wav_stereo("fm_full3_stereo_plain_vs_fm_AD_raw.wav", audio3)
    print("Raw Mode3 length (samples):", audio3.shape[0],
          "=> seconds:", audio3.shape[0] / SR)

    # --- overlay to exactly 27 minutes ---

    # Upmix mono Mode1 to stereo
    stereo1 = np.stack([audio1, audio1], axis=-1)

    # Tile both to target length
    stereo1_t = tile_to_length(stereo1, target_samples)
    stereo3_t = tile_to_length(audio3, target_samples)

    print("Tiled Mode1 length (samples):", stereo1_t.shape[0],
          "=> seconds:", stereo1_t.shape[0] / SR)
    print("Tiled Mode3 length (samples):", stereo3_t.shape[0],
          "=> seconds:", stereo3_t.shape[0] / SR)

    # Overlay (sum)
    overlay = stereo1_t + stereo3_t

    # Soft normalize
    peak = np.max(np.abs(overlay))
    if peak > 0:
        overlay *= (0.95 / peak)

    print("Overlay length (samples):", overlay.shape[0],
          "=> seconds:", overlay.shape[0] / SR)

    write_wav_stereo("fm_27min_overlay_AD.wav", overlay, sr=SR)
    print("Wrote fm_27min_overlay_AD.wav")

