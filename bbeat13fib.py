import numpy as np
import wave
import math
import random

# ============================
# Global parameters
# ============================

SR = 48000              # sample rate
BPM = 140
STEPS_PER_BAR = 16      # 16th notes in 4/4
STEP_DURATION_SEC = (60.0 / BPM) / 4.0  # 16th note
BAR_DURATION_SEC = STEPS_PER_BAR * STEP_DURATION_SEC

TARGET_MINUTES = 13
TARGET_SECONDS = TARGET_MINUTES * 60

# For reproducibility but still some variation per section
MASTER_SEED = 12345
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)


# ============================
# Utility: Euclidean patterns
# ============================

def euclidean_pattern(pulses, steps):
    """Simple Euclidean rhythm generator.
    Returns a list of step indices (0..steps-1) where hits occur."""
    if pulses <= 0:
        return []
    if pulses >= steps:
        return list(range(steps))
    pattern = []
    bucket = 0
    for i in range(steps):
        bucket += pulses
        if bucket >= steps:
            bucket -= steps
            pattern.append(i)
    return pattern


# ============================
# Core Megadrive-ish drum voices
# (with harder tanh drive)
# ============================

def voice_kick():
    n = int(SR * 0.23)
    t = np.arange(n) / SR
    f0, f1 = 120.0, 45.0
    pitch = f1 + (f0 - f1) * np.exp(-t * 25.0)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    mod = np.sin(2 * np.pi * 300.0 * t) * 4.0
    sig = np.sin(phase + mod) * np.exp(-t * 18.0)
    return np.tanh(sig * 5.0).astype(np.float32)


def voice_snare():
    # Shorter, snappier snare with heavy drive
    n = int(SR * 0.22)
    t = np.arange(n) / SR
    noise = (np.random.rand(n) * 2.0 - 1.0)
    noise_env = np.exp(-t * 35.0)        # faster decay
    body = np.sin(2 * np.pi * 200.0 * t +
                  np.sin(2 * np.pi * 500.0 * t) * 5.0)
    body_env = np.exp(-t * 45.0)         # faster decay
    sig = noise * noise_env * 0.9 + body * body_env * 0.5
    return np.tanh(sig * 5.0).astype(np.float32)


def voice_hat():
    n = int(SR * 0.07)
    t = np.arange(n) / SR
    noise = np.random.rand(n) * 2.0 - 1.0
    sq = np.sign(np.sin(2 * np.pi * 6000.0 * t))
    xorish = np.sign(noise * sq)
    sig = (noise * 0.5 + xorish * 0.5) * np.exp(-t * 60.0)
    return np.tanh(sig * 5.0).astype(np.float32)


# ============================
# Extra percussion / timbral voices (A + C)
# ============================

def voice_tom_low():
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    f0, f1 = 120.0, 70.0
    pitch = f1 + (f0 - f1) * np.exp(-t * 12.0)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    mod = np.sin(2 * np.pi * 400.0 * t) * 3.0
    sig = np.sin(phase + mod) * np.exp(-t * 18.0)
    return np.tanh(sig * 4.0).astype(np.float32)


def voice_tom_mid():
    n = int(SR * 0.22)
    t = np.arange(n) / SR
    f0, f1 = 180.0, 110.0
    pitch = f1 + (f0 - f1) * np.exp(-t * 14.0)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    mod = np.sin(2 * np.pi * 500.0 * t) * 2.5
    sig = np.sin(phase + mod) * np.exp(-t * 20.0)
    return np.tanh(sig * 4.0).astype(np.float32)


def voice_clap():
    n = int(SR * 0.18)
    t = np.arange(n) / SR
    # multi-tap noise "clap"
    noise = np.random.rand(n) * 2.0 - 1.0
    env1 = np.exp(-t * 40.0)
    delayed = np.roll(noise, int(0.004 * SR))
    delayed2 = np.roll(noise, int(0.008 * SR))
    sig = (noise + delayed * 0.7 + delayed2 * 0.5) * env1
    return np.tanh(sig * 5.0).astype(np.float32)


def voice_fm_ping_high():
    n = int(SR * 0.15)
    t = np.arange(n) / SR
    base = 900.0
    mod_f = 1400.0
    mod_index = 6.0
    mod = np.sin(2 * np.pi * mod_f * t) * mod_index
    sig = np.sin(2 * np.pi * base * t + mod) * np.exp(-t * 35.0)
    return np.tanh(sig * 4.5).astype(np.float32)


def voice_fm_metal_tick():
    n = int(SR * 0.08)
    t = np.arange(n) / SR
    base = 4000.0
    mod_f = 2300.0
    mod_index = 8.0
    mod = np.sin(2 * np.pi * mod_f * t) * mod_index
    sig = np.sin(2 * np.pi * base * t + mod) * np.exp(-t * 55.0)
    return np.tanh(sig * 4.0).astype(np.float32)


def voice_fm_bell():
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    base = 600.0
    mod_f = 300.0
    mod_index = 10.0
    mod = np.sin(2 * np.pi * mod_f * t) * mod_index
    sig = np.sin(2 * np.pi * base * t + mod) * np.exp(-t * 6.0)
    return np.tanh(sig * 3.5).astype(np.float32)


# Map of layer types for A + C hybrid
LAYER_TYPES = [
    ("tom_low",      voice_tom_low,      True),   # bass-ish
    ("tom_mid",      voice_tom_mid,      False),
    ("clap",         voice_clap,         False),
    ("fm_ping_high", voice_fm_ping_high, False),
    ("fm_metal",     voice_fm_metal_tick, False),
    ("fm_bell",      voice_fm_bell,      False),
]


# ============================
# Base Fibonacci break pattern
# ============================

def fibonacci_sequence(n):
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq


def build_base_pattern():
    fib = fibonacci_sequence(10)
    kick_steps = sorted({f % STEPS_PER_BAR for f in fib})
    snare_steps = sorted({(f + 4) % STEPS_PER_BAR for f in fib})
    hat_accent = sorted({(f + 2) % STEPS_PER_BAR for f in fib})

    # ensure classic backbeat-ish snares
    for s in (4, 12):   # steps 5, 13 (1-based)
        snare_steps.append(s)
    snare_steps = sorted(set(snare_steps))

    hat_all = list(range(STEPS_PER_BAR))

    return {
        "kick": set(kick_steps),
        "snare": set(snare_steps),
        "hat_all": set(hat_all),
        "hat_accent": set(hat_accent),
    }


BASE_PATTERN = build_base_pattern()


# ============================
# Mixing helpers
# ============================

def place_stereo(mix_l, mix_r, sample, start_idx, pan=0.0, gain=1.0):
    """Place mono sample into stereo buffer with equal-power panning.
    pan: -1 = left, 0 = centre, +1 = right"""
    if gain == 0.0:
        return
    if start_idx >= len(mix_l):
        return
    end_idx = min(start_idx + len(sample), len(mix_l))
    seg = sample[: end_idx - start_idx] * gain

    angle = (pan + 1.0) * math.pi / 4.0
    lg = math.cos(angle)
    rg = math.sin(angle)

    mix_l[start_idx:end_idx] += seg * lg
    mix_r[start_idx:end_idx] += seg * rg


# ============================
# Layer generation per FULL section
# ============================

def build_layers_for_section(section_index, fib_number, bars_in_section):
    """For each FULL section:
    - Create fib_number new layers.
    - Each layer chooses a type from LAYER_TYPES (A + C hybrid).
    - Each layer has its own Euclidean pattern & bar-wise rotation.
    Returns list of dicts describing layers:
      { "sample": mono_array, "pattern_steps_per_bar": [...],
        "pan": float, "gain": float, "rotate_per_bar": int }"""
    layers = []
    if fib_number <= 0:
        return layers

    for layer_idx in range(fib_number):
        # Deterministic but unique-ish seed per section+layer
        layer_seed = MASTER_SEED + section_index * 1000 + layer_idx
        rng = random.Random(layer_seed)

        layer_type_name, layer_fn, is_bassish = LAYER_TYPES[layer_idx % len(LAYER_TYPES)]

        sample = layer_fn()

        # pulses in 16 steps: between 2 and 8 for variety
        pulses = rng.randint(2, 8)
        base_steps = euclidean_pattern(pulses, STEPS_PER_BAR)

        # bar-wise rotation to create evolving feel
        rotate_per_bar = rng.choice([1, 2, 3, 4, 5, 7])
        # random pan slightly biased by type
        if is_bassish:
            pan = rng.uniform(-0.2, 0.2)
        else:
            pan = rng.uniform(-0.9, 0.9)
        gain = rng.uniform(0.18, 0.35)

        layers.append({
            "name": layer_type_name,
            "sample": sample,
            "is_bassish": is_bassish,
            "base_steps": base_steps,
            "rotate_per_bar": rotate_per_bar,
            "pan": pan,
            "gain": gain,
            "seed": layer_seed,
        })

    return layers


# ============================
# Section plan: Fibonacci full/lull until >= TARGET_SECONDS
# ============================

def build_section_plan():
    """Builds sequence: (bars, mode, fib_number_for_this_section),
    with mode alternating FULL / LULL, and fib_number = that Fibonacci number.
    Stops once total duration >= TARGET_SECONDS."""
    fib = [1, 1]
    sections = []
    total_time = 0.0
    i = 0

    while total_time < TARGET_SECONDS:
        if i >= len(fib):
            fib.append(fib[-1] + fib[-2])

        fib_n = fib[i]
        bars = fib_n
        mode = "full" if i % 2 == 0 else "lull"  # even index: full, odd: lull

        duration = bars * BAR_DURATION_SEC
        sections.append((bars, mode, fib_n))
        total_time += duration
        i += 1

    return sections


# ============================
# Render full track
# ============================

def render_track():
    # Pre-generate core drums
    kick = voice_kick()
    snare = voice_snare()
    hat = voice_hat()

    sections = build_section_plan()

    all_l = []
    all_r = []

    for sec_index, (bars, mode, fib_n) in enumerate(sections):
        print(f"Rendering section {sec_index}: {bars} bars, mode={mode}, Fn={fib_n}")

        n_samples_section = int(round(bars * BAR_DURATION_SEC * SR))
        mix_l = np.zeros(n_samples_section, dtype=np.float32)
        mix_r = np.zeros(n_samples_section, dtype=np.float32)

        # Build extra layers only for FULL sections (A + C hybrid)
        if mode == "full":
            layers = build_layers_for_section(sec_index, fib_n, bars)
        else:
            layers = []

        # Sequence steps
        total_steps = bars * STEPS_PER_BAR

        for step in range(total_steps):
            bar_idx = step // STEPS_PER_BAR
            local_step = step % STEPS_PER_BAR
            start_sample = int(round(step * STEP_DURATION_SEC * SR))

            # Base patterns:
            # Kick only in full; snare & hats in both
            if mode == "full":
                if local_step in BASE_PATTERN["kick"]:
                    place_stereo(mix_l, mix_r, kick, start_sample, pan=0.0, gain=0.9)

            if local_step in BASE_PATTERN["snare"]:
                place_stereo(mix_l, mix_r, snare, start_sample, pan=0.2, gain=0.85)

            if local_step in BASE_PATTERN["hat_all"]:
                accent = local_step in BASE_PATTERN["hat_accent"]
                g = 0.35 if accent else 0.2
                place_stereo(mix_l, mix_r, hat, start_sample, pan=-0.3, gain=g)

            # Extra layers for this full section
            for lyr in layers:
                base_steps = lyr["base_steps"]
                rotate = lyr["rotate_per_bar"]
                rotated_steps = [(s + bar_idx * rotate) % STEPS_PER_BAR for s in base_steps]
                if local_step in rotated_steps:
                    place_stereo(mix_l, mix_r, lyr["sample"], start_sample,
                                 pan=lyr["pan"], gain=lyr["gain"])

        all_l.append(mix_l)
        all_r.append(mix_r)

    # Concatenate all sections
    full_l = np.concatenate(all_l)
    full_r = np.concatenate(all_r)
    stereo = np.stack([full_l, full_r], axis=1)

    # Normalise & gentle master drive
    max_val = np.max(np.abs(stereo)) + 1e-9
    stereo = stereo / max_val * 0.95
    stereo = np.tanh(stereo * 2.0)

    return stereo


# ============================
# WAV writer
# ============================

def write_wav_stereo(filename, data, samplerate=SR):
    data = np.clip(data, -1.0, 1.0)
    data_i16 = (data * 32767.0).astype(np.int16)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(data_i16.tobytes())


if __name__ == "__main__":
    print("Building Fibonacci-structured, layered breakbeat...")
    stereo_data = render_track()
    duration = stereo_data.shape[0] / SR
    print(f"Rendered duration: {duration:.2f} seconds (~{duration/60:.2f} minutes)")
    out_name = "fibonacci_megadrive_fib_layers_13min.wav"
    write_wav_stereo(out_name, stereo_data)
    print(f"Written to {out_name}")
