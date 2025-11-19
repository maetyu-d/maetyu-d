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
# Layer generation per FULL section (rhythm)
# ============================

def build_layers_for_section(section_index, fib_number, bars_in_section):
    """For each FULL section:
    - Create fib_number new rhythmic layers.
    - Each layer chooses a type from LAYER_TYPES (A + C hybrid).
    - Each layer has its own Euclidean pattern & bar-wise rotation.
    Returns list of dicts describing layers."""
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
# Drum rendering (as before)
# ============================

def render_drums(sections, total_samples):
    kick = voice_kick()
    snare = voice_snare()
    hat = voice_hat()

    full_l = np.zeros(total_samples, dtype=np.float32)
    full_r = np.zeros(total_samples, dtype=np.float32)

    # compute section start offsets in samples
    bar_cursor = 0
    for sec_index, (bars, mode, fib_n) in enumerate(sections):
        section_start_bar = bar_cursor
        section_start_sample = int(round(section_start_bar * BAR_DURATION_SEC * SR))
        section_samples = int(round(bars * BAR_DURATION_SEC * SR))

        print(f"Drums section {sec_index}: {bars} bars, mode={mode}, Fn={fib_n}")

        mix_l = np.zeros(section_samples, dtype=np.float32)
        mix_r = np.zeros(section_samples, dtype=np.float32)

        if mode == "full":
            layers = build_layers_for_section(sec_index, fib_n, bars)
        else:
            layers = []

        total_steps = bars * STEPS_PER_BAR

        for step in range(total_steps):
            bar_idx = step // STEPS_PER_BAR
            local_step = step % STEPS_PER_BAR
            local_start = int(round(step * STEP_DURATION_SEC * SR))

            # Base patterns:
            if mode == "full":
                if local_step in BASE_PATTERN["kick"]:
                    place_stereo(mix_l, mix_r, kick, local_start, pan=0.0, gain=0.9)

            if local_step in BASE_PATTERN["snare"]:
                place_stereo(mix_l, mix_r, snare, local_start, pan=0.2, gain=0.85)

            if local_step in BASE_PATTERN["hat_all"]:
                accent = local_step in BASE_PATTERN["hat_accent"]
                g = 0.35 if accent else 0.2
                place_stereo(mix_l, mix_r, hat, local_start, pan=-0.3, gain=g)

            # Extra layers for this full section
            for lyr in layers:
                base_steps = lyr["base_steps"]
                rotate = lyr["rotate_per_bar"]
                rotated_steps = [(s + bar_idx * rotate) % STEPS_PER_BAR for s in base_steps]
                if local_step in rotated_steps:
                    place_stereo(mix_l, mix_r, lyr["sample"], local_start,
                                 pan=lyr["pan"], gain=lyr["gain"])

        end_idx = min(section_start_sample + section_samples, total_samples)
        seg_len = end_idx - section_start_sample
        full_l[section_start_sample:end_idx] += mix_l[:seg_len]
        full_r[section_start_sample:end_idx] += mix_r[:seg_len]

        bar_cursor += bars

    return full_l, full_r


# ============================
# Melodic voices (Aphex-y sad leads)
# ============================

# D natural minor scale degrees (relative semitones from root)
SCALE_DEGREES = [0, 2, 3, 5, 7, 10]  # D, E, F, G, A, C
BASE_MIDI_ROOT = 50  # D3-ish


def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def synth_melody_notes(mix_l, mix_r, start_sample, end_sample,
                       seed, linespec):
    """
    Generates a single melodic line between [start_sample, end_sample).
    - mix_l, mix_r: stereo buffers to add into
    - linespec: dict with melodic parameters
    """
    rng = random.Random(seed)
    duration_samples = max(0, end_sample - start_sample)
    if duration_samples <= 0:
        return

    duration_sec = duration_samples / SR

    note_period_steps = linespec["note_period_steps"]
    note_dur_steps = linespec["note_dur_steps"]
    timbre = linespec["timbre"]
    pan = linespec["pan"]
    gain = linespec["gain"]

    note_period_sec = note_period_steps * STEP_DURATION_SEC
    note_dur_sec = note_dur_steps * STEP_DURATION_SEC

    n_notes = int(duration_sec / note_period_sec) + 1
    current_scale_idx = rng.randint(0, len(SCALE_DEGREES) - 1)
    current_oct = rng.choice([-1, 0, 0, 1])  # bias mid, occasional jump

    base_midi = BASE_MIDI_ROOT

    for i in range(n_notes):
        note_start_sec = i * note_period_sec
        global_note_start = start_sample + int(round(note_start_sec * SR))
        if global_note_start >= end_sample:
            break

        # random walk on scale
        step_choice = rng.choice([-2, -1, -1, 0, 1, 1, 2])
        current_scale_idx = (current_scale_idx + step_choice) % len(SCALE_DEGREES)
        # occasional octave shifts
        if rng.random() < 0.12:
            current_oct += rng.choice([-1, 1])
            current_oct = max(-2, min(2, current_oct))

        midi_note = base_midi + SCALE_DEGREES[current_scale_idx] + 12 * current_oct + rng.choice([0, 0, 12])  # sometimes up an octave
        freq = midi_to_freq(midi_note)

        this_note_dur_sec = note_dur_sec * rng.uniform(0.7, 1.4)
        this_note_samp = int(this_note_dur_sec * SR)
        if this_note_samp <= 0:
            continue

        global_note_end = min(global_note_start + this_note_samp, end_sample)
        note_len = global_note_end - global_note_start
        if note_len <= 0:
            continue

        t = np.arange(note_len) / SR

        # timbre
        if timbre == "soft_sine_tri":
            base_wave = 0.6 * np.sin(2 * np.pi * freq * t) + 0.4 * np.sign(
                np.sin(2 * np.pi * freq * t)
            )
        elif timbre == "glass_fm":
            mod_f = freq * 2.5
            mod_index = 4.0
            mod = np.sin(2 * np.pi * mod_f * t) * mod_index
            base_wave = np.sin(2 * np.pi * freq * t + mod)
        else:  # airy-pad-ish
            mod_f = freq * 1.01
            mod_index = 2.0
            mod = np.sin(2 * np.pi * mod_f * t) * mod_index
            base_wave = 0.5 * np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * t + mod)

        # envelope: slowish attack, long tail, sad/floaty
        atk = 0.02
        env = np.exp(-t * 2.5)  # relatively long decay
        atk_samples = max(1, int(atk * SR))
        env[:atk_samples] *= np.linspace(0.0, 1.0, atk_samples)

        # small random tremolo for fragility
        trem = 1.0 + 0.12 * np.sin(2 * np.pi * rng.uniform(3.0, 6.0) * t + rng.random() * 2 * np.pi)
        wave = base_wave * env * trem

        wave = wave.astype(np.float32)

        # pan and mix
        angle = (pan + 1.0) * math.pi / 4.0
        lg = math.cos(angle)
        rg = math.sin(angle)

        end_idx = global_note_start + note_len
        mix_l[global_note_start:end_idx] += wave * gain * lg
        mix_r[global_note_start:end_idx] += wave * gain * rg


def build_melody_line_spec(section_index, line_index, fib_n):
    seed = MASTER_SEED + 100000 + section_index * 500 + line_index
    rng = random.Random(seed)

    # different note speeds / durations
    note_period_steps = rng.choice([2, 2, 4, 4, 6, 8])
    note_dur_steps = rng.choice([4, 6, 8, 12])

    # choose timbre
    timbre = rng.choice(["soft_sine_tri", "glass_fm", "pad_fm"])

    # pan across stereo field
    pan = rng.uniform(-0.8, 0.8)
    # later Fibonacci sections can be a bit quieter per line
    base_gain = 0.18
    gain = base_gain * (0.9 + rng.random() * 0.3) / (1.0 + 0.02 * max(0, fib_n - 5))

    return {
        "note_period_steps": note_period_steps,
        "note_dur_steps": note_dur_steps,
        "timbre": timbre,
        "pan": pan,
        "gain": gain,
    }


def render_melodies(sections, total_samples):
    mel_l = np.zeros(total_samples, dtype=np.float32)
    mel_r = np.zeros(total_samples, dtype=np.float32)

    # we need section start offsets in bars/samples
    bars_so_far = 0
    section_offsets = []
    for bars, mode, fib_n in sections:
        start_bar = bars_so_far
        start_sample = int(round(start_bar * BAR_DURATION_SEC * SR))
        section_samples = int(round(bars * BAR_DURATION_SEC * SR))
        section_offsets.append((start_bar, start_sample, section_samples))
        bars_so_far += bars

    # For each FULL section: create fib_n melodic lines that span that
    # full section + its following lull (if present).
    for idx, ((bars, mode, fib_n), (start_bar, start_sample, section_samples)) in enumerate(zip(sections, section_offsets)):
        if mode != "full":
            continue

        # span full + next lull
        span_start = start_sample
        span_end = start_sample + section_samples

        if idx + 1 < len(sections):
            bars2, mode2, fib2 = sections[idx + 1]
            if mode2 == "lull":
                # extend span over lull too
                span_end = span_start + int(round((bars + bars2) * BAR_DURATION_SEC * SR))

        span_end = min(span_end, total_samples)

        print(f"Melodies for full section {idx}: Fn={fib_n}, span_samples={span_end - span_start}")

        for line_index in range(fib_n):
            spec = build_melody_line_spec(idx, line_index, fib_n)
            line_seed = MASTER_SEED + 200000 + idx * 1000 + line_index
            synth_melody_notes(mel_l, mel_r, span_start, span_end, line_seed, spec)

    return mel_l, mel_r


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


# ============================
# Main
# ============================

if __name__ == "__main__":
    print("Building section plan...")
    sections = build_section_plan()

    # Total bars to figure overall length
    total_bars = sum(b for b, _, _ in sections)
    total_samples = int(round(total_bars * BAR_DURATION_SEC * SR))
    print(f"Total bars: {total_bars}, total_samples: {total_samples}")

    print("Rendering drums...")
    drums_l, drums_r = render_drums(sections, total_samples)

    print("Rendering melodies (Aphex-y sad lines)...")
    mel_l, mel_r = render_melodies(sections, total_samples)

    print("Mixing and mastering...")
    full_l = drums_l + mel_l
    full_r = drums_r + mel_r
    stereo = np.stack([full_l, full_r], axis=1)

    # Normalise & gentle master drive
    max_val = np.max(np.abs(stereo)) + 1e-9
    stereo = stereo / max_val * 0.9
    stereo = np.tanh(stereo * 1.8)

    duration = stereo.shape[0] / SR
    print(f"Rendered duration: {duration:.2f} seconds (~{duration/60:.2f} minutes)")

    out_name = "fibonacci_megadrive_fib_layers_13min.wav"
    write_wav_stereo(out_name, stereo)
    print(f"Written to {out_name}")

