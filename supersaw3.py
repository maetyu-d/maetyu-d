# supersaw_refactored.py
import numpy as np
import wave
from math import ceil

SAMPLE_RATE = 44_100
BLOCK_SIZE  = 2048

TOTAL_DURATION_S = 11 * 60
PHASE1_S         = 5 * 60
PHASE2_S         = TOTAL_DURATION_S - PHASE1_S

BASE_FREQ_HZ     = 110.0
NUM_LAYERS       = 6
SAWS_PER_SIDE    = 7
DETUNE_CENTS_MIN = 8.0
DETUNE_CENTS_MAX = 35.0
STEREO_SPREAD_CT = 3.0

MASTER_GAIN_DBFS = -1.0

def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

def cents_to_ratio(cents):
    return 2.0 ** (cents / 1200.0)

def make_symmetric_cents(num, spread_cents):
    if num == 1:
        return np.array([0.0], dtype=np.float64)
    idx = np.arange(num, dtype=np.float64) - (num - 1) / 2.0
    return (idx / ((num - 1) / 2.0)) * (spread_cents / 2.0)

def naive_saw_from_phase(phase):
    x = (phase / (2 * np.pi)) % 1.0
    return 2.0 * (x - np.floor(x + 0.5))

def block_time_envs(t0):
    x = (t0 / PHASE1_S) * NUM_LAYERS
    layer_pos = np.arange(NUM_LAYERS, dtype=np.float64)
    layer_gains = smoothstep01(np.clip(x - layer_pos, 0.0, 1.0))
    if t0 <= PHASE1_S:
        detune_now = DETUNE_CENTS_MIN
    else:
        u = (t0 - PHASE1_S) / max(PHASE2_S, 1e-9)
        detune_now = DETUNE_CENTS_MIN + (DETUNE_CENTS_MAX - DETUNE_CENTS_MIN) * smoothstep01(u)
    return layer_gains, detune_now

def write_wav_int16(path, stereo_float):
    x = np.clip(stereo_float, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(x.reshape(-1).tobytes())

def normalize_to_dbfs(buf, target_db=-1.0):
    peak = np.max(np.abs(buf))
    if peak < 1e-9:
        return buf
    target = 10 ** (target_db / 20.0)
    return buf * (target / peak)

def synth_supersaw_wav(
    out_path: str,
    total_seconds: int = TOTAL_DURATION_S,
    base_freq_hz: float = BASE_FREQ_HZ,
    num_layers: int = NUM_LAYERS,
    saws_per_side: int = SAWS_PER_SIDE,
):
    total_samples = int(total_seconds * SAMPLE_RATE)
    n_blocks = ceil(total_samples / BLOCK_SIZE)

    rng = np.random.default_rng(12345)
    phases = rng.uniform(0, 2 * np.pi, size=(num_layers, 2, saws_per_side)).astype(np.float64)

    layer_pan = np.linspace(0.2, 0.8, num_layers)
    layer_gain_L = np.cos(layer_pan * np.pi / 2)
    layer_gain_R = np.sin(layer_pan * np.pi / 2)

    base_offsets = make_symmetric_cents(saws_per_side, 1.0)

    outL = np.zeros(total_samples, dtype=np.float64)
    outR = np.zeros(total_samples, dtype=np.float64)

    write_idx = 0
    for _ in range(n_blocks):
        this_N = min(BLOCK_SIZE, total_samples - write_idx)
        if this_N <= 0:
            break

        t0 = write_idx / SAMPLE_RATE
        layer_gains, detune_cents_now = block_time_envs(t0)

        stereo_off_L = -STEREO_SPREAD_CT / 2.0
        stereo_off_R = +STEREO_SPREAD_CT / 2.0

        cents_core = base_offsets * detune_cents_now
        cents_L = cents_core + stereo_off_L
        cents_R = cents_core + stereo_off_R

        cents = np.stack([cents_L, cents_R], axis=0)[None, :, :]
        cents = np.repeat(cents, num_layers, axis=0)

        freqs = base_freq_hz * cents_to_ratio(cents)
        incs  = (2.0 * np.pi * freqs) / SAMPLE_RATE

        k = np.arange(this_N, dtype=np.float64)[:, None, None, None]
        block_phase = phases[None, ...] + k * incs[None, ...]
        block_saw   = naive_saw_from_phase(block_phase)

        block_mix   = block_saw.mean(axis=3)

        mixL = (block_mix[:, :, 0] * (layer_gains * layer_gain_L)[None, :]).sum(axis=1)
        mixR = (block_mix[:, :, 1] * (layer_gains * layer_gain_R)[None, :]).sum(axis=1)

        scale = 1.0 / max(num_layers, 1)
        outL[write_idx:write_idx + this_N] += mixL * scale
        outR[write_idx:write_idx + this_N] += mixR * scale

        phases = (phases + incs * this_N) % (2.0 * np.pi)
        write_idx += this_N

    stereo = np.stack([outL, outR], axis=1)
    stereo = normalize_to_dbfs(stereo, MASTER_GAIN_DBFS)
    write_wav_int16(out_path, stereo.astype(np.float32))
    print(f"Wrote: {out_path}")
