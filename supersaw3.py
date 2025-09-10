# supersaw_refactored.py
import numpy as np
import wave
from math import ceil
import os, sys, argparse

# ---------- Global settings (defaults) ----------
SAMPLE_RATE = 44_100
BLOCK_SIZE  = 2048

TOTAL_DURATION_S = 11 * 60   # 11 minutes
PHASE1_S         = 5 * 60    # layers fade-in
PHASE2_S         = TOTAL_DURATION_S - PHASE1_S

BASE_FREQ_HZ     = 110.0     # A2-ish
NUM_LAYERS       = 6
SAWS_PER_SIDE    = 7
DETUNE_CENTS_MIN = 8.0
DETUNE_CENTS_MAX = 35.0
STEREO_SPREAD_CT = 3.0       # cents L/R offset

MASTER_GAIN_DBFS = -1.0      # normalize peak to -1 dBFS

# ---------- Helpers ----------
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
    # Phase in radians -> naive saw [-1, 1]
    x = (phase / (2 * np.pi)) % 1.0
    return 2.0 * (x - np.floor(x + 0.5))

def block_time_envs(t0, faststart=False):
    """Return (layer_gains, detune_now) at block start time t0."""
    if faststart:
        layer_gains = np.ones(NUM_LAYERS, dtype=np.float64)
    else:
        if PHASE1_S <= 0:
            layer_gains = np.ones(NUM_LAYERS, dtype=np.float64)
        else:
            x = (t0 / PHASE1_S) * NUM_LAYERS
            layer_pos = np.arange(NUM_LAYERS, dtype=np.float64)
            layer_gains = smoothstep01(np.clip(x - layer_pos, 0.0, 1.0))

    if PHASE2_S <= 0:
        detune_now = DETUNE_CENTS_MAX
    else:
        if t0 <= PHASE1_S:
            detune_now = DETUNE_CENTS_MIN
        else:
            u = (t0 - PHASE1_S) / max(PHASE2_S, 1e-9)
            detune_now = DETUNE_CENTS_MIN + (DETUNE_CENTS_MAX - DETUNE_CENTS_MIN) * smoothstep01(u)

    return layer_gains, detune_now

def write_wav_int16(path, stereo_float, samplerate=SAMPLE_RATE):
    """Write float stereo [-1,1] to 16-bit PCM WAV."""
    x = np.clip(stereo_float, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    abspath = os.path.abspath(path)
    out_dir = os.path.dirname(abspath)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with wave.open(abspath, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(x.reshape(-1).tobytes())
    return abspath

def normalize_to_dbfs(buf, target_db=-1.0):
    peak = np.max(np.abs(buf))
    if peak < 1e-12:
        return buf
    target = 10 ** (target_db / 20.0)
    return buf * (target / peak)

# ---------- Synth core ----------
def synth_supersaw_wav(
    out_path: str,
    total_seconds: int = TOTAL_DURATION_S,
    base_freq_hz: float = BASE_FREQ_HZ,
    num_layers: int = NUM_LAYERS,
    saws_per_side: int = SAWS_PER_SIDE,
    sample_rate: int = SAMPLE_RATE,
    block_size: int = BLOCK_SIZE,
    faststart: bool = False,
    seed: int = 12345,
):
    """Render a wide, evolving supersaw to a stereo WAV."""
    total_seconds = max(int(total_seconds), 0)
    total_samples = int(total_seconds * sample_rate)
    if total_samples == 0:
        raise ValueError("total_seconds is 0. Increase --duration above 0.")

    n_blocks = ceil(total_samples / block_size)

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=(num_layers, 2, saws_per_side)).astype(np.float64)

    layer_pan    = np.linspace(0.2, 0.8, num_layers)
    layer_gain_L = np.cos(layer_pan * np.pi / 2)
    layer_gain_R = np.sin(layer_pan * np.pi / 2)

    base_offsets = make_symmetric_cents(saws_per_side, 1.0)

    outL = np.zeros(total_samples, dtype=np.float64)
    outR = np.zeros(total_samples, dtype=np.float64)

    write_idx = 0
    for _ in range(n_blocks):
        this_N = min(block_size, total_samples - write_idx)
        if this_N <= 0:
            break

        t0 = write_idx / sample_rate
        layer_gains, detune_cents_now = block_time_envs(t0, faststart=faststart)

        stereo_off_L = -STEREO_SPREAD_CT / 2.0
        stereo_off_R = +STEREO_SPREAD_CT / 2.0

        cents_core = base_offsets * detune_cents_now
        cents_L = cents_core + stereo_off_L
        cents_R = cents_core + stereo_off_R

        cents = np.stack([cents_L, cents_R], axis=0)[None, :, :]
        cents = np.repeat(cents, num_layers, axis=0)

        freqs = base_freq_hz * cents_to_ratio(cents)
        incs  = (2.0 * np.pi * freqs) / sample_rate

        k = np.arange(this_N, dtype=np.float64)[:, None, None, None]
        block_phase = phases[None, ...] + k * incs[None, ...]
        block_saw   = naive_saw_from_phase(block_phase)

        block_mix = block_saw.mean(axis=3)

        mixL = (block_mix[:, :, 0] * (layer_gains * layer_gain_L)[None, :]).sum(axis=1)
        mixR = (block_mix[:, :, 1] * (layer_gains * layer_gain_R)[None, :]).sum(axis=1)

        scale = 1.0 / max(num_layers, 1)
        outL[write_idx:write_idx + this_N] += mixL * scale
        outR[write_idx:write_idx + this_N] += mixR * scale

        phases = (phases + incs * this_N) % (2.0 * np.pi)
        write_idx += this_N

    stereo = np.stack([outL, outR], axis=1)
    stereo = normalize_to_dbfs(stereo, MASTER_GAIN_DBFS)
    abspath = write_wav_int16(out_path, stereo.astype(np.float32), samplerate=sample_rate)

    dur = total_samples / sample_rate
    size_mb = os.path.getsize(abspath) / 1_048_576
    print(f"Wrote: {abspath}")
    print(f"Duration: {dur:.2f} s | Size: {size_mb:.2f} MiB | SR: {sample_rate} Hz")
    return abspath

# ---------- CLI (sandbox & Jupyter safe) ----------
def _parse_args():
    parser = argparse.ArgumentParser(description="Render an evolving supersaw WAV.")
    parser.add_argument("output", nargs="?", default="supersaw.wav",
                        help="Output WAV filename (default: supersaw.wav)")
    parser.add_argument("-d", "--duration", type=float, default=30.0,
                        help="Duration in seconds (default: 30)")
    parser.add_argument("--freq", type=float, default=BASE_FREQ_HZ,
                        help=f"Base frequency in Hz (default: {BASE_FREQ_HZ})")
    parser.add_argument("-l", "--layers", type=int, default=NUM_LAYERS,
                        help=f"Number of layers (default: {NUM_LAYERS})")
    parser.add_argument("-s", "--saws", type=int, default=SAWS_PER_SIDE,
                        help=f"Saws per side (default: {SAWS_PER_SIDE})")
    parser.add_argument("--faststart", action="store_true",
                        help="Skip Phase-1 fade; start audible immediately.")
    parser.add_argument("--sr", "--samplerate", dest="sr", type=int, default=SAMPLE_RATE,
                        help=f"Sample rate in Hz (default: {SAMPLE_RATE})")
    parser.add_argument("--bs", "--blocksize", dest="bs", type=int, default=BLOCK_SIZE,
                        help=f"Block size (default: {BLOCK_SIZE})")
    return parser

if __name__ == "__main__":
    parser = _parse_args()
    # Ignore unknown args (sandbox/Jupyter injects extras)
    args, _ = parser.parse_known_args()

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    synth_supersaw_wav(
        out_path=out_path,
        total_seconds=max(1, int(args.duration)),
