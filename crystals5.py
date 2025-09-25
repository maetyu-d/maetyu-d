# logistic_all_params_piece_glassy.py
# Three logistic-map lines with time-varying r(t) ending at 4.0.
# All parameters are driven by the logistic map. Timbre = glassy sinewave banks.
# Pure NumPy + stdlib. Outputs "logistic_all_params_piece_glassy.wav"

import numpy as np
import wave, struct

# ---------------------------- CONFIG ----------------------------
SR        = 44100
DUR_SEC   = 360            # long form
CTRL_HZ   = 200            # control-rate for parameter curves
X0        = 0.5
VOICES    = 5              # voices per line (each is a sine bank)
BANK_PARTIALS = 7          # sines per voice (additive)
PANS      = [-0.6, 0.1, 0.7]
TARGET_DB = -18.0
FADE_SEC  = 8.0
OUT_WAV   = "logistic_all_params_piece_glassy.wav"

# r(t) WAYPOINTS per line (final waypoint at 4.0 as requested)
R_WAYPOINTS = [
    [(0.00, 2.80), (0.30, 3.20), (0.70, 3.60), (1.00, 4.00)],
    [(0.00, 3.05), (0.35, 3.50), (0.75, 3.90), (1.00, 4.00)],
    [(0.00, 3.30), (0.40, 3.75), (0.85, 3.97), (1.00, 4.00)],
]

# ------------------------- HELPERS ------------------------------
N  = SR * DUR_SEC
t  = np.arange(N) / SR
M  = int(DUR_SEC * CTRL_HZ)
tc = np.linspace(0.0, DUR_SEC, M)

def r_profile_from_waypoints(waypoints, tc, dur):
    xs = np.array([w[0] for w in waypoints]) * dur
    ys = np.array([w[1] for w in waypoints])
    return np.interp(tc, xs, ys)

def logistic_timevarying_ctrl(x0, r_arr):
    x = np.empty_like(r_arr, dtype=np.float64)
    x[0] = x0
    for n in range(len(r_arr)-1):
        x[n+1] = r_arr[n] * x[n] * (1.0 - x[n])
    return x

def moving_avg(x, win):
    if win <= 1: return x
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, k, mode="same")

def map01(x, lo, hi):
    return lo + (hi - lo) * np.clip(x, 0.0, 1.0)

def cents_to_ratio(c):  return 2.0 ** (c / 1200.0)
def softsat(x, amt):    return np.tanh(amt * x)

# Lightweight feedback comb for shimmer (mono before stereo split)
def comb_feedback(x, delay_samps, fb, mix):
    d = int(max(1, delay_samps))
    y = np.zeros_like(x)
    buf = np.zeros(d, dtype=np.float64)
    idx = 0
    for n in range(len(x)):
        out = buf[idx]
        val = x[n] + out * fb
        buf[idx] = val
        idx = (idx + 1) % d
        y[n] = (1.0 - mix) * x[n] + mix * out
    return y

# --------------------- BUILD ONE LOGISTIC LINE -------------------
def make_line(r_waypoints, pan_bias=0.0, seed=0):
    rng = np.random.default_rng(100 + seed)

    # 1) r(t) and control-rate logistic x(t)
    r_arr = r_profile_from_waypoints(r_waypoints, tc, DUR_SEC)
    x = logistic_timevarying_ctrl(X0, r_arr)

    # 2) Three time-scales (retain character, avoid zipper)
    slow = moving_avg(x, int(1.2 * CTRL_HZ))    # ~1.2 s
    med  = moving_avg(x, int(0.30 * CTRL_HZ))   # ~0.3 s
    fast = moving_avg(x, int(0.06 * CTRL_HZ))   # ~60 ms

    # 3) Map control streams to parameters (control-rate)
    # Expanded ranges (as requested)
    base_hz     = map01(slow,  28.0, 220.0)     # A0..A3 region
    detune_ct   = map01(med,    8.0, 48.0)      # inter-voice cents
    vib_hz      = map01(fast,   0.03, 1.00)     # vibrato rate
    vib_ct      = map01(med,    0.1,  10.0)     # vibrato depth (cents)
    # Sine-bank “glass” controls
    inharm_amt  = map01(med,    0.0,  0.15)     # partial ratio stretch
    bank_spread = map01(slow,   0.05, 0.35)     # partial detune across bank
    bank_decay  = map01(med,    0.35, 0.85)     # partial amplitude falloff
    fm_mix      = map01(med,    0.00, 0.30)     # light FM shimmer mix
    fm_index    = map01(fast,   0.00, 2.00)     # FM index (small → glassy)
    fm_ratio    = map01(med,    1.25, 3.00)     # modulator/carrier ratio
    # Space/texture
    haas_ms     = map01(slow,   0.0,  14.0)     # stereo width
    comb_ms     = map01(slow,   8.0,  30.0)
    comb_fb     = map01(med,    0.12, 0.50)
    comb_mix    = map01(med,    0.05, 0.40)
    amp_env     = map01(slow,   0.70, 1.00)
    drive       = map01(med,    0.80, 1.30)     # gentle tanh (keep glassy)
    # Structural “events”
    sub_gate    = moving_avg((x > 0.82).astype(float), int(0.15 * CTRL_HZ))
    air_gate    = moving_avg((x > 0.92).astype(float), int(0.10 * CTRL_HZ))
    # Lo-fi grit
    bits        = map01(med,   12.0, 6.0)       # 12→6 bits (min lowered)
    srate_div   = map01(fast,   1.0, 8.0)       # 1→8× decimation (max raised)

    # 4) Upsample control streams to audio rate
    def up(v): return np.interp(t, tc, v)
    base_hz, detune_ct, vib_hz, vib_ct   = map(up, (base_hz, detune_ct, vib_hz, vib_ct))
    inharm_amt, bank_spread, bank_decay  = map(up, (inharm_amt, bank_spread, bank_decay))
    fm_mix, fm_index, fm_ratio           = map(up, (fm_mix, fm_index, fm_ratio))
    haas_ms, comb_ms, comb_fb, comb_mix  = map(up, (haas_ms, comb_ms, comb_fb, comb_mix))
    amp_env, drive                       = map(up, (amp_env, drive))
    sub_gate, air_gate                   = map(up, (sub_gate, air_gate))
    bits, srate_div                      = map(up, (bits, srate_div))

    # 5) Sinewave bank synthesis per voice
    line = np.zeros(N, dtype=np.float64)

    # Voice-level vibrato (cents → ratio)
    vibrato = np.sin(2*np.pi*vib_hz*t) * vib_ct

    # Precompute base carrier freq per sample
    for v in range(VOICES):
        # Inter-voice detune in cents (symmetric)
        d = np.linspace(-1.0, 1.0, VOICES)[v]
        voice_ratio = cents_to_ratio(d * detune_ct + vibrato)
        fc = base_hz * voice_ratio  # carrier freq

        # Light FM for “glassiness”
        fm = fm_index * np.sin(2*np.pi * (fc * fm_ratio) * t)
        # Sine-bank phases with slight inharmonic partial ratios
        # Partial ratios p_k ≈ k * (1 + inharm_amt * offset)
        partials = np.arange(1, BANK_PARTIALS + 1)
        # per-partial tiny offsets spread across the bank
        offsets = np.linspace(-1.0, 1.0, BANK_PARTIALS) * bank_spread
        # amplitude envelope across partials
        amps = (1.0 / partials**bank_decay)
        amps /= np.sum(amps)  # normalise bank energy

        # Build voice as sum of sine partials (with mild phase randomisation)
        voice = np.zeros(N, dtype=np.float64)
        for k, (pk, off, ak) in enumerate(zip(partials, offsets, amps)):
            # inharmonic stretched ratio for this partial
            pratio = pk * (1.0 + inharm_amt * off)
            # instantaneous phase increment; add some FM to the carrier only (common shimmer),
            # and tiny PM to partials via 'off' so they glint differently
            phase_inc = (fc * pratio) / SR
            # accumulate phase with a small PM tied to logistic air_gate (subtle sparkle)
            pm = 0.015 * off * air_gate
            phase = np.cumsum(phase_inc) + pm * np.sin(2*np.pi*phase_inc * np.arange(N))
            # carrier FM mixed in post (keeps partials coherent)
            s = np.sin(2*np.pi*phase + fm_mix * fm)
            voice += ak * s

        line += voice

    line /= VOICES  # average voices

    # 6) Sub-octave reinforcement when system heats up (sine)
    sub_phase = np.cumsum((0.5 * base_hz) / SR)
    sub = np.sin(2*np.pi*sub_phase) * sub_gate
    line += 0.20 * sub

    # 7) Add airy noise “frost” gated by extremes
    air = (rng.normal(0, 1, N) * 0.04) * air_gate
    line += air

    # 8) Gentle nonlinear color (keep glass intact)
    line = softsat(line, drive)

    # 9) Comb shimmer (mono pre-stereo)
    delay_samples = int(np.median(comb_ms) * 0.001 * SR)
    line = comb_feedback(line, delay_samples, float(np.median(comb_fb)),
                         float(np.median(comb_mix)))

    # 10) Bit-crush + simple decimation (musical grit)
    q_bits = int(np.clip(np.median(bits), 4, 16))
    levels = 2**q_bits
    line = np.round((line * 0.5 + 0.5) * (levels - 1)) / (levels - 1)
    line = line * 2.0 - 1.0
    div = int(np.clip(np.median(srate_div), 1, 8))
    if div > 1:
        line = np.repeat(line[::div], div)[:N]

    # 11) Slow amplitude breathing
    line *= amp_env

    # 12) Haas stereo + constant-power pan
    d_samp = int(np.mean(haas_ms) * 0.001 * SR)
    L = line.copy(); R = line.copy()
    if d_samp > 0:
        R = np.concatenate([np.zeros(d_samp), R[:-d_samp]])
    pan = np.clip(pan_bias, -1.0, 1.0)
    gL = np.sqrt((1.0 - pan) * 0.5)
    gR = np.sqrt((1.0 + pan) * 0.5)
    return gL * L, gR * R

# --------------------------- RENDER ------------------------------
L_mix = np.zeros(N, dtype=np.float64)
R_mix = np.zeros(N, dtype=np.float64)

for i, (wps, pan) in enumerate(zip(R_WAYPOINTS, PANS)):
    Li, Ri = make_line(r_waypoints=wps, pan_bias=pan, seed=i)
    L_mix += Li
    R_mix += Ri

# Fades
fadeN = int(FADE_SEC * SR)
fade_in  = np.linspace(0, 1, fadeN)
fade_out = np.linspace(1, 0, fadeN)
L_mix[:fadeN]  *= fade_in;   R_mix[:fadeN]  *= fade_in
L_mix[-fadeN:] *= fade_out;  R_mix[-fadeN:] *= fade_out

# Normalize to -18 dBFS RMS + gentle safety
stereo = np.stack([L_mix, R_mix], axis=1)
rms = np.sqrt(np.mean(stereo**2))
gain = 10 ** ((TARGET_DB - 20*np.log10(max(rms, 1e-12))) / 20.0)
stereo *= gain
stereo = np.tanh(stereo * 1.04)

with wave.open(OUT_WAV, "w") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    ints = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    wf.writeframes(struct.pack("<" + "h"*ints.size, *ints.flatten()))

print("Wrote:", OUT_WAV)
