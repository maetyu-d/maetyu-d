# logistic_glassy_11min_sparkly_clear.py
# Three logistic-map lines with r(t) ending at 4.0.
# Timbre: glassy sinewave banks (additive + light FM shimmer).
# All parameters driven by the logistic map.
# Stereo WAV output, ~11 minutes.

import numpy as np
import wave, struct

# ---------------------------- CONFIG ----------------------------
SR        = 44100
DUR_SEC   = 660            # 11 minutes
CTRL_HZ   = 200
X0        = 0.5
VOICES    = 5              # sine-bank voices per line
BANK_PARTIALS = 11         # more partials for sparkle
PANS      = [-0.6, 0.1, 0.7]
TARGET_DB = -18.0
FADE_SEC  = 8.0
OUT_WAV   = "logistic_glassy_11min_sparkly_clear.wav"

# r(t) WAYPOINTS per line (end at 4.0 = chaotic edge)
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

# Lightweight feedback comb (mono before stereo split)
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

    # r(t) and control-rate logistic x(t)
    r_arr = r_profile_from_waypoints(r_waypoints, tc, DUR_SEC)
    x = logistic_timevarying_ctrl(X0, r_arr)

    # Three time-scales
    slow = moving_avg(x, int(1.2 * CTRL_HZ))
    med  = moving_avg(x, int(0.30 * CTRL_HZ))
    fast = moving_avg(x, int(0.06 * CTRL_HZ))

    # --------- Sparkly & wide but clear parameter maps ----------
    base_hz     = map01(slow,  28.0, 220.0)     # wide pitch region
    detune_ct   = map01(med,    8.0, 40.0)      # inter-voice cents
    vib_hz      = map01(fast,   0.03, 1.00)     # vibrato rate
    vib_ct      = map01(med,    0.1,  10.0)     # vibrato depth (cents)

    # Glassy sine-bank controls (expanded for sparkle)
    inharm_amt  = map01(med,    0.05, 0.25)     # more inharmonic spread
    bank_spread = map01(slow,   0.10, 0.45)     # wider partial spacing
    bank_decay  = map01(med,    0.30, 0.75)     # slower decay = brighter banks

    # FM shimmer (stronger but still musical)
    fm_mix      = map01(med,    0.10, 0.45)
    fm_index    = map01(fast,   0.40, 3.00)
    fm_ratio    = map01(med,    1.50, 3.50)

    # Space / clarity
    haas_ms     = map01(slow,   4.0,  22.0)     # wider stereo spread
    comb_ms     = map01(slow,   8.0,  26.0)
    comb_fb     = map01(med,    0.10, 0.35)     # lighter feedback for clarity
    comb_mix    = map01(med,    0.04, 0.25)     # subtler chorus

    amp_env     = map01(slow,   0.72, 1.00)
    drive       = map01(med,    0.75, 1.15)     # gentler drive → clearer

    # Structural “events”
    sub_gate    = moving_avg((x > 0.82).astype(float), int(0.15 * CTRL_HZ))
    air_gate    = moving_avg((x > 0.92).astype(float), int(0.10 * CTRL_HZ))

    # Lo-fi grit (clearer): higher bit depth, less decimation
    bits        = map01(med,   12.0, 10.0)      # 12→10 bits (was 12→6)
    srate_div   = map01(fast,   1.0,  4.0)      # 1→4× decimation (was up to 8)

    # Upsample control streams to audio rate
    def up(v): return np.interp(t, tc, v)
    base_hz, detune_ct, vib_hz, vib_ct   = map(up, (base_hz, detune_ct, vib_hz, vib_ct))
    inharm_amt, bank_spread, bank_decay  = map(up, (inharm_amt, bank_spread, bank_decay))
    fm_mix, fm_index, fm_ratio           = map(up, (fm_mix, fm_index, fm_ratio))
    haas_ms, comb_ms, comb_fb, comb_mix  = map(up, (haas_ms, comb_ms, comb_fb, comb_mix))
    amp_env, drive                       = map(up, (amp_env, drive))
    sub_gate, air_gate                   = map(up, (sub_gate, air_gate))
    bits, srate_div                      = map(up, (bits, srate_div))

    # Sinewave bank synthesis per voice
    line = np.zeros(N, dtype=np.float64)
    vibrato = np.sin(2*np.pi*vib_hz*t) * vib_ct

    for v in range(VOICES):
        d = np.linspace(-1.0, 1.0, VOICES)[v]
        voice_ratio = cents_to_ratio(d * detune_ct + vibrato)
        fc = base_hz * voice_ratio  # carrier

        # FM shimmer
        fm = fm_index * np.sin(2*np.pi * (fc * fm_ratio) * t)

        # Partial setup
        partials = np.arange(1, BANK_PARTIALS + 1)
        offsets  = np.linspace(-1.0, 1.0, BANK_PARTIALS) * bank_spread
        amps     = (1.0 / partials**bank_decay)
        amps    /= np.sum(amps)

        voice = np.zeros(N, dtype=np.float64)
        for pk, off, ak in zip(partials, offsets, amps):
            pratio = pk * (1.0 + inharm_amt * off)
            phase_inc = (fc * pratio) / SR
            # Subtle PM tied to “air” to make highs twinkle
            pm = 0.02 * off * air_gate
            phase = np.cumsum(phase_inc) + pm * np.sin(2*np.pi*phase_inc * np.arange(N))
            s = np.sin(2*np.pi*phase + fm_mix * fm)
            voice += ak * s

        line += voice

    line /= VOICES

    # Sub-octave reinforcement (sine), gated
    sub_phase = np.cumsum((0.5 * base_hz) / SR)
    sub = np.sin(2*np.pi*sub_phase) * sub_gate
    line += 0.16 * sub  # slightly lower for clarity

    # Airy noise frost (tightened for clarity)
    air = (rng.normal(0, 1, N) * 0.03) * air_gate
    line += air

    # Gentle nonlinear color
    line = softsat(line, drive)

    # Subtle comb shimmer
    delay_samples = int(np.median(comb_ms) * 0.001 * SR)
    line = comb_feedback(line, delay_samples, float(np.median(comb_fb)),
                         float(np.median(comb_mix)))

    # Lo-fi grit (mild)
    q_bits = int(np.clip(np.median(bits), 4, 16))
    levels = 2**q_bits
    line = np.round((line * 0.5 + 0.5) * (levels - 1)) / (levels - 1)
    line = line * 2.0 - 1.0
    div = int(np.clip(np.median(srate_div), 1, 8))
    if div > 1:
        line = np.repeat(line[::div], div)[:N]

    # Slow amplitude breathing
    line *= amp_env

    # Haas stereo + constant-power pan
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

# Fades and leveling
fadeN = int(FADE_SEC * SR)
fade_in  = np.linspace(0, 1, fadeN)
fade_out = np.linspace(1, 0, fadeN)
L_mix[:fadeN]  *= fade_in;   R_mix[:fadeN]  *= fade_in
L_mix[-fadeN:] *= fade_out;  R_mix[-fadeN:] *= fade_out

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
