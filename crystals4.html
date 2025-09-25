# logistic_all_params_piece_chaos.py
# All parameters driven by the logistic map x_{n+1} = r*x_n*(1-x_n).
# Three independent lines with time-varying growth rates (now ending at r=4.0).
# Expanded timbral ranges; lower bit-depth min; higher decimation max.
# Pure NumPy + stdlib. Outputs "logistic_all_params_piece_chaos.wav"

import numpy as np
import wave, struct

# ---------------------------- CONFIG ----------------------------
SR        = 44100         # sample rate
DUR_SEC   = 360           # piece length (seconds)
CTRL_HZ   = 200           # control-rate for parameter curves
VOICES    = 7             # supersaw voices per line
X0        = 0.5           # logistic seed (normalized)
PANS      = [-0.6, 0.1, 0.7]   # constant-power pan per line (-1..+1)
TARGET_DB = -18.0         # RMS target
FADE_SEC  = 8.0           # fade in/out (seconds)

OUT_WAV   = "logistic_all_params_piece_chaos.wav"

# r(t) WAYPOINTS per line: (time_fraction, r_value)
# CHANGED: last waypoint set to 4.0 (chaotic limit).
R_WAYPOINTS = [
    # Line A: gentle → musical → complex → chaotic
    [(0.00, 2.80), (0.30, 3.20), (0.70, 3.60), (1.00, 4.00)],
    # Line B: hotter overall, late chaos
    [(0.00, 3.05), (0.35, 3.50), (0.75, 3.90), (1.00, 4.00)],
    # Line C: mid → edge → full chaos
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
    for n in range(len(r_arr) - 1):
        x[n+1] = r_arr[n] * x[n] * (1.0 - x[n])
    return x

def moving_avg(x, win):
    if win <= 1: return x
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, k, mode="same")

def map01(x, lo, hi):
    return lo + (hi - lo) * np.clip(x, 0.0, 1.0)

def cents_to_ratio(c):  return 2.0 ** (c / 1200.0)
def saw_from_phase(p):  return 2.0 * (p - np.floor(p + 0.5))
def pulse_from_phase(p, d): return np.where(np.mod(p,1.0) < d, 1.0, -1.0)
def softsat(x, amt): return np.tanh(amt * x)

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
def make_line(r_waypoints, pan_bias=0.0, seed=0, voices=7):
    rng = np.random.default_rng(100 + seed)

    # r(t) and control-rate logistic x(t)
    r_arr = r_profile_from_waypoints(r_waypoints, tc, DUR_SEC)
    x = logistic_timevarying_ctrl(X0, r_arr)

    # Three time-scales (retain character, avoid zipper)
    slow = moving_avg(x, int(1.2 * CTRL_HZ))    # ~1.2 s
    med  = moving_avg(x, int(0.30 * CTRL_HZ))   # ~0.3 s
    fast = moving_avg(x, int(0.06 * CTRL_HZ))   # ~60 ms

    # ---------- Expanded PARAMETER RANGES (more dramatic) ----------
    base_hz   = map01(slow,  28.0, 220.0)   # wider pitch region (A0..A3)
    detune_ct = map01(med,    8.0, 48.0)    # wider supersaw spread (cents)
    vib_hz    = map01(fast,   0.03, 1.00)   # faster vibrato upper bound
    vib_ct    = map01(med,    0.2, 12.0)    # wider vibrato depth (cents)
    pwm_depth = map01(med,    0.00, 0.60)   # deeper PWM
    pwm_hz    = map01(fast,   0.02, 0.25)   # faster PWM rates
    amp_env   = map01(slow,   0.68, 1.00)   # deeper breathing
    drive     = map01(med,    0.85, 1.45)   # stronger tanh drive
    haas_ms   = map01(slow,   0.0,  14.0)   # wider stereo Haas
    # Texture/space
    comb_ms   = map01(slow,   8.0,  30.0)   # longer chorus delay
    comb_fb   = map01(med,    0.12, 0.50)   # more feedback
    comb_mix  = map01(med,    0.05, 0.40)   # wetter mix
    # Logistic “events”
    sub_gate   = moving_avg((x > 0.82).astype(float), int(0.15 * CTRL_HZ))
    noise_gate = moving_avg((x > 0.92).astype(float), int(0.10 * CTRL_HZ))
    # Lo-fi grit (CHANGED): bits min = 6.0 (was 7), srate_div max = 8 (was 5)
    bits      = map01(med,  12.0,  6.0)    # 12→6 bits
    srate_div = map01(fast,  1.0,   8.0)    # 1→8× decimation

    # Upsample control to audio rate
    def up(v): return np.interp(t, tc, v)
    base_hz, detune_ct, vib_hz, vib_ct = map(up, (base_hz, detune_ct, vib_hz, vib_ct))
    pwm_depth, pwm_hz, amp_env, drive, haas_ms = map(up, (pwm_depth, pwm_hz, amp_env, drive, haas_ms))
    comb_ms, comb_fb, comb_mix = map(up, (comb_ms, comb_fb, comb_mix))
    sub_gate, noise_gate = map(up, (sub_gate, noise_gate))
    bits, srate_div = map(up, (bits, srate_div))

    # Supersaw voices
    line = np.zeros(N, dtype=np.float64)
    vibrato = np.sin(2*np.pi*vib_hz*t) * vib_ct
    spread = np.linspace(-1.0, 1.0, voices)
    for d in spread:
        cents = d * detune_ct
        ratio = cents_to_ratio(cents + vibrato)
        freq  = base_hz * ratio
        phase = (rng.random() + np.cumsum(freq / SR)) % 1.0
        line += saw_from_phase(phase)
    line /= voices

    # Hoover-ish PWM octave
    duty = 0.5 + 0.5 * pwm_depth * np.sin(2*np.pi*pwm_hz*t + 0.31 * seed)
    phase_pwm = (np.cumsum(2.0 * base_hz / SR)) % 1.0
    pwm = pulse_from_phase(phase_pwm, duty)
    line = 0.86 * line + 0.14 * pwm

    # Sub-octave & airy noise swells as logistic events
    phase_sub = (np.cumsum(0.5 * base_hz / SR)) % 1.0
    sub   = saw_from_phase(phase_sub) * sub_gate
    noise = (rng.normal(0, 1, N) * 0.06) * noise_gate
    line = line + 0.22 * sub + noise

    # Nonlinear color + gentle comb shimmer
    line = softsat(line, drive)
    delay_samples = int(np.median(comb_ms) * 0.001 * SR)
    line = comb_feedback(line, delay_samples, float(np.median(comb_fb)),
                         float(np.median(comb_mix)))

    # Lo-fi grit (bit-crush + simple decimation)
    q_bits = int(np.clip(np.median(bits), 4, 16))
    levels = 2**q_bits
    line = np.round((line * 0.5 + 0.5) * (levels - 1)) / (levels - 1)
    line = line * 2.0 - 1.0
    div = int(np.clip(np.median(srate_div), 1, 8))
    if div > 1:
        line = np.repeat(line[::div], div)[:N]

    # Slow amplitude “breathing”
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
    Li, Ri = make_line(r_waypoints=wps, pan_bias=pan, seed=i, voices=VOICES)
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
stereo = np.tanh(stereo * 1.05)

with wave.open(OUT_WAV, "w") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    ints = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    wf.writeframes(struct.pack("<" + "h"*ints.size, *ints.flatten()))

print("Wrote:", OUT_WAV)
