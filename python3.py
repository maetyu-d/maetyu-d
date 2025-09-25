# logistic_three_lines_drone_MORPH.py
# Three logistic-map lines with time-varying r(t) per line.
# Pure NumPy + stdlib; outputs "logistic_three_lines_drone_MORPH.wav"

import numpy as np
import wave, struct

# ---------------------------- Config ----------------------------
SR       = 44100          # sample rate
DUR_SEC  = 360            # length (e.g., 6 minutes). Increase for longer pieces.
CTRL_HZ  = 200            # control-rate for parameter curves
VOICES   = 7              # supersaw voices per line
X0       = 0.5            # logistic seed
PANS     = [-0.6, 0.1, 0.7]  # constant-power pan per line
TARGET_DB = -18.0
FADE_SEC  = 8.0
OUT_WAV   = "logistic_three_lines_drone_MORPH.wav"

# Each line gets its own r(t) waypoints: (time_frac, r_value)
R_WAYPOINTS = [
    # Line A: gentle → musical → complex
    [(0.00, 2.80), (0.30, 3.20), (0.70, 3.60), (1.00, 3.80)],
    # Line B: a bit hotter overall
    [(0.00, 3.05), (0.35, 3.50), (0.75, 3.85), (1.00, 3.95)],
    # Line C: starts mid, ends near chaotic edge
    [(0.00, 3.30), (0.40, 3.70), (0.85, 3.95), (1.00, 3.99)],
]

# ------------------------- Helpers ------------------------------
N  = SR * DUR_SEC
t  = np.arange(N) / SR
M  = int(DUR_SEC * CTRL_HZ)
tc = np.linspace(0.0, DUR_SEC, M)

def r_profile_from_waypoints(waypoints, tc, dur):
    """Piecewise-linear r(t) from fractional time waypoints."""
    x = np.array([w[0] for w in waypoints]) * dur
    y = np.array([w[1] for w in waypoints])
    return np.interp(tc, x, y)

def logistic_timevarying_ctrl(x0, r_arr):
    """Control-rate logistic with time-varying r."""
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
def saw_from_phase(p):  return 2.0 * (p - np.floor(p + 0.5))
def pulse_from_phase(p, d): return np.where(np.mod(p,1.0) < d, 1.0, -1.0)
def softsat(x, amt): return np.tanh(amt * x)

# Simple feedback comb (light chorus shimmer)
def comb_feedback(x, delay_samps, fb, mix):
    d = int(max(1, delay_samps))
    y = np.zeros_like(x); buf = np.zeros(d, dtype=np.float64); idx = 0
    for n in range(len(x)):
        out = buf[idx]
        val = x[n] + out * fb
        buf[idx] = val
        idx = (idx + 1) % d
        y[n] = (1.0 - mix) * x[n] + mix * out
    return y

# --------------------- Line / Layer builder ---------------------
def make_line(waypoints, pan_bias=0.0, seed=0, voices=7):
    rng = np.random.default_rng(100 + seed)

    # 1) r(t) per line, then logistic at control rate
    r_arr = r_profile_from_waypoints(waypoints, tc, DUR_SEC)
    x = logistic_timevarying_ctrl(X0, r_arr)

    # 2) Three timescales (retain character, avoid zipper)
    slow = moving_avg(x, int(1.2 * CTRL_HZ))    # ~1.2 s
    med  = moving_avg(x, int(0.30 * CTRL_HZ))   # ~0.3 s
    fast = moving_avg(x, int(0.06 * CTRL_HZ))   # ~60 ms

    # 3) Map control streams to parameters (control-rate)
    base_hz   = map01(slow,  32.0, 160.0)   # C1..E3 span for drama
    detune_ct = map01(med,    6.0, 36.0)
    vib_hz    = map01(fast,   0.04, 0.85)
    vib_ct    = map01(med,    0.2, 10.0)
    pwm_depth = map01(med,    0.00, 0.50)
    pwm_hz    = map01(fast,   0.03, 0.20)
    amp_env   = map01(slow,   0.72, 1.00)
    drive     = map01(med,    0.85, 1.35)
    haas_ms   = map01(slow,   0.0,  12.0)
    comb_ms   = map01(slow,   9.0,  26.0)
    comb_fb   = map01(med,    0.12, 0.40)
    comb_mix  = map01(med,    0.05, 0.30)

    # Logistic “event” gates for structure
    sub_gate   = moving_avg((x > 0.82).astype(float), int(0.15 * CTRL_HZ))
    noise_gate = moving_avg((x > 0.92).astype(float), int(0.10 * CTRL_HZ))

    # 4) Upsample control to audio rate
    def up(v): return np.interp(t, tc, v)
    base_hz, detune_ct, vib_hz, vib_ct = map(up, (base_hz, detune_ct, vib_hz, vib_ct))
    pwm_depth, pwm_hz, amp_env, drive, haas_ms = map(up, (pwm_depth, pwm_hz, amp_env, drive, haas_ms))
    comb_ms, comb_fb, comb_mix = map(up, (comb_ms, comb_fb, comb_mix))
    sub_gate, noise_gate = map(up, (sub_gate, noise_gate))

    # 5) Supersaw voices
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

    # 6) Hoover-ish PWM octave
    duty = 0.5 + 0.5 * pwm_depth * np.sin(2*np.pi*pwm_hz*t + 0.31 * seed)
    phase_pwm = (np.cumsum(2.0 * base_hz / SR)) % 1.0
    pwm = pulse_from_phase(phase_pwm, duty)
    line = 0.86 * line + 0.14 * pwm

    # 7) Sub-octave + airy noise swells
    phase_sub = (np.cumsum(0.5 * base_hz / SR)) % 1.0
    sub   = saw_from_phase(phase_sub) * sub_gate
    noise = (rng.normal(0, 1, N) * 0.06) * noise_gate
    line = line + 0.22 * sub + noise

    # 8) Nonlinear color + gentle chorus shimmer
    line = softsat(line, drive)
    delay_samples = int(np.median(comb_ms) * 0.001 * SR)
    line = comb_feedback(line, delay_samples, float(np.median(comb_fb)),
                         float(np.median(comb_mix)))

    # 9) Slow amplitude breathing
    line *= amp_env

    # 10) Haas stereo + constant-power pan
    d_samp = int(np.mean(haas_ms) * 0.001 * SR)
    L = line.copy(); R = line.copy()
    if d_samp > 0:
        R = np.concatenate([np.zeros(d_samp), R[:-d_samp]])

    pan = np.clip(pan_bias, -1.0, 1.0)
    gL = np.sqrt((1.0 - pan) * 0.5)
    gR = np.sqrt((1.0 + pan) * 0.5)
    return gL * L, gR * R

# --------------------------- Render -----------------------------
L_mix = np.zeros(N, dtype=np.float64)
R_mix = np.zeros(N, dtype=np.float64)

for i, (wps, pan) in enumerate(zip(R_WAYPOINTS, PANS)):
    Li, Ri = make_line(waypoints=wps, pan_bias=pan, seed=i, voices=VOICES)
    L_mix += Li;  R_mix += Ri

# Fades
fadeN = int(FADE_SEC * SR)
fade_in  = np.linspace(0, 1, fadeN)
fade_out = np.linspace(1, 0, fadeN)
L_mix[:fadeN] *= fade_in;   R_mix[:fadeN] *= fade_in
L_mix[-fadeN:] *= fade_out; R_mix[-fadeN:] *= fade_out

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
