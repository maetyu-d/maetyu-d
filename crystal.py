# logistic_supersaw_drone.py
# Stereo Hoover-tinged supersaw drone with logistic-map control
# Pure NumPy + stdlib; no external deps.

import numpy as np
import wave, struct

SR = 44100
DUR_SEC = 610  # ~10 minutes
N = SR * DUR_SEC
t = np.arange(N) / SR

# Three logistic-map growth rates:
R_VALUES = [2.9, 3.5, 3.9]    # stable → periodic → chaotic-ish
X0 = 0.5                      # “Middle C” seed as normalized mid state
CTRL_HZ = 200                 # control-rate for parameter curves (fast enough, still cheap)

# ---------- helpers ----------
def logistic_series_ctrl(r, x0, M):
    x = np.empty(M, dtype=np.float64)
    x[0] = x0
    for n in range(M - 1):
        x[n + 1] = r * x[n] * (1.0 - x[n])
    return x

def moving_avg(x, win):
    if win <= 1: return x
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, k, mode="same")

def m01(x, lo, hi):
    return lo + (hi - lo) * np.clip(x, 0.0, 1.0)

def cents_to_ratio(c):
    return 2.0 ** (c / 1200.0)

def saw_from_phase(phase):
    # phase in [0,1)
    return 2.0 * (phase - np.floor(phase + 0.5))

def pulse_from_phase(phase, duty):
    return np.where(np.mod(phase, 1.0) < duty, 1.0, -1.0)

# ---------- layer builder ----------
def make_layer(r, pan_bias=0.0, seed=0, voices=7):
    # control-rate timebase
    M = int(DUR_SEC * CTRL_HZ)
    tc = np.linspace(0.0, DUR_SEC, M)

    # raw logistic
    x = logistic_series_ctrl(r, X0, M)

    # multi-timescale smoothing (keeps character while avoiding zipper)
    slow = moving_avg(x, int(CTRL_HZ * 1.5))     # ~1.5 s
    med  = moving_avg(x, int(CTRL_HZ * 0.30))    # ~0.3 s
    fast = moving_avg(x, int(CTRL_HZ * 0.06))    # ~60 ms

    # map control streams → parameters (control-rate)
    base_hz   = m01(slow, 39.0, 96.0)               # E1..G2-ish
    detune_ct = m01(med,   6.0, 28.0)               # cents spread
    vib_hz    = m01(fast,  0.04, 0.55)
    vib_cent  = m01(med,   0.2, 6.0)
    pwm_depth = m01(med,   0.00, 0.45)
    pwm_hz    = m01(fast,  0.02, 0.18)
    amp_env   = m01(slow,  0.78, 1.00)              # slow breathing
    drive     = m01(med,   0.8, 1.25)
    haas_ms   = m01(slow,  0.0, 12.0)               # stereo width

    # upsample control-rate parameters to audio rate by linear interpolation
    base_hz   = np.interp(t, tc, base_hz)
    detune_ct = np.interp(t, tc, detune_ct)
    vib_hz    = np.interp(t, tc, vib_hz)
    vib_cent  = np.interp(t, tc, vib_cent)
    pwm_depth = np.interp(t, tc, pwm_depth)
    pwm_hz    = np.interp(t, tc, pwm_hz)
    amp_env   = np.interp(t, tc, amp_env)
    drive     = np.interp(t, tc, drive)
    haas_ms   = np.interp(t, tc, haas_ms)

    # supersaw oscillators
    rng = np.random.default_rng(100 + seed)
    phase0 = rng.random(voices)
    spread = np.linspace(-1.0, 1.0, voices)

    layer = np.zeros(N, dtype=np.float64)
    vibrato = np.sin(2 * np.pi * vib_hz * t) * vib_cent
    for d in spread:
        cents = d * detune_ct
        ratio = cents_to_ratio(cents + vibrato)
        freq  = base_hz * ratio
        phase = (rng.random() + np.cumsum(freq / SR)) % 1.0
        layer += saw_from_phase(phase)
    layer /= voices

    # faint PWM octave for Hoover bite
    duty  = 0.5 + 0.5 * pwm_depth * np.sin(2 * np.pi * pwm_hz * t + 0.3 * seed)
    phase_pwm = (np.cumsum(2.0 * base_hz / SR)) % 1.0
    pwm = pulse_from_phase(phase_pwm, duty)
    layer = 0.88 * layer + 0.12 * pwm

    # gentle nonlinear drive + slow amplitude envelope
    layer = np.tanh(drive * layer) * amp_env

    # Haas stereo widening (simple sample offset on R)
    d_samp = int(np.mean(haas_ms) * 0.001 * SR)
    L = layer.copy()
    R = layer.copy()
    if d_samp > 0:
        R = np.concatenate([np.zeros(d_samp), R[:-d_samp]])

    # constant-power pan per layer
    pan = np.clip(pan_bias, -1.0, 1.0)
    gL = np.sqrt((1.0 - pan) * 0.5)
    gR = np.sqrt((1.0 + pan) * 0.5)
    return gL * L, gR * R

# ---------- render ----------
pans = [-0.6, 0.2, 0.7]  # spread the three r-layers
L = np.zeros(N, dtype=np.float64)
R = np.zeros(N, dtype=np.float64)

for i, (r, pan) in enumerate(zip(R_VALUES, pans)):
    Li, Ri = make_layer(r, pan_bias=pan, seed=i, voices=7)
    L += Li
    R += Ri

# fades
fade_sec = 8.0
fadeN = int(fade_sec * SR)
fade_in  = np.linspace(0, 1, fadeN)
fade_out = np.linspace(1, 0, fadeN)
L[:fadeN] *= fade_in;  R[:fadeN] *= fade_in
L[-fadeN:] *= fade_out; R[-fadeN:] *= fade_out

# normalize to -18 dBFS RMS with gentle safety
stereo = np.stack([L, R], axis=1)
rms = np.sqrt(np.mean(stereo**2))
target_db = -18.0
gain = 10 ** ((target_db - 20*np.log10(max(rms, 1e-12))) / 20.0)
stereo *= gain
stereo = np.tanh(stereo * 1.05)

# write WAV
out_path = "logistic_supersaw_drone_3r.wav"
with wave.open(out_path, "w") as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)   # 16-bit
    wf.setframerate(SR)
    ints = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    wf.writeframes(struct.pack("<" + "h" * ints.size, *ints.flatten()))

print("Wrote:", out_path)
