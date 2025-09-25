# logistic_three_lines_drone.py
# Three logistic-map "lines" (r=2.9, 3.5, 3.9) driving a Hoover-tinged supersaw drone
# Pure NumPy + stdlib. Writes "logistic_three_lines_drone.wav"

import numpy as np
import wave, struct

# ---------------------------- Config ----------------------------
SR = 44100            # sample rate
DUR_SEC = 360         # length in seconds (e.g., 6 minutes)
CTRL_HZ = 200         # control-rate for parameter curves (fast & smooth)
VOICES = 7            # supersaw voices per layer
R_VALUES = [2.9, 3.5, 3.9]   # three growth rates (stable → chaotic-ish)
X0 = 0.5              # logistic seed (tie to “middle C” conceptually)

# Stereo placement of the three lines (constant-power pan, -1..+1)
PANS = [-0.6, 0.1, 0.7]

# Output filename
OUT_WAV = "logistic_three_lines_drone.wav"

# ------------------------- Helpers ------------------------------
N = SR * DUR_SEC
t = np.arange(N) / SR
M = int(DUR_SEC * CTRL_HZ)
tc = np.linspace(0.0, DUR_SEC, M)

def logistic_series_ctrl(r, x0, M):
    """Logistic map at control rate."""
    x = np.empty(M, dtype=np.float64)
    x[0] = x0
    for n in range(M - 1):
        x[n + 1] = r * x[n] * (1.0 - x[n])
    return x

def moving_avg(x, win):
    if win <= 1: return x
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, k, mode="same")

def map01(x, lo, hi):
    return lo + (hi - lo) * np.clip(x, 0.0, 1.0)

def cents_to_ratio(c):  # c in cents
    return 2.0 ** (c / 1200.0)

def saw_from_phase(phase):
    # phase in [0,1)
    return 2.0 * (phase - np.floor(phase + 0.5))

def pulse_from_phase(phase, duty):
    return np.where(np.mod(phase, 1.0) < duty, 1.0, -1.0)

# --------------------- Layer / “line” builder -------------------
def make_layer(r, pan_bias=0.0, seed=0, voices=7):
    """
    Build one logistic-driven supersaw layer with fixed growth rate r.
    The same logistic stream drives all key parameters.
    """
    rng = np.random.default_rng(100 + seed)

    # 1) Logistic at control rate + multi-timescale smoothing
    x = logistic_series_ctrl(r, X0, M)
    slow = moving_avg(x, int(1.4 * CTRL_HZ))  # ~1.4 s
    med  = moving_avg(x, int(0.30 * CTRL_HZ)) # ~0.3 s
    fast = moving_avg(x, int(0.06 * CTRL_HZ)) # ~60 ms

    # 2) Map to synth parameters (still at control rate)
    base_hz   = map01(slow, 39.0, 96.0)    # E1..G2-ish
    detune_ct = map01(med,  6.0, 28.0)     # cents spread across voices
    vib_hz    = map01(fast, 0.04, 0.55)
    vib_ct    = map01(med,  0.2, 6.0)
    pwm_depth = map01(med,  0.00, 0.45)
    pwm_hz    = map01(fast, 0.02, 0.18)
    amp_env   = map01(slow, 0.78, 1.00)    # slow “breathing”
    drive     = map01(med,  0.80, 1.25)
    haas_ms   = map01(slow, 0.0, 12.0)     # stereo width (Haas delay)

    # 3) Optional “event” gates from extremes (musical structure cues)
    sub_gate   = moving_avg((x > 0.82).astype(float), int(0.15 * CTRL_HZ))
    noise_gate = moving_avg((x > 0.92).astype(float), int(0.10 * CTRL_HZ))

    # 4) Upsample control streams to audio rate
    def up(v): return np.interp(t, tc, v)
    base_hz, detune_ct, vib_hz, vib_ct = map(up, (base_hz, detune_ct, vib_hz, vib_ct))
    pwm_depth, pwm_hz, amp_env, drive, haas_ms = map(up, (pwm_depth, pwm_hz, amp_env, drive, haas_ms))
    sub_gate, noise_gate = map(up, (sub_gate, noise_gate))

    # 5) Supersaw voices
    layer = np.zeros(N, dtype=np.float64)
    vibrato = np.sin(2*np.pi*vib_hz*t) * vib_ct
    # symmetric detune factors per-voice
    spread = np.linspace(-1.0, 1.0, voices)
    for d in spread:
        cents = d * detune_ct
        ratio = cents_to_ratio(cents + vibrato)
        freq  = base_hz * ratio
        phase = (rng.random() + np.cumsum(freq / SR)) % 1.0
        layer += saw_from_phase(phase)
    layer /= voices

    # 6) Subtle Hoover character: PWM octave
    duty = 0.5 + 0.5 * pwm_depth * np.sin(2*np.pi*pwm_hz*t + 0.31 * seed)
    phase_pwm = (np.cumsum(2.0 * base_hz / SR)) % 1.0
    pwm = pulse_from_phase(phase_pwm, duty)
    layer = 0.88 * layer + 0.12 * pwm

    # 7) Add sub-octave & airy noise when logistic is “hot”
    phase_sub = (np.cumsum(0.5 * base_hz / SR)) % 1.0
    sub = saw_from_phase(phase_sub) * sub_gate
    noise = (rng.normal(0, 1, N) * 0.06) * noise_gate
    layer = layer + 0.22 * sub + noise

    # 8) Nonlinear drive + slow amplitude breathing
    layer = np.tanh(drive * layer) * amp_env

    # 9) Haas stereo widening (R delayed by a small amount)
    d_samp = int(np.mean(haas_ms) * 0.001 * SR)
    L = layer.copy(); R = layer.copy()
    if d_samp > 0:
        R = np.concatenate([np.zeros(d_samp), R[:-d_samp]])

    # 10) Constant-power pan for this line
    pan = np.clip(pan_bias, -1.0, 1.0)
    gL = np.sqrt((1.0 - pan) * 0.5)
    gR = np.sqrt((1.0 + pan) * 0.5)
    return gL * L, gR * R

# ------------------------- Render mix ---------------------------
L_mix = np.zeros(N, dtype=np.float64)
R_mix = np.zeros(N, dtype=np.float64)

for i, (r, pan) in enumerate(zip(R_VALUES, PANS)):
    Li, Ri = make_layer(r=r, pan_bias=pan, seed=i, voices=VOICES)
    L_mix += Li
    R_mix += Ri

# Fades
fade_sec = 8.0
fadeN = int(fade_sec * SR)
fade_in  = np.linspace(0, 1, fadeN)
fade_out = np.linspace(1, 0, fadeN)
L_mix[:fadeN] *= fade_in;  R_mix[:fadeN] *= fade_in
L_mix[-fadeN:] *= fade_out; R_mix[-fadeN:] *= fade_out

# Leveling to -18 dBFS RMS + gentle safety
stereo = np.stack([L_mix, R_mix], axis=1)
rms = np.sqrt(np.mean(stereo**2))
target_db = -18.0
gain = 10 ** ((target_db - 20*np.log10(max(rms, 1e-12))) / 20.0)
stereo *= gain
stereo = np.tanh(stereo * 1.05)

# Write WAV (16-bit PCM)
with wave.open(OUT_WAV, "w") as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    ints = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    wf.writeframes(struct.pack("<" + "h" * ints.size, *ints.flatten()))

print("Wrote:", OUT_WAV)
