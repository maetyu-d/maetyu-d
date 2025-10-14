import numpy as np, wave

# ====== Config ======
SR   = 44100
BPM  = 96
BARS = 16
SPB  = SR * 60 / BPM
eighth    = int(SPB/2)
sixteenth = int(SPB/4)

# ====== Safe, length-stable utilities ======
def softclip(x, d=1.0):
    return np.tanh(x * d)

def norm(x, peak=0.98):
    m = np.max(np.abs(x))
    return x * (peak/m) if m > 0 else x

def fit1d(sig, seg_len):
    """Trim or pad a 1D array to seg_len samples."""
    sig = np.asarray(sig, dtype=float)
    n = sig.shape[0]
    if n == seg_len:
        return sig
    if n > seg_len:
        return sig[:seg_len]
    # pad with zeros
    return np.pad(sig, (0, seg_len - n))

def one_pole_lp(x, fc_hz, sr=SR):
    """First-order low-pass (stable, len-preserving)."""
    if fc_hz <= 0:
        return np.zeros_like(x, dtype=float)
    a = np.exp(-2.0 * np.pi * fc_hz / sr)
    b = 1.0 - a
    y = 0.0
    out = np.zeros_like(x, dtype=float)
    for n, xn in enumerate(x):
        y = b * xn + a * y
        out[n] = y
    return out

def one_pole_hp(x, fc_hz, sr=SR):
    """First-order high-pass via complement (len-preserving)."""
    if fc_hz <= 0:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    return x - one_pole_lp(x, fc_hz, sr)

def env(L, g, a=5, d=120, s=0.6, r=200):
    A, D, R = int(a*SR/1000), int(d*SR/1000), int(r*SR/1000)
    e = np.zeros(L); on = min(L, g)
    e[:min(A,on)] = np.linspace(0,1,min(A,on),endpoint=False)
    i=A
    dlen = max(0, min(D, on-i))
    if dlen>0:
        e[i:i+dlen] = np.linspace(1, s, dlen, endpoint=False)
        i += dlen
    if on>i: e[i:on] = s
    if L>on and R>0:
        e[on:on+R] = np.linspace(e[on-1] if on>0 else 0, 0, min(R, L-on))
    return e

def saw(f, N, ph=0.0):
    t = np.arange(N)/SR
    return 2.0 * ((f*t + ph/(2*np.pi)) % 1.0 - 0.5)

def sqr(f, N, duty=0.5, ph=0.0):
    t = np.arange(N)/SR
    return np.where(np.sin(2*np.pi*f*t + ph) >= np.cos(np.pi*(1-duty)), 1.0, -1.0)

def sine(f, N, ph=0.0):
    t = np.arange(N)/SR
    return np.sin(2*np.pi*f*t + ph)

def ensure_stereo(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.column_stack([x, x])
    return x

# ====== Shimmer (stable; 3s tail) ======
def multitap_cloud_long(x, mix=0.7, pre_ms=30):
    """Multi-tap reverb with taps up to ~3s."""
    # Taps & gains shaped for a long Lexicon-style wash
    delays_ms = [90, 140, 220, 350, 550, 900, 1300, 1800, 2400, 3000]
    gains     = [0.34,0.30,0.26,0.22,0.18,0.15, 0.12, 0.10, 0.08, 0.06]
    x = ensure_stereo(x)
    N = x.shape[0]
    out = x * (1 - mix)
    pre = int(pre_ms * SR / 1000)
    acc = np.zeros_like(x)
    for d_ms, g in zip(delays_ms, gains):
        d = pre + int(d_ms * SR / 1000)
        if d >= N:
            continue
        pad = np.zeros((d,2))
        acc += g * np.vstack([pad, x[:-d]])
    return out + mix * acc

def pitch_up_resample(sig, semitones=12.0):
    sig = ensure_stereo(sig)
    N = sig.shape[0]
    if N < 2:
        return sig.copy()
    ratio    = 2 ** (semitones / 12.0)
    n_fast   = max(2, int(np.ceil(N / ratio)))
    idx_fast = np.linspace(0, N-1, num=n_fast)
    tN       = np.arange(N)
    fast     = np.empty((n_fast, 2))
    for ch in range(2):
        fast[:, ch] = np.interp(idx_fast, tN, sig[:, ch])
    idx_back = np.linspace(0, n_fast-1, num=N)
    tf       = np.arange(n_fast)
    up       = np.empty((N,2))
    for ch in range(2):
        up[:, ch] = np.interp(idx_back, tf, fast[:, ch])
    return up

def shimmer_long(send_stereo, amount=0.38, hp_hz=1200.0, detune_cents=6.0):
    """Vintage shimmer with octave and slight detune, ~3s tail."""
    s = ensure_stereo(send_stereo)
    # high-pass -> octave up (3 voices slight detune) -> long cloud
    hpL = one_pole_hp(s[:,0], hp_hz, SR)
    hpR = one_pole_hp(s[:,1], hp_hz, SR)
    hp  = np.column_stack([hpL, hpR])
    # detuned stack
    up1 = pitch_up_resample(hp, 12.0)
    up2 = pitch_up_resample(hp, 12.0 + detune_cents/100.0*12.0)
    up3 = pitch_up_resample(hp, 12.0 - detune_cents/100.0*12.0)
    up  = 0.5*up1 + 0.3*up2 + 0.2*up3
    cloud = multitap_cloud_long(up, mix=0.7, pre_ms=30)
    return cloud * amount

# ====== Musical setup ======
A2, G2, F2, E2 = 110.0, 98.0, 87.31, 82.41
A3, D4, E4, G4, A4 = 220.0, 293.66, 329.63, 392.0, 440.0

pattern = [A2,A2,A2,A2, G2,G2,A2,A2, F2,F2,F2,F2, E2,E2,E2,E2]
gate    = int(0.9 * eighth)
L       = int(SPB * 4 * BARS)

mix = np.zeros((L,2))

# ====== Drums ======
beats_total = int(L / SPB)

# Kick (quarters)
for b in range(beats_total):
    n0 = int(b * SPB); n1 = min(L, n0 + int(0.20*SR)); seg = n1-n0
    t  = np.arange(seg)/SR
    f  = 80*np.exp(-t*14) + 40
    s  = np.sin(2*np.pi*np.cumsum(f)/SR)
    s  = one_pole_lp(s * np.linspace(1,0,seg), 3000.0, SR)
    s  = fit1d(s, seg)
    s  = softclip(s, 2.1) * 0.68
    mix[n0:n1] += s[:,None]

# Snare (on 3 + extra ghost at 3.5 every other bar)
snare_beats = [b+2 for b in range(0, beats_total, 4)]
snare_beats += [b+3.5 for b in range(0, beats_total, 8)]
for sb in snare_beats:
    n0 = int(sb * SPB); n1 = min(L, n0 + int(0.22*SR)); seg = n1-n0
    nz = np.random.uniform(-1,1,seg) * np.linspace(1,0,seg)
    bd = sine(190, seg) * np.linspace(1,0,seg)
    s  = 0.8 * one_pole_hp(nz, 1800.0, SR) + 0.5 * bd
    s  = fit1d(s, seg)
    s  = softclip(s, 1.5) * 0.66
    mix[n0:n1,0] += s*0.9
    mix[n0:n1,1] += (np.concatenate([np.zeros(2), s[:-2]]) if seg>2 else s) * 1.1

# Hi-hats (16ths, dense with accents)
total_16 = L // sixteenth
for i in range(total_16):
    if np.random.rand() > 0.92:
        continue
    n0 = i * sixteenth; n1 = min(L, n0 + int(0.045*SR)); seg = n1-n0
    nz = np.random.uniform(-1,1,seg)
    s  = one_pole_hp(nz, 8000.0, SR) * np.linspace(1,0,seg)
    if i % 4 == 2:  # accent the "e" of the beat
        s *= 1.2
    s  = fit1d(s, seg)
    s  = softclip(s, 1.15) * 0.22
    mix[n0:n1,0] += s*0.7
    mix[n0:n1,1] += s

# ====== Bass layers ======
# Main bass (saw/square blend, widening over time)
amp_profile = np.linspace(0.65, 1.12, L)
for step in range(L // eighth):
    n0 = step*eighth; n1 = min(L, n0+eighth); seg = n1-n0
    note = pattern[step % len(pattern)]
    g = min(gate, seg)
    o = 0.6*saw(note, seg) + 0.4*sqr(note*1.005, seg, 0.48)
    e = env(seg, g, 5, 80, 0.7, 60)
    l = softclip(o*e*amp_profile[n0:n1], 2.0)
    r = l if seg <= 12 else np.concatenate([np.zeros(12), l[:-12]])
    l = fit1d(l, seg); r = fit1d(r, seg)
    mix[n0:n1,0] += l*0.85
    mix[n0:n1,1] += r*0.85

# Sub bass (sine, gentle LP for roundness)
for step in range(L // eighth):
    n0 = step*eighth; n1 = min(L, n0+eighth); seg = n1-n0
    note = pattern[step % len(pattern)] / 2.0
    s = sine(note, seg) * env(seg, int(0.85*seg), 3, 80, 0.7, 90)
    s = one_pole_lp(s, 90.0, SR)
    s = fit1d(s, seg)
    mix[n0:n1,0] += s*0.24
    mix[n0:n1,1] += s*0.24

# Octave stabs for aggression
for b in range(beats_total):
    n0 = int(b*SPB); n1 = min(L, n0 + int(0.18*SR)); seg = n1-n0
    s  = saw(A2*2, seg) * env(seg, int(0.15*seg), 4, 120, 0.0, 80)
    s  = one_pole_lp(s, 600.0, SR)
    s  = fit1d(s, seg)
    s  = softclip(s, 1.3) * 0.26
    mix[n0:n1,0] += s
    mix[n0:n1,1] += s

# ====== Pad ======
pad = 0.6*(saw(A4, L) + saw(A4*0.992, L, np.pi/3))
pad = one_pole_lp(pad, 1200.0, SR) * np.linspace(0,1,L) * 0.30
padR = np.concatenate([np.zeros(10), pad[:-10]])
PAD  = np.column_stack([pad, padR])
mix += PAD

# ====== Evolving lead ======
def rand_lfo(N, depth_cents=30.0, rate_hz=0.25):
    t = np.arange(N)/SR
    sine_lfo = np.sin(2*np.pi*rate_hz*t)
    noise = np.random.randn(N)
    # mild smooth
    noise = one_pole_lp(noise, 8.0, SR)
    blend = (0.6*sine_lfo + 0.4*noise/np.max(np.abs(noise))) * (depth_cents/100.0)
    ratio = 2 ** (blend / 12.0)
    return ratio

lead = np.zeros((L,2))
motif = [
    (1.0,  E4,   0.75), (2.0,  F2*4, 0.5), (3.0,  A4,   0.5),
    (5.0,  G4,   0.5),  (5.5, A4,    0.5), (6.0,  E4,   0.75),
    (9.0,  D4,   0.5),  (10.0,E4,    0.5), (11.0, G4,   0.5),
    (11.5, A4,   0.5),  (13.0,E4,    0.75),(14.0, F2*4, 0.5), (15.0, A4, 0.5),
]
for beat, fq, dur in motif:
    n0 = int(beat*SPB); seg = int(dur*SPB); n1 = min(L, n0+seg)
    r  = rand_lfo(seg, depth_cents=30.0, rate_hz=0.25)
    base = (0.50*sqr(fq*r, seg, 0.45) +
            0.35*saw(fq*0.997*r, seg) +
            0.15*sqr(fq*1.01*r, seg, 0.42))
    e    = env(seg, int(0.85*seg), 8, 120, 0.7, 240)
    s    = one_pole_lp(base, 1400.0, SR) * e
    s    = fit1d(s, seg)
    s    = softclip(s, 1.35) * 0.42
    # ping-pong pan
    if int(beat) % 2 == 0:
        lead[n0:n1,0] += s
        lead[n0:n1,1] += s * 0.7
    else:
        lead[n0:n1,0] += s * 0.7
        lead[n0:n1,1] += s
mix += lead

# ====== 3s Shimmer on pad+lead bus ======
send = PAD*0.7 + lead*1.0
mix  = mix + shimmer_long(send, amount=0.38, hp_hz=1200.0, detune_cents=6.0)

# ====== Master ======
mix = softclip(mix, 1.25)
mix = norm(mix, 0.97)

with wave.open("carpenter_stage3.wav", "wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(np.int16(np.clip(mix, -1, 1) * 32767).tobytes())

print("wrote carpenter_stage3.wav")
