import numpy as np, wave

SR = 44100
BPM = 96
BARS = 16
SPB = SR * 60 / BPM
eighth = int(SPB / 2)

# ---------- utilities ----------
def softclip(x, d=1.0): return np.tanh(x * d)
def norm(x, p=0.98): m = np.max(np.abs(x)); return x * (p/m) if m > 0 else x

def lpf_ma(x, taps):
    if taps <= 1: return x
    k = np.ones(taps) / taps
    return np.convolve(x, k, mode="same")

def hpf_ma(x, taps): return x - lpf_ma(x, taps)

def env(L, g, a=5, d=120, s=0.6, r=200):
    A, D, R = int(a*SR/1000), int(d*SR/1000), int(r*SR/1000)
    e = np.zeros(L); on = min(L, g)
    e[:min(A,on)] = np.linspace(0,1,min(A,on),endpoint=False)
    i = A
    dlen = max(0, min(D, on - i))
    if dlen > 0:
        e[i:i+dlen] = np.linspace(1, s, dlen, endpoint=False)
        i += dlen
    if on > i: e[i:on] = s
    if L > on and R > 0:
        e[on:on+R] = np.linspace(e[on-1] if on>0 else 0, 0, min(R, L - on))
    return e

def saw(f,N,ph=0): t=np.arange(N)/SR; return 2*((f*t+ph/(2*np.pi))%1-0.5)
def sqr(f,N,d=0.5,ph=0): t=np.arange(N)/SR; return np.where(np.sin(2*np.pi*f*t+ph)>=np.cos(np.pi*(1-d)),1,-1)
def sine(f,N,ph=0): t=np.arange(N)/SR; return np.sin(2*np.pi*f*t+ph)

# ---------- shimmer pieces (fixed) ----------
def ensure_stereo(x):
    """Return a 2D (N,2) array."""
    if x.ndim == 1: 
        return np.column_stack([x, x])
    return x

def mtap(x, del_ms=(90,140,200,260,340), gains=(.28,.23,.19,.16,.12), mix=.35, pre=28):
    x = ensure_stereo(x)
    N = x.shape[0]                 # always defined
    out = x * (1 - mix)
    pre_samp = int(pre * SR / 1000)
    acc = np.zeros_like(x)
    for dms, g in zip(del_ms, gains):
        d = pre_samp + int(dms * SR / 1000)
        if d >= N: 
            continue
        pad = np.zeros((d, 2))
        acc += g * np.vstack([pad, x[:-d]])
    return out + mix * acc

def pitch_up(sig, semitones=12.0):
    sig = ensure_stereo(sig)
    N = sig.shape[0]               # always defined
    ratio = 2 ** (semitones / 12.0)
    n_fast = max(1, int(N / ratio))
    idx_fast = np.linspace(0, N-1, num=n_fast)
    tN = np.arange(N)
    fast = np.zeros((n_fast, 2))
    for ch in range(2):
        fast[:, ch] = np.interp(idx_fast, tN, sig[:, ch])
    # stretch back to N
    idx_back = np.linspace(0, n_fast-1, num=N)
    tf = np.arange(n_fast)
    up = np.zeros((N, 2))
    for ch in range(2):
        up[:, ch] = np.interp(idx_back, tf, fast[:, ch])
    return up

def shimmer(x, amt=0.16):
    x = ensure_stereo(x)
    taps_hp = max(3, int(SR/1200))   # ~1.2kHz HP proxy
    hp = np.column_stack([
        hpf_ma(x[:,0], taps_hp),
        hpf_ma(x[:,1], taps_hp)
    ])
    up = pitch_up(hp, 12.0)
    cloud = mtap(up, mix=0.6)
    return cloud * amt

# ---------- arrangement ----------
A2,G2,F2,E2,E4,A4 = 110.0,98.0,87.31,82.41,329.63,440.0
pattern = [A2,A2,A2,A2, G2,G2,A2,A2, F2,F2,F2,F2, E2,E2,E2,E2]
gate = int(0.9 * eighth)
L = int(SPB * 4 * BARS)

audio = np.zeros((L, 2))

# Kick (quarters)
beats_total = int(L / SPB)
for b in range(beats_total):
    n0 = int(b * SPB)
    n1 = min(L, n0 + int(0.20 * SR))
    N = n1 - n0
    t = np.arange(N) / SR
    f = 80 * np.exp(-t*14) + 40
    s = np.sin(2*np.pi*np.cumsum(f)/SR)
    s = lpf_ma(s * np.linspace(1,0,N), max(3, int(SR/3000)))
    s = softclip(s, 2.0) * 0.56
    audio[n0:n1] += s[:, None]

# Bass ostinato
for step in range(L // eighth):
    n0 = step * eighth
    n1 = min(L, n0 + eighth)
    seg = n1 - n0
    note = pattern[step % len(pattern)]
    g = min(gate, seg)
    o = 0.6 * saw(note, seg) + 0.4 * sqr(note*1.005, seg, 0.48)
    e = env(seg, g, 5, 80, 0.7, 60)
    l = softclip(o * e, 1.7)
    # slight stereo offset
    r = l if seg <= 24 else np.concatenate([np.zeros(12), l[:-12]])
    audio[n0:n1, 0] += l * 0.80
    audio[n0:n1, 1] += r * 0.80

# Snare on 3
for b in range(0, beats_total, 4):
    n0 = int((b + 2) * SPB)
    n1 = min(L, n0 + int(0.22 * SR))
    N = n1 - n0
    nz = np.random.uniform(-1, 1, N) * np.linspace(1, 0, N)
    bd = sine(190, N) * np.linspace(1, 0, N)
    s = 0.8 * hpf_ma(nz, 2000) + 0.5 * bd
    s = softclip(s, 1.4) * 0.50
    # tiny stereo skew
    audio[n0:n1, 0] += s * 0.9
    audio[n0:n1, 1] += (np.concatenate([np.zeros(2), s[:-2]]) if N > 2 else s) * 1.1

# Hats (8ths, random density)
for i in range(beats_total * 2):
    if np.random.rand() > 0.95: 
        continue
    n0 = int(i * eighth)
    n1 = min(L, n0 + int(0.05 * SR))
    N = n1 - n0
    nz = np.random.uniform(-1, 1, N)
    s = hpf_ma(nz, 6000) * np.linspace(1, 0, N)
    s = softclip(s, 1.1) * 0.17
    audio[n0:n1] += s[:, None] * np.array([0.7, 1.0])

# Pad
pad = 0.6 * (saw(E4, L) + saw(E4 * 0.993, L, np.pi/3))
pad = lpf_ma(pad, 24) * np.linspace(0, 1, L) * 0.25
padR = np.concatenate([np.zeros(10), pad[:-10]])
P = np.column_stack([pad, padR])
audio += P

# Lead (short motif just to introduce tension)
motif = [
    (1.0, E4, 0.75), (2.0, F2*4, 0.5), (3.0, A4, 0.5),
    (5.0, 392.0, 0.5), (5.5, 440.0, 0.5), (6.0, E4, 0.75),
]
lead = np.zeros((L, 2))
for beat, fq, dur in motif:
    n0 = int(beat * SPB)
    seg = int(dur * SPB)
    n1 = min(L, n0 + seg)
    N = n1 - n0
    base = 0.55 * sqr(fq, N, 0.48) + 0.45 * saw(fq * 0.997, N)
    e = env(N, int(0.85 * seg), 8, 120, 0.7, 200)
    s = lpf_ma(base, 14) * e
    s = softclip(s, 1.2) * 0.30
    lead[n0:n1, 0] += s
    lead[n0:n1, 1] += s * 0.92
audio += lead

# Light "haunting" shimmer on pad+lead bus
send = P * 0.6 + lead * 0.8
audio += shimmer(send, amt=0.15)

# Master
audio = softclip(audio, 1.15)
audio = norm(audio, 0.97)

with wave.open("carpenter_stage2.wav", "wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(np.int16(np.clip(audio, -1, 1) * 32767).tobytes())

print("wrote carpenter_stage2.wav")

