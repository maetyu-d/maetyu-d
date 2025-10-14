import numpy as np, wave, os

# =========================
# Global config
# =========================
SR   = 44100
BPM  = 96
BARS = 16
SPB  = SR * 60 / BPM
EIGHTH    = int(SPB / 2)
SIXTEENTH = int(SPB / 4)
TRIPLET   = int(SPB / 3)  # quarter-note triplet step

# =========================
# Safe DSP helpers
# =========================
def softclip(x, d=1.0):
    return np.tanh(x * d)

def norm(x, peak=0.98):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x

def fit1d(sig, n):
    sig = np.asarray(sig, dtype=float)
    L = sig.shape[0]
    if L == n: return sig
    if L >  n: return sig[:n]
    return np.pad(sig, (0, n - L))

def ensure_stereo(x):
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, x]) if x.ndim == 1 else x

def one_pole_lp(x, fc_hz, sr=SR):
    if fc_hz <= 0: return np.zeros_like(x, dtype=float)
    a = np.exp(-2.0 * np.pi * fc_hz / sr)
    b = 1.0 - a
    y = 0.0
    out = np.zeros_like(x, dtype=float)
    for i, xn in enumerate(x):
        y = b * xn + a * y
        out[i] = y
    return out

def one_pole_hp(x, fc_hz, sr=SR):
    if fc_hz <= 0: return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    return x - one_pole_lp(x, fc_hz, sr)

def env(L, g, a=5, d=120, s=0.6, r=200):
    """ADSR in ms; g = gate length in samples"""
    A, D, R = int(a*SR/1000), int(d*SR/1000), int(r*SR/1000)
    e = np.zeros(L); on = min(L, g)
    e[:min(A,on)] = np.linspace(0, 1, min(A,on), endpoint=False)
    i = A
    dlen = max(0, min(D, on - i))
    if dlen > 0:
        e[i:i+dlen] = np.linspace(1, s, dlen, endpoint=False)
        i += dlen
    if on > i:
        e[i:on] = s
    if L > on and R > 0:
        e[on:on+R] = np.linspace(e[on-1] if on > 0 else 0, 0, min(R, L - on))
    return e

def sine(f, N, ph=0.0):
    t = np.arange(N) / SR
    return np.sin(2*np.pi*f*t + ph)

def saw(f, N, ph=0.0):
    t = np.arange(N) / SR
    return 2.0 * ((f*t + ph/(2*np.pi)) % 1.0 - 0.5)

def sqr(f, N, duty=0.5, ph=0.0):
    t = np.arange(N) / SR
    return np.where(np.sin(2*np.pi*f*t + ph) >= np.cos(np.pi*(1-duty)), 1.0, -1.0)

# =========================
# Shimmer (light and long)
# =========================
def multitap_cloud(x, mix=0.6, pre_ms=28, delays_ms=(90,140,200,260,340), gains=(.28,.23,.19,.16,.12)):
    x = ensure_stereo(x)
    N = x.shape[0]
    out = x * (1 - mix)
    pre = int(pre_ms * SR / 1000)
    acc = np.zeros_like(x)
    for d_ms, g in zip(delays_ms, gains):
        d = pre + int(d_ms * SR / 1000)
        if d >= N: continue
        pad = np.zeros((d, 2))
        acc += g * np.vstack([pad, x[:-d]])
    return out + mix * acc

def multitap_cloud_long(x, mix=0.7, pre_ms=30):
    delays = [90,140,220,350,550,900,1300,1800,2400,3000]
    gains  = [0.34,0.30,0.26,0.22,0.18,0.15,0.12,0.10,0.08,0.06]
    return multitap_cloud(x, mix=mix, pre_ms=pre_ms, delays_ms=delays, gains=gains)

def pitch_up_resample(sig, semitones=12.0):
    sig = ensure_stereo(sig)
    N = sig.shape[0]
    if N < 2:
        return sig.copy()
    ratio   = 2 ** (semitones / 12.0)
    n_fast  = max(2, int(np.ceil(N / ratio)))
    idx_fast = np.linspace(0, N-1, num=n_fast)
    tN      = np.arange(N)
    fast = np.empty((n_fast, 2))
    for ch in range(2):
        fast[:, ch] = np.interp(idx_fast, tN, sig[:, ch])
    # stretch back
    idx_back = np.linspace(0, n_fast-1, num=N)
    tf = np.arange(n_fast)
    up = np.empty((N, 2))
    for ch in range(2):
        up[:, ch] = np.interp(idx_back, tf, fast[:, ch])
    return up

def shimmer_light(send, amount=0.15):
    s = ensure_stereo(send)
    hp = np.column_stack([one_pole_hp(s[:,0], 1200.0), one_pole_hp(s[:,1], 1200.0)])
    up = pitch_up_resample(hp, 12.0)
    return multitap_cloud(up, mix=0.6) * amount

def shimmer_long(send, amount=0.50, detune_cents=9.0, hp_hz=1200.0):
    s  = ensure_stereo(send)
    hp = np.column_stack([one_pole_hp(s[:,0], hp_hz), one_pole_hp(s[:,1], hp_hz)])
    up1 = pitch_up_resample(hp, 12.0)
    up2 = pitch_up_resample(hp, 12.0 + detune_cents/100.0*12.0)
    up3 = pitch_up_resample(hp, 12.0 - detune_cents/100.0*12.0)
    up  = 0.5*up1 + 0.3*up2 + 0.2*up3
    return multitap_cloud_long(up, mix=0.7) * amount

# =========================
# IO helper
# =========================
def write_wav(path, x):
    x = ensure_stereo(x)
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(np.int16(x * 32767).tobytes())

# =========================
# Notes
# =========================
A2, G2, F2, E2 = 110.0, 98.0, 87.31, 82.41
A3, D4, E4, G4, A4 = 220.0, 293.66, 329.63, 392.0, 440.0

# =========================
# Triplet synth generator (Stages 1 & 2)
# =========================
def render_triplet_synth(L, scale_notes=(E4, G4, A4), fc=1600.0, gain=0.12, stereo_offset=8):
    """Light triplet arpeggio: subtle Carpenter pulse with gentle LP and tiny stereo delay."""
    outL = np.zeros(L); outR = np.zeros(L)
    gate = int(TRIPLET * 0.85)
    idx = 0
    step = TRIPLET
    trip_seq = list(scale_notes)
    s_i = 0
    while idx < L:
        n0 = idx
        n1 = min(L, n0 + step)
        seg = n1 - n0
        f  = trip_seq[s_i % len(trip_seq)]
        s_i += 1
        o = 0.5 * saw(f, seg) + 0.5 * sine(f, seg)
        e = env(seg, min(gate, seg), a=5, d=80, s=0.6, r=120)
        v = one_pole_lp(o, fc) * e * gain
        v = fit1d(v, seg)
        outL[n0:n1] += v
        # tiny inter-channel offset for width
        if seg > stereo_offset:
            outR[n0+stereo_offset:n1] += v[:-stereo_offset]
        else:
            outR[n0:n1] += v
        idx += step
    return np.column_stack([outL, outR])

# =========================
# MBV-style generators
# =========================
def mbv_drone_distant(L, base_freq=E4, fc=1600.0, gain=0.10):
    """
    Distant MBV drone: detuned saw stack, gentle vibrato, filtered, slow tremolo, long fade-in.
    """
    t = np.arange(L)/SR
    vibr = 2 ** ( (np.sin(2*np.pi*0.3*t) * 5.0) / 1200.0 )  # ~5 cents vibrato
    voices = []
    detunes = [1.0, 0.997, 1.003]
    for d in detunes:
        v = saw(base_freq*d*vibr, L, ph=np.random.rand()*np.pi*2)
        voices.append(v)
    s = sum(voices) / len(voices)
    s = one_pole_lp(s, fc)
    trem = 0.5 + 0.5*np.sin(2*np.pi*0.15*t + 1.0)  # slow amp trem
    fade = np.linspace(0, 1, L) ** 2
    mono = s * trem * fade * gain
    # mild stereo smear
    right = np.concatenate([np.zeros(20), mono[:-20]]) if L > 20 else mono
    return np.column_stack([mono*0.9, right])

def mbv_wall_huge(L, base_freq=A4, fc=1800.0, gain=0.35):
    """
    Huge MBV wall: more detuned voices, asymmetric channel detune, extra saturation and width.
    """
    t = np.arange(L)/SR
    vibrL = 2 ** ( (np.sin(2*np.pi*0.35*t) * 9.0) / 1200.0 )
    vibrR = 2 ** ( (np.sin(2*np.pi*0.31*t + 0.7) * 9.0) / 1200.0 )
    detunes = [0.988, 0.995, 1.0, 1.005, 1.012]
    left = np.zeros(L); right = np.zeros(L)
    for d in detunes:
        left  += saw(base_freq*d*vibrL, L, ph=np.random.rand()*np.pi*2)
        right += saw(base_freq*d*vibrR, L, ph=np.random.rand()*np.pi*2)
    left  = one_pole_lp(left/len(detunes),  fc)
    right = one_pole_lp(right/len(detunes), fc)
    # subtle channel delays for width
    dL, dR = 18, 26
    leftD  = left
    rightD = np.concatenate([np.zeros(dR), right[:-dR]]) if L > dR else right
    # gentle tilt EQ via HP/LP pair
    leftD  = one_pole_hp(leftD, 120.0);  rightD = one_pole_hp(rightD, 120.0)
    wall = np.column_stack([leftD, rightD]) * gain
    wall = softclip(wall, 1.3)
    return wall

# =========================
# STAGE 1 — Minimal Dread (+ light triplet)
# =========================
def render_stage1():
    pattern = [A2, A2, G2, F2, E2, E2, F2, G2]
    L = int(SPB * 4 * BARS)
    beats_total = int(L / SPB)
    out_dir = "stems_stage1"; os.makedirs(out_dir, exist_ok=True)

    # Stems
    st_kick   = np.zeros((L, 2))
    st_bass   = np.zeros((L, 2))
    st_pad    = np.zeros((L, 2))
    st_trip   = np.zeros((L, 2))  # NEW: triplet synth

    # Kick only on downbeats (beats 1 & 3 each bar)
    for b in range(0, beats_total, 2):
        n0 = int(b * SPB); n1 = min(L, n0 + int(0.22 * SR)); seg = n1 - n0
        t  = np.arange(seg) / SR; f = 70*np.exp(-t*14) + 35
        s  = np.sin(2*np.pi*np.cumsum(f)/SR)
        s  = one_pole_lp(s * np.linspace(1,0,seg), 2400.0); s = fit1d(s, seg)
        s  = softclip(s, 1.7) * 0.48
        st_kick[n0:n1] += s[:, None] * 0.8

    # Very dry bass (half density)
    gate = int(0.85 * EIGHTH)
    for i in range(L // (2 * EIGHTH)):
        n0 = i * 2 * EIGHTH; n1 = min(L, n0 + 2 * EIGHTH); seg = n1 - n0
        f  = pattern[i % len(pattern)]
        o  = 0.65 * saw(f, seg) + 0.35 * sqr(f * 0.997, seg, 0.5)
        e  = env(seg, gate, 10, 160, 0.55, 240)
        s  = softclip(one_pole_lp(o, 800.0) * e, 1.2) * 0.55
        s  = fit1d(s, seg)
        st_bass[n0:n1, 0] += s * 0.9
        st_bass[n0:n1, 1] += s * 0.85

    # Soft pad (low)
    pad = 0.5 * (saw(A3, L) + saw(A3 * 0.995, L, np.pi/3))
    pad = one_pole_lp(pad, 900.0) * np.linspace(0, 1, L) * 0.15
    padR = np.concatenate([np.zeros(12), pad[:-12]])
    st_pad += np.column_stack([pad*0.9, padR*0.8])

    # NEW: light triplet arpeggio (very subtle)
    st_trip += render_triplet_synth(L, scale_notes=(E4, G4, A4), fc=1500.0, gain=0.10, stereo_offset=8)

    # Mix
    mix = st_kick + st_bass + st_pad + st_trip
    mix = softclip(mix, 1.08); mix = norm(mix, 0.90)

    # Write stems + mix
    write_wav(os.path.join(out_dir, "kick.wav"),   st_kick)
    write_wav(os.path.join(out_dir, "bass.wav"),   st_bass)
    write_wav(os.path.join(out_dir, "pad.wav"),    st_pad)
    write_wav(os.path.join(out_dir, "triplet.wav"),st_trip)
    write_wav("carpenter_stage1.wav", mix)
    write_wav(os.path.join(out_dir, "FULL_mix.wav"), mix)
    print("Stage 1 done →", out_dir)

# =========================
# STAGE 2 — Tension Rise (+ light triplet, distant MBV drone)
# =========================
def render_stage2():
    pattern = [A2,A2,A2,A2, G2,G2,A2,A2, F2,F2,F2,F2, E2,E2,E2,E2]
    L = int(SPB * 4 * BARS)
    beats_total = int(L / SPB)
    out_dir = "stems_stage2"; os.makedirs(out_dir, exist_ok=True)

    # Stems
    st_kick = np.zeros((L, 2))
    st_snare = np.zeros((L, 2))
    st_hats = np.zeros((L, 2))
    st_bass = np.zeros((L, 2))
    st_pad  = np.zeros((L, 2))
    st_lead = np.zeros((L, 2))
    st_trip = np.zeros((L, 2))  # NEW
    st_mbv  = np.zeros((L, 2))  # NEW distant drone
    st_shim = np.zeros((L, 2))

    # Kick every beat
    for b in range(beats_total):
        n0 = int(b * SPB); n1 = min(L, n0 + int(0.20*SR)); seg = n1 - n0
        t  = np.arange(seg)/SR; f = 80*np.exp(-t*14) + 40
        s  = np.sin(2*np.pi*np.cumsum(f)/SR)
        s  = one_pole_lp(s * np.linspace(1,0,seg), 3000.0); s = fit1d(s, seg)
        s  = softclip(s, 2.0) * 0.56
        st_kick[n0:n1] += s[:, None]

    # Snare on 3
    for b in range(0, beats_total, 4):
        n0 = int((b + 2) * SPB); n1 = min(L, n0 + int(0.22*SR)); seg = n1 - n0
        nz = np.random.uniform(-1,1,seg) * np.linspace(1,0,seg)
        bd = sine(190, seg) * np.linspace(1,0,seg)
        s  = 0.8 * one_pole_hp(nz, 1800.0) + 0.5 * bd
        s  = fit1d(s, seg)
        s  = softclip(s, 1.4) * 0.5
        st_snare[n0:n1, 0] += s * 0.9
        st_snare[n0:n1, 1] += (np.concatenate([np.zeros(2), s[:-2]]) if seg > 2 else s) * 1.1

    # Hats light
    for i in range(beats_total * 2):
        if np.random.rand() > 0.95: continue
        n0 = int(i * EIGHTH); n1 = min(L, n0 + int(0.05*SR)); seg = n1 - n0
        nz = np.random.uniform(-1,1,seg)
        s  = one_pole_hp(nz, 6000.0) * np.linspace(1,0,seg)
        s  = fit1d(s, seg)
        s  = softclip(s, 1.1) * 0.17
        st_hats[n0:n1] += s[:, None] * np.array([0.7, 1.0])

    # Bass (stereo offset)
    gate = int(0.9 * EIGHTH)
    for step in range(L // EIGHTH):
        n0 = step * EIGHTH; n1 = min(L, n0 + EIGHTH); seg = n1 - n0; g = min(gate, seg)
        f  = pattern[step % len(pattern)]
        o  = 0.6 * saw(f, seg) + 0.4 * sqr(f * 1.005, seg, 0.48)
        e  = env(seg, g, 5, 80, 0.7, 60)
        l  = softclip(o * e, 1.7)
        r  = l if seg <= 24 else np.concatenate([np.zeros(12), l[:-12]])
        l  = fit1d(l, seg); r = fit1d(r, seg)
        st_bass[n0:n1, 0] += l * 0.8
        st_bass[n0:n1, 1] += r * 0.8

    # Pad + lead
    pad = 0.6 * (saw(E4, L) + saw(E4*0.993, L, np.pi/3))
    pad = one_pole_lp(pad, 1200.0) * np.linspace(0,1,L) * 0.25
    st_pad += np.column_stack([pad, np.concatenate([np.zeros(10), pad[:-10]])])

    motif = [
        (1.0, E4, 0.75), (2.0, F2*4, 0.5), (3.0, A4, 0.5),
        (5.0, 392.0, 0.5), (5.5, 440.0, 0.5), (6.0, E4, 0.75),
        (9.0, 293.66, 0.5), (10.0, E4, 0.5),
        (11.0, 392.0, 0.5), (11.5, 440.0, 0.5),
        (13.0, E4, 0.75), (14.0, F2*4, 0.5), (15.0, A4, 0.5),
    ]
    for beat, fq, dur in motif:
        n0 = int(beat * SPB); seg = int(dur * SPB); n1 = min(L, n0 + seg)
        base = 0.55 * sqr(fq, seg, 0.48) + 0.45 * saw(fq * 0.997, seg)
        e    = env(seg, int(0.85 * seg), 8, 120, 0.7, 200)
        s    = one_pole_lp(base, 1400.0) * e
        s    = fit1d(s, seg); s = softclip(s, 1.2) * 0.30
        st_lead[n0:n1, 0] += s
        st_lead[n0:n1, 1] += s * 0.92

    # NEW: light triplet arpeggio
    st_trip += render_triplet_synth(L, scale_notes=(E4, G4, A4), fc=1700.0, gain=0.13, stereo_offset=10)

    # NEW: distant MBV drone (fades in, very low)
    st_mbv += mbv_drone_distant(L, base_freq=E4, fc=1600.0, gain=0.08)

    # Shimmer bus (pad + lead)
    send = st_pad + st_lead
    st_shim += shimmer_light(send, amount=0.15)

    # Mix
    mix = st_kick + st_snare + st_hats + st_bass + st_pad + st_lead + st_trip + st_mbv + st_shim
    mix = softclip(mix, 1.15); mix = norm(mix, 0.97)

    # Write stems + mix
    write_wav(os.path.join(out_dir, "kick.wav"),   st_kick)
    write_wav(os.path.join(out_dir, "snare.wav"),  st_snare)
    write_wav(os.path.join(out_dir, "hats.wav"),   st_hats)
    write_wav(os.path.join(out_dir, "bass.wav"),   st_bass)
    write_wav(os.path.join(out_dir, "pad.wav"),    st_pad)
    write_wav(os.path.join(out_dir, "lead.wav"),   st_lead)
    write_wav(os.path.join(out_dir, "triplet.wav"),st_trip)
    write_wav(os.path.join(out_dir, "mbv_distant.wav"), st_mbv)
    write_wav(os.path.join(out_dir, "shimmer.wav"), st_shim)
    write_wav("carpenter_stage2.wav", mix)
    write_wav(os.path.join(out_dir, "FULL_mix.wav"), mix)
    print("Stage 2 done →", out_dir)

# =========================
# STAGE 3 — Cathedral of Panic (huge MBV wall)
# =========================
def render_stage3():
    pattern = [A2,A2,A2,A2, G2,G2,A2,A2, F2,F2,F2,F2, E2,E2,E2,E2]
    L = int(SPB * 4 * BARS)
    beats = int(L / SPB)
    out_dir = "stems_stage3"; os.makedirs(out_dir, exist_ok=True)

    # Stems
    st_kick  = np.zeros((L, 2))
    st_snare = np.zeros((L, 2))
    st_hats  = np.zeros((L, 2))
    st_bass  = np.zeros((L, 2))
    st_sub   = np.zeros((L, 2))
    st_stabs = np.zeros((L, 2))
    st_pad   = np.zeros((L, 2))
    st_lead  = np.zeros((L, 2))
    st_rise  = np.zeros((L, 2))
    st_mbv   = np.zeros((L, 2))  # NEW: huge MBV wall
    st_shim  = np.zeros((L, 2))

    # Sub drop at start
    n0 = 0; n1 = min(L, int(0.6 * SR)); seg = n1 - n0
    t  = np.arange(seg)/SR; f = 80*np.exp(-t*4) + 30
    sub = np.sin(2*np.pi*np.cumsum(f)/SR) * np.linspace(1,0,seg)
    sub = one_pole_lp(sub, 120.0) * 1.1; sub = fit1d(sub, seg)
    st_sub[n0:n1] += sub[:, None] * 0.6

    # Kick every beat, harder
    for b in range(beats):
        n0 = int(b * SPB); n1 = min(L, n0 + int(0.22*SR)); seg = n1 - n0
        t  = np.arange(seg)/SR; f = 90*np.exp(-t*16) + 45
        s  = np.sin(2*np.pi*np.cumsum(f)/SR)
        s  = one_pole_lp(s*np.linspace(1,0,seg), 3200.0); s = fit1d(s, seg)
        s  = softclip(s, 2.3) * 0.72
        st_kick[n0:n1] += s[:, None]

    # Snares (3 + 3.5 ghost)
    for b in range(0, beats, 4):
        for off, lev in [(2, 0.7), (3.5, 0.45)]:
            n0 = int((b + off) * SPB); n1 = min(L, n0 + int(0.22*SR)); seg = n1 - n0
            nz = np.random.uniform(-1,1,seg) * np.linspace(1,0,seg)
            bd = sine(200, seg) * np.linspace(1,0,seg)
            s  = 0.85 * one_pole_hp(nz, 2000.0) + 0.5 * bd
            s  = fit1d(s, seg); s = softclip(s, 1.6) * lev
            st_snare[n0:n1, 0] += s * 0.9
            st_snare[n0:n1, 1] += (np.concatenate([np.zeros(2), s[:-2]]) if seg > 2 else s) * 1.12

    # Dense 16th hats
    total_16 = L // SIXTEENTH
    for i in range(total_16):
        if np.random.rand() > 0.90: continue
        n0 = i * SIXTEENTH; n1 = min(L, n0 + int(0.05*SR)); seg = n1 - n0
        nz = np.random.uniform(-1,1,seg)
        s  = one_pole_hp(nz, 9000.0) * np.linspace(1,0,seg)
        if i % 4 == 2: s *= 1.25
        s  = fit1d(s, seg); s = softclip(s, 1.2) * 0.22
        st_hats[n0:n1, 0] += s * 0.7
        st_hats[n0:n1, 1] += s

    # Main bass (wider/driven)
    gate = int(0.9 * EIGHTH)
    amp  = np.linspace(0.7, 1.2, L)
    for step in range(L // EIGHTH):
        n0 = step * EIGHTH; n1 = min(L, n0 + EIGHTH); seg = n1 - n0
        f  = pattern[step % len(pattern)]; g = min(gate, seg)
        o  = 0.55*saw(f, seg) + 0.45*sqr(f*1.01, seg, 0.46)
        e  = env(seg, g, 5, 80, 0.7, 60)
        l  = softclip(o * e * amp[n0:n1], 2.2)
        r  = l if seg <= 14 else np.concatenate([np.zeros(14), l[:-14]])
        l  = fit1d(l, seg); r = fit1d(r, seg)
        st_bass[n0:n1, 0] += l * 0.9
        st_bass[n0:n1, 1] += r * 0.9

    # Sub bass bed
    for step in range(L // EIGHTH):
        n0 = step * EIGHTH; n1 = min(L, n0 + EIGHTH); seg = n1 - n0
        f  = pattern[step % len(pattern)] / 2.0
        s  = sine(f, seg) * env(seg, int(0.85*seg), 3, 80, 0.7, 90)
        s  = one_pole_lp(s, 100.0); s = fit1d(s, seg)
        st_sub[n0:n1] += s[:, None] * 0.26

    # Octave stabs
    for b in range(beats):
        n0 = int(b * SPB); n1 = min(L, n0 + int(0.18*SR)); seg = n1 - n0
        s  = saw(A2*2, seg) * env(seg, int(0.2*seg), 5, 120, 0.0, 80)
        s  = one_pole_lp(s, 700.0); s = fit1d(s, seg)
        s  = softclip(s, 1.35) * 0.28
        st_stabs[n0:n1] += s[:, None]

    # Thick pad
    pad = 0.6 * (saw(A4, L) + saw(A4*0.992, L, np.pi/3))
    pad = one_pole_lp(pad, 1400.0) * np.linspace(0,1,L) * 0.32
    st_pad += np.column_stack([pad, np.concatenate([np.zeros(14), pad[:-14]])])

    # Evolving lead
    def rand_lfo(N, depth_cents=35.0, rate_hz=0.27):
        t = np.arange(N)/SR
        s = np.sin(2*np.pi*rate_hz*t)
        n = np.random.randn(N); n = one_pole_lp(n, 8.0)
        blend = (0.6*s + 0.4*n/np.max(np.abs(n))) * (depth_cents/100.0)
        return 2 ** (blend / 12.0)

    motif = [
        (1.0,E4,0.75),(2.0,F2*4,0.5),(3.0,A4,0.5),(5.0,G4,0.5),(5.5,A4,0.5),(6.0,E4,0.75),
        (9.0,D4,0.5),(10.0,E4,0.5),(11.0,G4,0.5),(11.5,A4,0.5),(13.0,E4,0.75),(14.0,F2*4,0.5),(15.0,A4,0.5)
    ]
    for beat, fq, dur in motif:
        n0 = int(beat * SPB); seg = int(dur * SPB); n1 = min(L, n0 + seg)
        r  = rand_lfo(seg)
        base = (0.5*sqr(fq*r, seg, 0.45) +
                0.35*saw(fq*0.997*r, seg) +
                0.15*sqr(fq*1.01*r, seg, 0.42))
        e  = env(seg, int(0.85*seg), 8, 120, 0.7, 240)
        s  = one_pole_lp(base, 1600.0) * e
        s  = fit1d(s, seg); s = softclip(s, 1.4) * 0.45
        if int(beat) % 2 == 0:
            st_lead[n0:n1, 0] += s; st_lead[n0:n1, 1] += s * 0.7
        else:
            st_lead[n0:n1, 0] += s * 0.7; st_lead[n0:n1, 1] += s

    # Riser (last 4 beats)
    rise_len = int(SPB * 4)
    start = max(0, L - rise_len); seg = L - start
    nz = np.random.randn(seg)
    s  = one_pole_hp(nz, 5000.0) * np.linspace(0,1,seg)
    s  = one_pole_lp(s, 4000.0)
    s  = fit1d(s, seg) * 0.18
    st_rise[start:L] += np.column_stack([s*0.9, s])

    # NEW: Huge MBV wall of sound (foreground)
    st_mbv += mbv_wall_huge(L, base_freq=A4, fc=1800.0, gain=0.35)

    # Shimmer bus (pad + lead + MBV wall a bit)
    send = st_pad + st_lead + st_mbv * 0.35
    st_shim += shimmer_long(send, amount=0.50, detune_cents=9.0, hp_hz=1200.0)

    # Mix
    mix = (st_kick + st_snare + st_hats + st_bass + st_sub +
           st_stabs + st_pad + st_lead + st_rise + st_mbv + st_shim)
    mix = softclip(mix, 1.30); mix = norm(mix, 0.97)

    # Write stems + mix
    write_wav(os.path.join(out_dir, "kick.wav"),  st_kick)
    write_wav(os.path.join(out_dir, "snare.wav"), st_snare)
    write_wav(os.path.join(out_dir, "hats.wav"),  st_hats)
    write_wav(os.path.join(out_dir, "bass_main.wav"), st_bass)
    write_wav(os.path.join(out_dir, "bass_sub.wav"),  st_sub)
    write_wav(os.path.join(out_dir, "stabs.wav"),     st_stabs)
    write_wav(os.path.join(out_dir, "pad.wav"),       st_pad)
    write_wav(os.path.join(out_dir, "lead.wav"),      st_lead)
    write_wav(os.path.join(out_dir, "riser.wav"),     st_rise)
    write_wav(os.path.join(out_dir, "mbv_wall.wav"),  st_mbv)
    write_wav(os.path.join(out_dir, "shimmer.wav"),   st_shim)
    write_wav("carpenter_stage3.wav", mix)
    write_wav(os.path.join(out_dir, "FULL_mix.wav"), mix)
    print("Stage 3 done →", out_dir)

# =========================
# Run all three
# =========================
if __name__ == "__main__":
    render_stage1()
    render_stage2()
    render_stage3()
    print("All stages rendered with stems (triplet synth + MBV drone/wall added).")

