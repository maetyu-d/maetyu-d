import numpy as np, wave

SR = 44100
BPM = 96
BARS = 16
SPB = SR * 60 / BPM
eighth = int(SPB / 2)

def softclip(x, d=1): return np.tanh(x * d)
def norm(x, p=0.98): m = np.max(np.abs(x)); return x * (p/m) if m > 0 else x
def lpf_ma(x, t):
    if t <= 1: return x
    k = np.ones(t) / t
    return np.convolve(x, k, mode="same")
def hpf_ma(x, t): return x - lpf_ma(x, t)

def env(L, g, a=5, d=120, s=0.6, r=200):
    A, D, R = int(a*SR/1000), int(d*SR/1000), int(r*SR/1000)
    e = np.zeros(L); on = min(L, g)
    e[:min(A,on)] = np.linspace(0,1,min(A,on),endpoint=False)
    i=A; dl=max(0,min(D,on-i))
    if dl>0: e[i:i+dl] = np.linspace(1,s,dl,endpoint=False); i += dl
    if on>i: e[i:on] = s
    if L>on and R>0: e[on:on+R] = np.linspace(e[on-1] if on>0 else 0,0,min(R,L-on))
    return e

def saw(f,N,ph=0): t=np.arange(N)/SR; return 2*((f*t+ph/(2*np.pi))%1-0.5)
def sqr(f,N,d=0.5,ph=0): t=np.arange(N)/SR; return np.sign(np.sin(2*np.pi*f*t+ph)-(1-2*d))
def sine(f,N,ph=0): t=np.arange(N)/SR; return np.sin(2*np.pi*f*t+ph)

# --- fixed shimmer reverb ---
def mtap(x, del_ms=(90,140,200,260,340), g=(.28,.23,.19,.16,.12), mix=.35, pre=28):
    if x.ndim == 1: 
        x = np.column_stack([x, x])
    N = x.shape[0]  # always defined now
    out = x * (1 - mix)
    pre = int(pre * SR / 1000)
    acc = np.zeros_like(x)
    for dms, gg in zip(del_ms, g):
        d = pre + int(dms * SR / 1000)
        if d >= N: continue
        pad = np.zeros((d,2))
        acc += gg * np.vstack([pad, x[:-d]])
    return out + mix * acc

def pitch_up(sig):
    if sig.ndim == 1: sig = np.column_stack([sig, sig])
    N = sig.shape[0]
    idx = np.linspace(0, N-1, num=max(1, N//2))
    fast = np.zeros((idx.size, 2))
    t = np.arange(N)
    for ch in range(2):
        fast[:, ch] = np.interp(idx, t, sig[:, ch])
    ib = np.linspace(0, fast.shape[0]-1, num=N)
    up = np.zeros((N, 2))
    tf = np.arange(fast.shape[0])
    for ch in range(2):
        up[:, ch] = np.interp(ib, tf, fast[:, ch])
    return up

def shimmer(x, amt=.16):
    hp = np.column_stack([
        hpf_ma(x[:,0], max(3,int(SR/1200))),
        hpf_ma(x[:,1], max(3,int(SR/1200)))
    ])
    up = pitch_up(hp)
    cloud = mtap(up, mix=.6)
    return cloud * amt

# --- instruments ---
A2,G2,F2,E2,E4,A4 = 110,98,87.31,82.41,329.63,440
pat=[A2,A2,A2,A2,G2,G2,A2,A2,F2,F2,F2,F2,E2,E2,E2,E2]
gate=int(0.9*eighth)
L=int(SPB*4*BARS)
audio=np.zeros((L,2))

# kick
for b in range(int(L/SPB)):
    n0=int(b*SPB); n1=min(L,n0+int(0.2*SR)); N=n1-n0
    t=np.arange(N)/SR; f=80*np.exp(-t*14)+40
    s=np.sin(2*np.pi*np.cumsum(f)/SR)
    s=lpf_ma(s*np.linspace(1,0,N),max(3,int(SR/3000)))
    s=softclip(s,2)*0.56
    audio[n0:n1]+=s[:,None]

# bass
for s in range(L//eighth):
    n0=s*eighth; n1=min(L,n0+eighth); seg=n1-n0
    f=pat[s%len(pat)]; g=min(gate,seg)
    o=0.6*saw(f,seg)+0.4*sqr(f*1.005,seg,0.48)
    e=env(seg,g,5,80,.7,60)
    l=o*e; r=l if seg<=24 else np.concatenate([np.zeros(12),l[:-12]])
    l=softclip(l,1.7); r=softclip(r,1.7)
    audio[n0:n1,0]+=l*0.8; audio[n0:n1,1]+=r*0.8

# snare
for b in range(0,int(L/SPB),4):
    n0=int((b+2)*SPB); n1=min(L,n0+int(0.22*SR)); N=n1-n0
    nz=np.random.uniform(-1,1,N)*np.linspace(1,0,N)
    bd=sine(190,N)*np.linspace(1,0,N)
    s=0.8*hpf_ma(nz,2000)+0.5*bd
    s=softclip(s,1.4)*0.5
    audio[n0:n1,0]+=s*0.9; audio[n0:n1,1]+=s

# hats
for i in range(int(L/SPB)*2):
    if np.random.rand()>0.95: continue
    n0=int(i*eighth); n1=min(L,n0+int(0.05*SR)); N=n1-n0
    nz=np.random.uniform(-1,1,N)
    s=hpf_ma(nz,6000)*np.linspace(1,0,N)
    s=softclip(s,1.1)*0.17
    audio[n0:n1]+=s[:,None]

# pad
pad = 0.6*(saw(E4,L)+saw(E4*0.993,L,np.pi/3))
pad = lpf_ma(pad,24)*np.linspace(0,1,L)*0.25
padR = np.concatenate([np.zeros(10),pad[:-10]])
P = np.column_stack([pad,padR])
audio += P

# lead motif
motif=[(1.0,E4,0.75),(2.0,87.31*4,0.5),(3.0,A4,0.5)]
lead=np.zeros((L,2))
for beat,fq,dur in motif:
    n0=int(beat*SPB); seg=int(dur*SPB); n1=min(L,n0+seg); N=n1-n0
    base=0.55*sqr(fq,N,0.48)+0.45*saw(fq*0.997,N)
    e=env(N,int(0.85*seg),8,120,0.7,200)
    s=lpf_ma(base,14)*e; s=softclip(s,1.2)*0.30
    lead[n0:n1,0]+=s; lead[n0:n1,1]+=s
audio+=lead

# shimmer
audio += shimmer(P*0.6+lead*0.8,0.15)

audio = softclip(audio,1.15)
audio = norm(audio,0.97)
with wave.open("carpenter_stage2.wav","wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(np.int16(np.clip(audio,-1,1)*32767).tobytes())
print("wrote carpenter_stage2.wav")
