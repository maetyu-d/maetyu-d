import numpy as np, wave
SR=44100; BPM=96; BARS=16; SPB=SR*60/BPM; eighth=int(SPB/2); sixteenth=int(SPB/4)
def softclip(x,d=1): return np.tanh(x*d)
def norm(x,p=0.98): m=np.max(np.abs(x)); return x*(p/m) if m>0 else x
def lpf_ma(x,t):
    if t<=1: return x
    k=np.ones(t)/t; return np.convolve(x,k,mode="same")
def hpf_ma(x,t): return x-lpf_ma(x,t)
def env(L,g,a=5,d=120,s=0.6,r=200):
    A=int(a*SR/1000);D=int(d*SR/1000);R=int(r*SR/1000);e=np.zeros(L);on=min(L,g)
    e[:min(A,on)]=np.linspace(0,1,min(A,on),endpoint=False); i=A; dl=max(0,min(D,on-i))
    if dl>0: e[i:i+dl]=np.linspace(1,s,dl,endpoint=False); i+=dl
    if on>i: e[i:on]=s
    if L>on and R>0: e[on:on+R]=np.linspace(e[on-1] if on>0 else 0,0,min(R,L-on))
    return e
def saw(f,N,ph=0): t=np.arange(N)/SR; return 2*((f*t+ph/(2*np.pi))%1-0.5)
def sqr(f,N,d=0.5,ph=0): t=np.arange(N)/SR; return np.where(np.sin(2*np.pi*f*t+ph)>=np.cos(np.pi*(1-d)),1,-1)
def sine(f,N,ph=0): t=np.arange(N)/SR; return np.sin(2*np.pi*f*t+ph)
def mtap(x,del_ms,gains,mix=.7,pre=30):
    if x.ndim==1: x=np.column_stack([x,x]); N=x.shape[0]
    out=x*(1-mix); pre=int(pre*SR/1000); acc=np.zeros_like(x)
    for dms,g in zip(del_ms,gains):
        d=pre+int(dms*SR/1000)
        if d>=N: continue
        pad=np.zeros((d,2)); acc+=g*np.vstack([pad,x[:-d]])
    return out+mix*acc
def pitch_up(sig,semi=12.0):
    if sig.ndim==1: sig=np.column_stack([sig,sig]); N=sig.shape[0]
    r=2**(semi/12.0); idx=np.linspace(0,N-1,num=max(1,int(N/r))); fast=np.zeros((idx.size,2)); t=np.arange(N)
    for ch in range(2): fast[:,ch]=np.interp(idx,t,sig[:,ch])
    ib=np.linspace(0,fast.shape[0]-1,num=N); up=np.zeros((N,2)); tf=np.arange(fast.shape[0])
    for ch in range(2): up[:,ch]=np.interp(ib,tf,fast[:,ch])
    return up
def shimmer(x,amt=.38,detune_cents=6.0):
    taps=max(3,int(SR/1200)); hp=np.column_stack([hpf_ma(x[:,0],taps),hpf_ma(x[:,1],taps)])
    up1=pitch_up(hp,12.0); up2=pitch_up(hp,12.0+detune_cents/100.0*12.0); up3=pitch_up(hp,12.0-detune_cents/100.0*12.0)
    up=(0.5*up1+0.3*up2+0.2*up3)
    d=[90,140,220,350,550,900,1300,1800,2400,3000]; g=[.34,.30,.26,.22,.18,.15,.12,.10,.08,.06]
    return mtap(up,d,g,mix=.7,pre=30)*amt
A2,G2,F2,E2,A3,E4,G4,A4,D4=110.0,98.0,87.31,82.41,220.0,329.63,392.0,440.0,293.66
pat=[A2,A2,A2,A2,G2,G2,A2,A2,F2,F2,F2,F2,E2,E2,E2,E2]; gate=int(0.9*eighth)
L=int(SPB*4*BARS); mix=np.zeros((L,2))
# drums
beats=int(L/SPB); kicks=[b for b in range(beats)]; sna=[b+2 for b in range(0,beats,4)]+[b+3.5 for b in range(0,beats,8)]
for b in kicks:
    n0=int(b*SPB); n1=min(L,n0+int(0.2*SR)); N=n1-n0; t=np.arange(N)/SR; f=80*np.exp(-t*14)+40
    s=np.sin(2*np.pi*np.cumsum(f)/SR); s=lpf_ma(s*np.linspace(1,0,N),max(3,int(SR/3000))); s=softclip(s,2.1)*0.68
    mix[n0:n1,0]+=s; mix[n0:n1,1]+=s
for b in sna:
    n0=int(b*SPB); n1=min(L,n0+int(0.22*SR)); N=n1-n0
    nz=np.random.uniform(-1,1,N)*np.linspace(1,0,N); bd=sine(190,N)*np.linspace(1,0,N)
    s=0.8*hpf_ma(nz,max(3,int(SR/1800)))+0.5*bd; s=softclip(s,1.5)*0.66
    mix[n0:n1,0]+=s*0.9; mix[n0:n1,1]+=np.concatenate([np.zeros(2),s[:-2]]) if N>2 else s
# hats 16ths
for i in range(L//sixteenth):
    if np.random.rand()>0.92: continue
    n0=i*sixteenth; n1=min(L,n0+int(0.045*SR)); N=n1-n0; nz=np.random.uniform(-1,1,N)
    s=hpf_ma(nz,max(3,int(SR/8000)))*np.linspace(1,0,N); s=softclip(s,1.15)*0.22
    if i%4==2: s*=1.2
    mix[n0:n1,0]+=s*0.7; mix[n0:n1,1]+=s
# bass main
amp=np.linspace(0.65,1.12,L)
for s in range(L//eighth):
    n0=s*eighth; n1=min(L,n0+eighth); seg=n1-n0; f=pat[s%len(pat)]; g=min(gate,seg)
    o=0.6*saw(f,seg)+0.4*sqr(f*1.005,seg,0.48); e=env(seg,g,5,80,.7,60); l=softclip(o*e*amp[n0:n1],2.0)
    r=l if seg<=12 else np.concatenate([np.zeros(12),l[:-12]])
    mix[n0:n1,0]+=l*0.85; mix[n0:n1,1]+=r*0.85
# sub + stabs
for s in range(L//eighth):
    n0=s*eighth; n1=min(L,n0+eighth); seg=n1-n0; f=pat[s%len(pat)]/2.0
    ss=sine(f,seg)*env(seg,int(0.85*seg),3,80,.7,90); ss=lpf_ma(ss,18); mix[n0:n1,0]+=ss*0.24; mix[n0:n1,1]+=ss*0.24
for b in range(int(L/SPB)):
    n0=int(b*SPB); n1=min(L,n0+int(0.18*SR)); N=n1-n0; s=saw(A2*2,N)*np.linspace(1,0,N); s=lpf_ma(s,18); s=softclip(s,1.3)*0.26
    mix[n0:n1,0]+=s; mix[n0:n1,1]+=s
# pad
p=lpf_ma(0.6*(saw(A4,L)+saw(A4*0.992,L,np.pi/3)),22)*np.linspace(0,1,L)*0.30
PAD=np.column_stack([p,np.concatenate([np.zeros(10),p[:-10]])]); mix+=PAD
# evolved lead
lead=np.zeros((L,2)); motif=[(1.0,E4,0.75),(2.0,87.31*4,0.5),(3.0,440.0,0.5),(5.0,392.0,0.5),(5.5,440.0,0.5),(6.0,329.63,0.75),
                              (9.0,293.66,0.5),(10.0,329.63,0.5),(11.0,392.0,0.5),(11.5,440.0,0.5),(13.0,329.63,0.75),(14.0,87.31*4,0.5),(15.0,440.0,0.5)]
def randlfo(N,depth_cents=30.0,rate=0.25):
    t=np.arange(N)/SR; s=np.sin(2*np.pi*rate*t); n=np.random.randn(N); n=lpf_ma(n,800)
    b=(0.6*s+0.4*n/np.max(np.abs(n)))*(depth_cents/100.0); return 2**((b/12.0))
for beat,fq,dur in motif:
    n0=int(beat*SPB); seg=int(dur*SPB); n1=min(L,n0+seg); N=n1-n0; r=randlfo(N)
    b=0.5*sqr(fq*r,N,0.45)+0.35*saw(fq*0.997*r,N)+0.15*sqr(fq*1.01*r,N,0.42); e=env(N,int(0.85*seg),8,120,0.7,240); s=lpf_ma(b,14)*e; s=softclip(s,1.35)*0.42
    if int(beat)%2==0: lead[n0:n1,0]+=s; lead[n0:n1,1]+=s*0.7
    else: lead[n0:n1,0]+=s*0.7; lead[n0:n1,1]+=s
mix+=lead
# shimmer 3s
send=PAD*0.7+lead*1.0; mix+=shimmer(send,.38,6.0)
mix=softclip(mix,1.25); mix=norm(mix,0.97)
with wave.open("carpenter_stage3.wav","wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(np.int16(np.clip(mix,-1,1)*32767).tobytes())
print("wrote carpenter_stage3.wav")
