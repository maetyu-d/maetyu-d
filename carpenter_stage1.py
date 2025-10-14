import numpy as np, wave
SR=44100; BPM=96; BARS=16; SPB=SR*60/BPM; eighth=int(SPB/2)
def softclip(x,d=1): return np.tanh(x*d)
def norm(x,p=0.98): m=np.max(np.abs(x)); return x*(p/m) if m>0 else x
def lpf1(x,fc):
    c=np.exp(-2*np.pi*fc/SR); a=1-c; y=0; o=np.zeros_like(x)
    for i in range(len(x)): y=a*x[i]+c*y; o[i]=y
    return o
def saw(f,N,ph=0): t=np.arange(N)/SR; return 2*((f*t+ph/(2*np.pi))%1-0.5)
def sqr(f,N,d=0.5,ph=0): t=np.arange(N)/SR; return np.sign(np.sin(2*np.pi*f*t+ph)-(1-2*d))
def env(L,g,a=5,d=120,s=0.6,r=200):
    A=int(a*SR/1000);D=int(d*SR/1000);R=int(r*SR/1000);e=np.zeros(L);on=min(L,g)
    e[:min(A,on)]=np.linspace(0,1,min(A,on),endpoint=False); i=A; dl=max(0,min(D,on-i))
    if dl>0: e[i:i+dl]=np.linspace(1,s,dl,endpoint=False); i+=dl
    if on>i: e[i:on]=s
    if L>on and R>0: e[on:on+R]=np.linspace(e[on-1] if on>0 else 0,0,min(R,L-on))
    return e
A2,G2,F2,E2,A3=110.0,98.0,87.31,82.41,220.0
pat=[A2,A2,A2,A2,G2,G2,A2,A2,F2,F2,F2,F2,E2,E2,E2,E2]; gate=int(0.9*eighth)
L=int(SPB*4*BARS); x=np.zeros((L,2))
# bass
for s in range(L//eighth):
    n0=s*eighth; n1=min(L,n0+eighth); seg=n1-n0; f=pat[s%len(pat)]; g=min(gate,seg)
    o=0.6*saw(f,seg)+0.4*sqr(f*1.005,seg,0.48); e=env(seg,g)
    l=o*e; r=l.copy()
    if seg>12: r=np.concatenate([np.zeros(12),l[:-12]])
    x[n0:n1,0]+=l; x[n0:n1,1]+=r
# kick
for b in range(int(L/SPB)):
    n0=int(b*SPB); n1=min(L,n0+int(0.22*SR)); N=n1-n0
    t=np.arange(N)/SR; f=80*np.exp(-t*14)+40; s=np.sin(2*np.pi*np.cumsum(f)/SR)
    s=lpf1(s*np.linspace(1,0,N),3000); s=softclip(s,2)*0.6; x[n0:n1,0]+=s; x[n0:n1,1]+=s
# pad
p=saw(A3,L)+saw(A3*0.995,L,np.pi/3); p=lpf1(0.6*p,900)*np.linspace(0,1,L)*0.23
x[:,0]+=p; x[:,1]+=np.concatenate([np.zeros(8),p[:-8]])
x=softclip(x,1.2); x=norm(x)
with wave.open("carpenter_stage1.wav","wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(np.int16(np.clip(x,-1,1)*32767).tobytes())
print("wrote carpenter_stage1.wav")
