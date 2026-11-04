# Generative ambient pad: slow chord progression, detuned sines, soft rain-like noise, long reverb-ish delay.
import numpy as np, wave, sys
SR=44100; DUR=float(sys.argv[1]) if len(sys.argv)>1 else 60
t=np.arange(int(SR*DUR))/SR
def hz(n): return 440*2**((n-69)/12)
chords=[[50,57,62,65,69],[46,53,58,62,65],[43,50,55,58,62],[45,52,57,61,64]]  # Dm, Bb, Gm, A
seg=10.0
L=np.zeros_like(t); R=np.zeros_like(t)
rng=np.random.default_rng(7)
for i,ch in enumerate(chords*int(np.ceil(DUR/seg/4)+1)):
    start=i*seg
    if start>=DUR: break
    s=int(start*SR); e=min(len(t),int((start+seg+6)*SR))
    tt=t[s:e]-start
    env=np.clip(tt/4,0,1)*np.clip((seg+6-tt)/6,0,1)
    env=env**2
    for n in ch:
        f=hz(n)
        for d,pan in ((-0.12,0.3),(0.12,0.7)):
            ff=f*2**(d/12/10)
            v=np.sin(2*np.pi*ff*tt+rng.uniform(0,6))*0.5+0.15*np.sin(2*np.pi*2*ff*tt)
            v*=env*(0.06 if n>55 else 0.08)
            L[s:e]+=v*(1-pan); R[s:e]+=v*pan
# soft high bell melody
mel=[74,77,76,72,69,74,72,70]
for k,n in enumerate(mel*int(DUR//20+1)):
    st=3+k*5.3+rng.uniform(-0.5,0.5)
    if st>DUR-4: break
    s=int(st*SR); e=min(len(t),s+int(5*SR)); tt=t[s:e]-st
    v=np.sin(2*np.pi*hz(n)*tt)*np.exp(-tt*1.1)*0.05
    p=rng.uniform(0.2,0.8); L[s:e]+=v*(1-p); R[s:e]+=v*p
# rain: filtered noise
noise=rng.standard_normal(len(t))
k=np.ones(8)/8; rain=np.convolve(noise,k,'same')*0.012
L+=rain; R+=np.roll(rain,300)
# simple stereo delay "reverb"
for dl,g in ((0.37,0.35),(0.53,0.25),(0.89,0.18),(1.31,0.12)):
    d=int(dl*SR); L[d:]+=R[:-d]*g; R[d:]+=L[:-d]*g
fade=np.clip(t/3,0,1)*np.clip((DUR-t)/4,0,1)
L*=fade; R*=fade
m=max(abs(L).max(),abs(R).max()); L/=m/0.7; R/=m/0.7
out=(np.stack([L,R],1)*32767).astype('<i2')
with wave.open('music.wav','wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(out.tobytes())
