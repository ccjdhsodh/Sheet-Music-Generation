"""Procedural compositions with paired acoustic variations and exact key labels."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tinysoundfont import Synth

SR=16000
SECONDS=12

def composition(seed):
    rng=np.random.default_rng(seed);notes=[];strings=[]
    beat=float(rng.uniform(.36,.8));scale=np.array([0,2,3,5,7,8,10] if rng.random()<.5 else [0,2,4,5,7,9,11])+rng.integers(0,12)
    def add(t,p,d,v,tag):
        if t<SECONDS-1:notes.append(dict(start=float(max(0,t)),end=float(min(SECONDS-.4,t+d)),pitch=int(p),velocity=int(v),tag=tag))
    for t in np.arange(.3,SECONDS-1,beat*2):
        degree=int(rng.choice([0,3,4,5]));bass=36+scale[degree]
        add(t,bass,beat*rng.uniform(1,2.8),rng.integers(28,78),'bass')
        for j,d in enumerate([0,2,4,2]):
            add(t+j*beat/2,48+scale[(degree+d)%7]+12*((degree+d)//7),beat*rng.uniform(.3,1.1),rng.integers(32,78),'accompaniment')
        melody=60+int(rng.choice(scale))
        for j in range(4):
            u=t+j*beat/2;melody=int(np.clip(melody+rng.choice([-2,-1,0,0,1,2]),60,88))
            add(u,melody,beat*rng.uniform(.3,.9),rng.integers(52,108),'melody')
            if rng.random()<.25:
                add(u-.08,melody+int(rng.choice([-2,-1,1,2])),.065,rng.integers(35,80),'grace')
            if rng.random()<.2:
                add(u+.12,melody,.12,rng.integers(30,95),'repeat')
        if rng.random()<.3:
            for interval in [0,4,7]:add(t,60+scale[degree]+interval,beat*.7,rng.integers(45,85),'chord')
    for t in np.arange(.2,SECONDS-1,beat*1.5):
        strings.append(dict(start=float(t),end=float(min(SECONDS-.2,t+beat*rng.uniform(1,3))),pitch=int(60+rng.choice(scale)+rng.choice([0,12])),velocity=int(rng.integers(40,100))))
    clean=[]
    for p in {n['pitch'] for n in notes}:
        seq=sorted([n for n in notes if n['pitch']==p],key=lambda n:n['start'])
        for a,b in zip(seq,seq[1:]):a['end']=min(a['end'],b['start']-.004)
        clean.extend(n for n in seq if n['end']-n['start']>.015)
    return sorted(clean,key=lambda n:(n['start'],n['pitch'])),strings,beat

def render(notes,font,program,pedal,strings=False):
    s=Synth(samplerate=SR,gain=-12);sid=s.sfload(Path(font).read_bytes());s.program_select(0,sid,0,program)
    events=[]
    for n in notes:events.extend([(round(n['start']*SR),1,n['pitch'],n['velocity']),(round(n['end']*SR),0,n['pitch'],0)])
    for t,value in pedal:events.append((round(t*SR),2,value,0))
    if strings:
        for t in np.arange(0,SECONDS,.04):events.append((round(t*SR),3,round(8192+180*np.sin(t*2*np.pi*5.3)),0))
    x=np.zeros((SECONDS*SR,2),np.float32);pos=0
    for frame,kind,p,v in sorted(events):
        frame=min(frame,len(x))
        if frame>pos:x[pos:frame]=np.frombuffer(s.generate_simple(frame-pos),np.float32).reshape(-1,2);pos=frame
        if kind==0:s.noteoff(0,p)
        elif kind==1:s.noteon(0,p,v)
        elif kind==2:s.control_change(0,64,p)
        else:s.pitchbend(0,p)
    if pos<len(x):x[pos:]=np.frombuffer(s.generate_simple(len(x)-pos),np.float32).reshape(-1,2)
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--fonts',type=Path,nargs='+',required=True)
    ap.add_argument('--compositions',type=int,default=256);ap.add_argument('--variants',type=int,default=2);ap.add_argument('--seed',type=int,default=20261016)
    a=ap.parse_args()
    if a.compositions<16:raise ValueError('At least 16 compositions for nonempty splits')
    dest=a.root/'synthetic';dest.mkdir(parents=True,exist_ok=True)
    font_hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in a.fonts]
    # Deterministic split by composition index; all acoustic variants stay together.
    rows=[]
    for i in range(a.compositions):
        split='train' if i<int(a.compositions*.75) else 'validation' if i<int(a.compositions*.875) else 'test'
        seed=a.seed+i;notes,strings,beat=composition(seed)
        for v in range(a.variants):
            clip=f'c{i:05}_v{v:02}';rng=np.random.default_rng(seed*100+v)
            font_index=(i+v)%len(a.fonts);font=a.fonts[font_index]
            row=dict(id=clip,composition_id=f'procedural-{seed}',split=split,font_sha256=font_hashes[font_index])
            rows.append(row)
            meta=dest/f'{clip}.json'
            if meta.exists():
                old=json.loads(meta.read_text())
                if old.get('seed')!=seed or old.get('font_sha256')!=row['font_sha256'] or old.get('split')!=split:raise ValueError('Existing dataset configuration mismatch; use a new root')
                continue
            pedal=[]
            if rng.random()<.6:
                for t in np.arange(.2,SECONDS-1,2*beat):pedal.extend([(float(t),127),(float(min(SECONDS-.1,t+beat*1.7)),0)])
            piano=render(notes,font,int(rng.choice([0,1,2])),pedal)
            violin=render(strings,font,int(rng.choice([40,41,42,48])),[],True)
            ratio=float(rng.uniform(0,1.5));mix=piano+ratio*violin
            rt=float(rng.uniform(.08,.6));ir=np.zeros((int(SR*rt)+1,2),np.float32);ir[0]=1
            for _ in range(30):
                t=int(rng.integers(1,len(ir)));ir[t]+=rng.uniform(-.06,.06,2)*np.exp(-t/(SR*rt/3))
            for ch in range(2):mix[:,ch]=fftconvolve(mix[:,ch],ir[:,ch])[:len(mix)]
            mix+=rng.normal(0,float(rng.uniform(0,.0001)),mix.shape)
            gain=float(rng.uniform(.4,.85))/max(abs(mix).max(),1e-8)
            sf.write(dest/f'{clip}.wav',mix*gain,SR,subtype='PCM_16')
            meta.write_text(json.dumps({**row,'schema':3,'seed':seed,'notes':notes,'pedal':pedal,'string_ratio':ratio,'reverb_seconds':rt,'seconds':SECONDS}),encoding='utf-8')
        if i%16==0:print(f'generated {i+1}/{a.compositions} compositions',flush=True)
    (a.root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    print(json.dumps({'clips':len(rows),'hours':len(rows)*SECONDS/3600,'fonts':len(set(font_hashes))}),flush=True)

if __name__=='__main__':main()
