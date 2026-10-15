import numpy as np
from scipy.signal import find_peaks

def build_features(notes,views):
    """Identical feature code for synthetic and real recordings; no true key/tempo inputs."""
    notes=sorted(notes,key=lambda n:(n['onset_time'],n['midi_note']))
    length=min(len(d['frame_output']) for d in views)
    onset=views[0]['reg_onset_output'][:length]
    env=onset.max(axis=1);env-=env.mean()
    ac=np.correlate(env,env,mode='full')[length-1:]
    lag=int(30+np.argmax(ac[30:90])) if length>90 else 50
    step=lag/100
    energy=np.zeros(12)
    for p in range(88):energy[(p+21)%12]+=onset[:,p].sum()
    energy/=max(energy.max(),1e-6)
    patches=[];context=[]
    last_by_pitch={}
    for j,n in enumerate(notes):
        t=float(n['onset_time']);p=int(n['midi_note']);k=p-21;i=round(t*100)
        indices=np.clip(np.arange(i-20,i+41),0,length-1)
        channels=[]
        for d in views:
            for key in ['reg_onset_output','reg_offset_output','frame_output','velocity_output']:
                channels.append(d[key][indices,k])
            for interval in [-12,12]:
                channels.append(d['reg_onset_output'][indices,np.clip(k+interval,0,87)])
        patches.append(np.array(channels,np.float32))
        prev=notes[max(0,j-1)]
        gap=min(3,t-last_by_pitch.get(p,t-3));last_by_pitch[p]=t
        active=[x for x in notes[:j] if x['offset_time']>t and x['onset_time']<t]
        consonance=np.mean([int((p-x['midi_note'])%12 in [0,3,4,5,7,8,9]) for x in active]) if active else .5
        context.append([p/108,min(6,n['offset_time']-t)/6,n['velocity']/127,gap/3,
            min(1,t-prev['onset_time']),np.sin(2*np.pi*t/step),np.cos(2*np.pi*t/step),
            energy[p%12],consonance,(p-prev['midi_note'])/48,min(len(active),12)/12,
            min(8,sum(abs(x['onset_time']-t)<.06 for x in notes))/8,*np.roll(energy,-p%12)])
    return notes,np.array(patches,np.float32),np.array(context,np.float32)

def candidates(results, duration):
    merged=[]
    for result in results:
        for n in result:
            if n['onset_time']>=duration-.1:continue
            if any(x['midi_note']==n['midi_note'] and abs(x['onset_time']-n['onset_time'])<.025 for x in merged):continue
            merged.append(dict(n))
    return sorted(merged,key=lambda n:(n['onset_time'],n['midi_note']))

def labels(notes,gt):
    y=np.zeros(len(notes),np.float32);vel=np.zeros(len(notes),np.float32);taken=set()
    pairs=sorted((abs(n['onset_time']-g['start']),i,j) for i,n in enumerate(notes) for j,g in enumerate(gt)
       if n['midi_note']==g['pitch'] and abs(n['onset_time']-g['start'])<.07)
    used=set()
    for delta,i,j in pairs:
        if i not in used and j not in taken:
            y[i]=1;vel[i]=gt[j]['velocity']/127;used.add(i);taken.add(j)
    return y,vel
