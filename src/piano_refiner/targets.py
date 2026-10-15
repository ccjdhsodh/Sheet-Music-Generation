import numpy as np
from scipy.optimize import linear_sum_assignment


def match_targets(notes, truth, tolerance=.07):
    """Maximum-cardinality, minimum-onset-error one-to-one matching per pitch.

    No onset merging here: distinct grace/repeated notes retain distinct labels.
    """
    n=len(notes)
    action=np.zeros(n,np.int64);velocity=np.zeros(n,np.float32);duration=np.zeros(n,np.float32)
    matched=np.full(n,-1,np.int64)
    for pitch in range(21,109):
        ii=[i for i,x in enumerate(notes) if x['midi_note']==pitch]
        jj=[j for j,x in enumerate(truth) if x['pitch']==pitch]
        if not ii or not jj:continue
        delta=np.array([[abs(notes[i]['onset_time']-truth[j]['start']) for j in jj] for i in ii])
        cost=np.where(delta<=tolerance,delta,10000.)
        ri,cj=linear_sum_assignment(cost)
        for a,b in zip(ri,cj):
            if delta[a,b]>tolerance:continue
            i,j=ii[a],jj[b];g=truth[j]
            prior=[p for p in truth if p['pitch']==pitch and 0<g['start']-p['start']<=.6]
            action[i]=2 if prior else 1
            velocity[i]=g['velocity']/127
            duration[i]=np.log1p(g['end']-g['start'])
            matched[i]=j
    return dict(action=action,velocity=velocity,duration=duration,matched=matched)


def positions(notes):
    """Continuous timing, not event-index ordering (chords may share a time)."""
    t=np.array([n['onset_time'] for n in notes],np.float32)
    p=np.array([n['midi_note'] for n in notes],np.float32)
    phase=t[:,None]*np.array([.5,1,2,4],np.float32)[None,:]
    return np.column_stack([np.sin(phase),np.cos(phase),t/16,p/108]).astype(np.float32)
