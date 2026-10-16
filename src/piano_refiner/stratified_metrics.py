"""Onset metrics with explicit subgroup denominators and piece-level pairing."""
import numpy as np
from .targets import match_targets

GROUPS=['all','bass','weak','short','repeated','dense']

def score_notes(prediction,truth,window=(.3,11.7),tolerance=.05):
    gt=[n for n in truth if window[0]<=n['start']<window[1]]
    pred=[n for n in prediction if window[0]<=n['onset_time']<window[1]]
    labels=match_targets(pred,gt,tolerance);matched=set(int(x) for x in labels['matched'] if x>=0)
    tp=len(matched);fp=len(pred)-tp;fn=len(gt)-tp
    groups={}
    for group in GROUPS:
        indices=[]
        for i,n in enumerate(gt):
            qualifies=(group=='all' or group=='bass' and n['pitch']<=48 or group=='weak' and n['velocity']<=45
              or group=='short' and n.get('original_duration',n['end']-n['start'])<=.15
              or group=='repeated' and n.get('is_repeat',False)
              or group=='dense' and any(.015<abs(n['start']-m['start'])<=.12 for m in gt))
            if qualifies:indices.append(i)
        groups[group]={'reference':len(indices),'matched':sum(i in matched for i in indices)}
    nearby=octave=0
    for i,n in enumerate(pred):
        if labels['matched'][i]>=0:continue
        related=[g for g in gt if abs(g['start']-n['onset_time'])<=.07]
        nearby+=any(abs(n['midi_note']-g['pitch']) in [1,2] for g in related)
        octave+=any(n['midi_note']-g['pitch'] in [12,19,24] for g in related)
    return dict(tp=tp,fp=fp,fn=fn,groups=groups,nearby_pitch_fp=int(nearby),harmonic_related_fp=int(octave))

def aggregate(rows):
    out={k:sum(r[k] for r in rows) for k in ['tp','fp','fn','nearby_pitch_fp','harmonic_related_fp']}
    tp,fp,fn=out['tp'],out['fp'],out['fn']
    out.update(precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))
    out['groups']={}
    for group in GROUPS:
        total=sum(r['groups'][group]['reference'] for r in rows);matched=sum(r['groups'][group]['matched'] for r in rows)
        out['groups'][group]=dict(reference=total,matched=matched,recall=matched/total if total else None)
    return out

def paired_interval(pieces,seed=20261016):
    """Bootstrap whole compositions, preserving variants/crops within a piece."""
    rng=np.random.default_rng(seed);keys=sorted(pieces);deltas=[]
    if len(keys)<2:return None
    for _ in range(1000):
        sample=[pieces[keys[i]] for i in rng.integers(0,len(keys),len(keys))]
        a=aggregate([r for p in sample for r in p['v2']]);b=aggregate([r for p in sample for r in p['v3']])
        deltas.append(b['f1']-a['f1'])
    return dict(compositions=len(keys),f1_delta_95_percentile=np.quantile(deltas,[.025,.975]).tolist(),resamples=1000)
