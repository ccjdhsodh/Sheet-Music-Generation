"""Windowed inference from original audio and cached mix/piano AMT posteriors."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,soundfile as sf,torch,pretty_midi
from .acoustic import acoustic_patches
from .acoustic_model import AcousticResidualRefiner
from .model import DualViewRefiner
from .features import candidates,build_features
from .targets import positions

def main():
    ap=argparse.ArgumentParser()
    for key in ['audio','mix','piano','baseline','checkpoint','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--device',default='cpu');a=ap.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if ck['baseline_sha256']!=hashlib.sha256(a.baseline.read_bytes()).hexdigest():raise ValueError('Wrong baseline checkpoint')
    bc=torch.load(a.baseline,map_location=a.device,weights_only=False)
    base=DualViewRefiner(**bc['config']).to(a.device);base.load_state_dict(bc['model']);base.eval()
    net=AcousticResidualRefiner().to(a.device);net.load_state_dict(ck['model']);net.eval()
    x,sr=sf.read(a.audio,always_2d=True);views=[dict(np.load(path)) for path in [a.mix,a.piano]]
    frames=min(round(len(x)/sr*100),*[len(d['frame_output']) for d in views]);seconds=frames/100
    from piano_transcription_inference.utilities import RegressionPostProcessor
    events=[]
    for view in views:
        ns,_=RegressionPostProcessor(100,88,.15,.3,.1,.2).output_dict_to_midi_events({k:v[:frames].copy() for k,v in view.items()});events.append(ns)
    notes=candidates(events,seconds)
    starts=list(range(0,max(1,frames-1200+1),600))
    if starts[-1]!=max(0,frames-1200):starts.append(max(0,frames-1200))
    scores=np.zeros(len(notes));base_scores=np.zeros(len(notes));weights=np.zeros(len(notes))
    for begin in starts:
        end=min(frames,begin+1200);start=begin/100;stop=end/100
        ids=[i for i,n in enumerate(notes) if start<=n['onset_time']<stop]
        if not ids:continue
        local=[{**notes[i],'onset_time':notes[i]['onset_time']-start,'offset_time':min(notes[i]['offset_time'],stop)-start} for i in ids]
        local,patch,context=build_features(local,[{k:v[begin:end] for k,v in d.items()} for d in views])
        raw=acoustic_patches(x[round(start*sr):round(stop*sr)],sr,local)
        tensor=lambda z:torch.from_numpy(z)[None].to(a.device)
        valid=torch.ones(1,len(local),dtype=torch.bool,device=a.device)
        with torch.no_grad():
            ctx=tensor(context);pos=tensor(positions(local));logits=base(tensor(patch),ctx,pos,valid)['action']
            p=(1-net(tensor(raw),logits,ctx,pos,valid).softmax(-1)[0,:,0]).cpu().numpy()
            bp=(1-logits.softmax(-1)[0,:,0]).cpu().numpy()
        for j,i in enumerate(ids):
            relative=local[j]['onset_time'];w=max(.1,min(relative,stop-start-relative,3)/3)
            scores[i]+=w*p[j];base_scores[i]+=w*bp[j];weights[i]+=w
    if len(notes):
        assert np.all(weights>0);scores/=weights;base_scores/=weights
    audit=[];selected=[]
    for i,n in enumerate(notes):
        keep=bool(scores[i]>=ck['threshold'])
        audit.append({**n,'score':float(scores[i]),'baseline_score':float(base_scores[i]),'selected':keep})
        if keep:selected.append({**n,'offset_time':min(n['offset_time'],seconds)})
    for pitch in range(21,109):
        seq=[n for n in selected if n['midi_note']==pitch]
        for p,q in zip(seq,seq[1:]):p['offset_time']=min(p['offset_time'],q['onset_time'])
    midi=pretty_midi.PrettyMIDI();inst=pretty_midi.Instrument(0)
    for n in selected:
        if n['offset_time']>n['onset_time']:inst.notes.append(pretty_midi.Note(int(n['velocity']),int(n['midi_note']),float(n['onset_time']),float(n['offset_time'])))
    midi.instruments.append(inst);a.output.parent.mkdir(parents=True,exist_ok=True);midi.write(str(a.output.with_suffix('.mid')))
    clean=lambda obj:json.dumps(obj,indent=2,default=lambda z:z.item())
    a.output.with_suffix('.json').write_text(clean(selected));a.output.with_name(a.output.stem+'_audit.json').write_text(clean(audit))
    report=dict(candidates=len(notes),selected=len(inst.notes),threshold=ck['threshold'],epoch=ck['epoch'],manual_edits=False,seconds=seconds)
    a.output.with_name(a.output.stem+'_report.json').write_text(clean(report));print(clean(report))
if __name__=='__main__':main()
