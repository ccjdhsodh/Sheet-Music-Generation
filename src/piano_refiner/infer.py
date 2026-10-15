"""Apply a trained refiner to cached candidates, with optional duration head."""
from pathlib import Path
import argparse,json
import numpy as np
import torch,pretty_midi
from .model import DualViewRefiner

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--features',type=Path,required=True);ap.add_argument('--notes',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--use-duration',action='store_true');ap.add_argument('--device',default='cpu')
    a=ap.parse_args();ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    net=DualViewRefiner(**ck['config']).to(a.device);net.load_state_dict(ck['model']);net.eval()
    with np.load(a.features,allow_pickle=False) as z:
        assert int(z['schema'])==3
        x=[torch.from_numpy(z[k])[None].to(a.device) for k in ['patch','context','position']]
    notes=json.loads(a.notes.read_text());assert len(notes)==x[0].shape[1]
    with torch.no_grad():out=net(*x,torch.ones(1,len(notes),device=a.device,dtype=torch.bool))
    score=(1-out['action'].softmax(-1)[0,:,0]).cpu().numpy();keep=[];audit=[]
    for i,n in enumerate(notes):
        n=dict(n);selected=bool(score[i]>=ck['threshold']);audit.append({**n,'score':float(score[i]),'selected':selected})
        if not selected:continue
        if a.use_duration:n['offset_time']=n['onset_time']+max(.02,float(out['duration'][0,i].expm1()))
        keep.append(n)
    for p in range(21,109):
        seq=sorted([n for n in keep if n['midi_note']==p],key=lambda n:n['onset_time'])
        for n,m in zip(seq,seq[1:]):n['offset_time']=min(n['offset_time'],m['onset_time'])
    a.output.parent.mkdir(parents=True,exist_ok=True);midi=pretty_midi.PrettyMIDI();inst=pretty_midi.Instrument(0)
    for n in keep:
        if n['offset_time']>n['onset_time']:inst.notes.append(pretty_midi.Note(n['velocity'],n['midi_note'],n['onset_time'],n['offset_time']))
    midi.instruments.append(inst);midi.write(str(a.output.with_suffix('.mid')));a.output.with_suffix('.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
