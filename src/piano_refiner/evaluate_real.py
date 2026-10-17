from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch
from .data import load_manifest
from .acoustic_model import model_for_checkpoint,AttackEvidenceRefiner
from .stratified_metrics import score_notes,aggregate,paired_interval

def main():
    ap=argparse.ArgumentParser()
    for key in ['root','baseline','checkpoint','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--device',default='cpu');a=ap.parse_args();torch.set_num_threads(2)
    bc=torch.load(a.baseline,map_location='cpu',weights_only=False);ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    digest=hashlib.sha256(a.baseline.read_bytes()).hexdigest()
    if digest!=ck['baseline_sha256']:raise ValueError('Baseline mismatch')
    net=model_for_checkpoint(ck).to(a.device);net.load_state_dict(ck['model']);net.eval()
    manifest=load_manifest(a.root/'manifest.jsonl');entries=[]
    for r in manifest:
        if r['split'] not in ['validation','test']:continue
        with np.load(a.root/'features'/f"{r['id']}.npz") as z:d={k:z[k] for k in ['context','position','patch']}
        with np.load(a.root/'acoustic-v3'/f"{r['id']}.npz") as z:
            if str(z['baseline_sha256'])!=digest:raise ValueError('Stale baseline logits')
            raw=z['raw'].astype(np.float32);logits=z['base_logits']
        notes=json.loads((a.root/'features'/f"{r['id']}.json").read_text())
        gt=json.loads((a.root/'synthetic'/f"{r['id']}.json").read_text())
        tensor=lambda x:torch.from_numpy(x)[None].to(a.device)
        with torch.no_grad():
            extra={'patch':tensor(d['patch'])} if isinstance(net,AttackEvidenceRefiner) else {}
            base=tensor(logits);pred=net(tensor(raw),base,tensor(d['context']),tensor(d['position']),torch.ones(1,len(notes),device=a.device,dtype=torch.bool),**extra)
            scores={'v2':(1-base.softmax(-1)[0,:,0]).cpu().numpy(),'v3':(1-pred.softmax(-1)[0,:,0]).cpu().numpy()}
        choices={'pool':notes,'v2':[n for i,n in enumerate(notes) if scores['v2'][i]>=bc['threshold']],
                 'v3':[n for i,n in enumerate(notes) if scores['v3'][i]>=ck['threshold']]}
        entries.append({**r,'metrics':{name:score_notes(ns,gt['notes'],gt['evaluation_window']) for name,ns in choices.items()}})
    summary={};intervals={}
    for split in ['validation','test']:
        summary[split]={};intervals[split]={}
        for condition in ['clean','strings10','strings0']:
            rows=[r for r in entries if r['split']==split and r['condition']==condition]
            summary[split][condition]={m:aggregate([r['metrics'][m] for r in rows]) for m in ['pool','v2','v3']}
            pieces={}
            for r in rows:
                p=pieces.setdefault(r['composition_id'],{'v2':[],'v3':[]})
                for m in p:p[m].append(r['metrics'][m])
            intervals[split][condition]=paired_interval(pieces)
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(summary=summary,paired_intervals=intervals,thresholds={'v2':bc['threshold'],'v3':ck['threshold']},
        protocol='Fixed checkpoints/thresholds, exact pitch and 50 ms onset, 0.3-11.7s crop interior; subgroup recall includes unproposed notes.',
        limitations=['Small MAESTRO subset, not a comprehensive benchmark.','MAESTRO may overlap frozen AMT pretraining; correction networks have not trained on these recordings.',
         'Short/dense notes are proxies, not score-annotated ornaments.','Added strings are synthetic, not recorded ensemble stems.','Intervals resample compositions, not individual notes.'],
        promotion='No model promotion or retraining in this diagnostic run; listening version 15 remains the reference.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'per_clip.json').write_text(json.dumps(entries,indent=2))
    print(json.dumps({'stage':'evaluation_complete','evaluated_clips':len(entries),'output':str(a.output)}),flush=True)
if __name__=='__main__':main()
