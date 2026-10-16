"""Controlled real-data adaptation of the frozen-baseline residual network.

No test data is loaded by this trainer. Selection rematches selected events at
50 ms on the real validation split, including unproposed reference notes.
"""
from pathlib import Path
import argparse,hashlib,json,random,time
import numpy as np
import torch
from torch.nn import functional as F
from .train_acoustic import AcousticDataset,pack,forward
from .acoustic_model import AcousticResidualRefiner
from .stratified_metrics import score_notes,aggregate


def retention_weights(notes,truth,matched,action):
    weights=np.ones(len(notes),np.float32)
    for i,j in enumerate(matched):
        if j<0:continue
        g=truth[int(j)]
        weights[i]+= .5*(g['pitch']<=48)+.5*(g['velocity']<=45)
        weights[i]+= .5*(g.get('original_duration',g['end']-g['start'])<=.15)
        weights[i]+=1.*(action[i]==2)
    return np.minimum(weights,3.)


class TrainingDataset(AcousticDataset):
    def __init__(self,root):
        super().__init__(root,root/'manifest.jsonl','train')
    def __getitem__(self,i):
        d=super().__getitem__(i);rid=self.rows[i]['id']
        notes=json.loads((self.root/'features'/f'{rid}.json').read_text())
        truth=json.loads((self.root/'synthetic'/f'{rid}.json').read_text())['notes']
        d['retention_weight']=retention_weights(notes,truth,d['matched'],d['action'])
        return d


def weighted_pack(items):
    b=pack(items);b['retention_weight']=torch.ones_like(b['action'],dtype=torch.float32)
    for i,item in enumerate(items):b['retention_weight'][i,:len(item['action'])]=torch.from_numpy(item['retention_weight'])
    return b


def acceptable(candidate,baseline):
    # Floors apply to each interference condition, not only pooled totals.
    for condition,ref in baseline.items():
        cur=candidate[condition]
        if cur['precision']<ref['precision']-.01:return False
        for group in ['all','bass','weak','short','repeated']:
            a=cur['groups'][group]['recall'];b=ref['groups'][group]['recall']
            if b is not None and (a is None or a<b-.005):return False
    return True


@torch.no_grad()
def validation_predictions(model,dataset,device):
    model.eval();rows=[]
    for i,r in enumerate(dataset.rows):
        b=pack([dataset[i]]);b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
        score=(1-forward(model,b).softmax(-1)[0,:,0]).cpu().numpy()
        base=(1-b['base_logits'].softmax(-1)[0,:,0]).cpu().numpy()
        notes=json.loads((dataset.root/'features'/f"{r['id']}.json").read_text())
        gt=json.loads((dataset.root/'synthetic'/f"{r['id']}.json").read_text())
        rows.append(dict(condition=r['condition'],score=score,base=base,notes=notes,truth=gt['notes']))
    return rows


def evaluate(rows,threshold,key='score'):
    result={}
    for condition in ['clean','strings10','strings0']:
        result[condition]=aggregate([score_notes([n for i,n in enumerate(r['notes']) if r[key][i]>=threshold],r['truth'])
            for r in rows if r['condition']==condition])
    return result


def objective(report):return float(np.mean([m['f1'] for m in report.values()]))


def main():
    ap=argparse.ArgumentParser()
    for name in ['real-root','synthetic-root','baseline','run']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--arm',choices=['data_only','retention'],required=True)
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--device',default='cpu')
    a=ap.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'best.pt').exists():raise ValueError('Use a fresh run directory')
    seed=20261017;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    digest=hashlib.sha256(a.baseline.read_bytes()).hexdigest()
    for root in [a.real_root,a.synthetic_root]:
        if json.loads((root/'acoustic-v3/metadata.json').read_text())['baseline_sha256']!=digest:raise ValueError('Baseline cache mismatch')
    baseline=torch.load(a.baseline,map_location='cpu',weights_only=False)
    train_sets=[TrainingDataset(root) for root in [a.real_root,a.synthetic_root]]
    ds=torch.utils.data.ConcatDataset(train_sets)
    sampling=torch.cat([torch.full((len(d),),.5/len(d)) for d in train_sets])
    sampler=torch.utils.data.WeightedRandomSampler(sampling,256,replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=torch.utils.data.DataLoader(ds,batch_size=4,sampler=sampler,collate_fn=weighted_pack)
    val=AcousticDataset(a.real_root,a.real_root/'manifest.jsonl','validation')
    model=AcousticResidualRefiner().to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    initial=validation_predictions(model,val,a.device);base=evaluate(initial,baseline['threshold'],'base')
    best=objective(base);best_epoch=-1;best_report=base
    def save(epoch,threshold):
        torch.save(dict(model=model.state_dict(),schema='real-adaptation-v4',baseline_sha256=digest,
            threshold=float(threshold),epoch=epoch,arm=a.arm),a.run/'best.pt')
    save(-1,baseline['threshold'])
    protocol=dict(seed=seed,arm=a.arm,real_train_clips=len(train_sets[0]),synthetic_train_clips=len(train_sets[1]),
        validation_clips=len(val),sampled_clips_per_epoch=256,domain_sampling='50/50 real/synthetic',
        architecture='Unchanged acoustic residual, zero initialized correction to v2',
        training_matching_ms=70,validation_matching_ms=50,test_loaded=False,
        selection='Mean condition F1; each condition precision within 1pp and each subgroup recall within 0.5pp of v2',
        baseline_sha256=digest,validation_baseline=base)
    (a.run/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for epoch in range(a.epochs):
        start=time.time();model.train();losses=[]
        for b in loader:
            b={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in b.items()};mask=b['valid']
            logits=forward(model,b);ce=F.cross_entropy(logits[mask],b['action'][mask],reduction='none')
            weights=b['retention_weight'][mask] if a.arm=='retention' else torch.ones_like(ce)
            loss=(ce*weights).sum()/weights.sum()+.01*(logits[mask]-b['base_logits'][mask]).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss))
        rows=validation_predictions(model,val,a.device)
        choices=[]
        for threshold in sorted(set(np.linspace(.1,.9,17).tolist()+[float(baseline['threshold'])])):
            report=evaluate(rows,threshold)
            if acceptable(report,base):choices.append((objective(report),threshold,report))
        current=max(choices,key=lambda x:x[0]) if choices else None
        if current and current[0]>best:
            best,threshold,best_report=current;best_epoch=epoch;save(epoch,threshold)
        line=dict(epoch=epoch,seconds=time.time()-start,loss=float(np.mean(losses)),
            eligible=len(choices),validation_mean_f1=current[0] if current else None,best_epoch=best_epoch,best_mean_f1=best)
        with (a.run/'history.jsonl').open('a') as f:f.write(json.dumps(line)+'\n')
        print(json.dumps(line),flush=True)
    report=dict(**protocol,selected_epoch=best_epoch,validation_selected=best_report,
        validation_mean_f1=best,fallback_to_v2=best_epoch==-1,training_complete=True,
        limitation='Only four validation compositions; no new test evaluation or listening promotion in this trainer.')
    (a.run/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'complete':True,'run':str(a.run),'selected_epoch':best_epoch}),flush=True)


if __name__=='__main__':main()
