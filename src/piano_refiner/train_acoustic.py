from pathlib import Path
import argparse,json,hashlib,random,time
import numpy as np
import torch
from torch.nn import functional as F
from .data import FeatureDataset,collate
from .acoustic_model import AcousticResidualRefiner

class AcousticDataset(FeatureDataset):
    def __getitem__(self,i):
        d=super().__getitem__(i)
        with np.load(self.root/'acoustic-v3'/f"{self.rows[i]['id']}.npz",allow_pickle=False) as z:
            d['raw']=z['raw'].astype(np.float32);d['base_logits']=z['base_logits']
        if len(d['raw'])!=len(d['action']):raise ValueError('Candidate/audio misalignment')
        return d

def pack(items):
    out=collate(items);b,n=out['valid'].shape
    for key in ['raw','base_logits']:
        out[key]=torch.zeros((b,n)+items[0][key].shape[1:])
        for i,d in enumerate(items):out[key][i,:len(d[key])]=torch.from_numpy(d[key])
    del out['patch']
    return out

def forward(model,b):return model(b['raw'],b['base_logits'],b['context'],b['position'],b['valid'])

@torch.no_grad()
def predict(model,loader,device):
    model.eval();ys=[];scores=[];baselines=[];gt=0
    for b in loader:
        gt+=b['gt_count'];b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()};mask=b['valid']
        ys.extend(b['action'][mask].cpu().tolist())
        scores.extend((1-forward(model,b).softmax(-1)[...,0])[mask].cpu().tolist())
        baselines.extend((1-b['base_logits'].softmax(-1)[...,0])[mask].cpu().tolist())
    return np.asarray(ys),np.asarray(scores),np.asarray(baselines),gt

def metrics(y,p,gt,t):
    selected=p>=t;positive=y>0;tp=int((selected&positive).sum());fp=int((selected&~positive).sum());fn=gt-tp
    repeat=y==2
    return dict(threshold=float(t),tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(gt,1),f1=2*tp/max(2*tp+fp+fn,1),
                reattack_candidate_retention=float(selected[repeat].mean()) if repeat.any() else 1.)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--epochs',type=int,default=12)
    a=ap.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'history.jsonl').exists():raise ValueError('Use a fresh run directory; this experiment does not silently restart')
    seed=20261016;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    digest=hashlib.sha256(a.baseline.read_bytes()).hexdigest()
    metadata=json.loads((a.root/'acoustic-v3/metadata.json').read_text())
    if metadata['baseline_sha256']!=digest:raise ValueError('Baseline checkpoint differs from feature cache')
    baseline=torch.load(a.baseline,map_location='cpu',weights_only=False);base_threshold=baseline['threshold']
    loaders={s:torch.utils.data.DataLoader(AcousticDataset(a.root,a.root/'manifest.jsonl',s),batch_size=4,shuffle=s=='train',collate_fn=pack) for s in ['train','validation','test']}
    model=AcousticResidualRefiner().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    y,p,b,gt=predict(model,loaders['validation'],a.device);base_val=metrics(y,b,gt,base_threshold)
    best=base_val['f1'];threshold=base_threshold;epoch_best=-1
    def save(epoch,t):
        torch.save(dict(model=model.state_dict(),schema='acoustic-v3',baseline_sha256=digest,threshold=t,epoch=epoch),a.run/'best.pt')
    save(-1,base_threshold)
    for epoch in range(a.epochs):
        began=time.time();model.train();losses=[]
        for batch in loaders['train']:
            batch={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in batch.items()}
            pred=forward(model,batch);mask=batch['valid']
            loss=F.cross_entropy(pred[mask],batch['action'][mask])+.01*(pred[mask]-batch['base_logits'][mask]).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss))
        y,p,b,gt=predict(model,loaders['validation'],a.device)
        choices=[metrics(y,p,gt,t) for t in np.linspace(.02,.98,97)]
        eligible=[r for r in choices if r['recall']>=base_val['recall']-.005 and r['reattack_candidate_retention']>=base_val['reattack_candidate_retention']-.005]
        current=max(eligible,key=lambda r:r['f1']) if eligible else None
        if current and current['f1']>best:
            best=current['f1'];threshold=current['threshold'];epoch_best=epoch;save(epoch,threshold)
        line=dict(epoch=epoch,seconds=time.time()-began,train_loss=float(np.mean(losses)),validation=current,selected_epoch=epoch_best)
        with (a.run/'history.jsonl').open('a') as f:f.write(json.dumps(line)+'\n')
        print(json.dumps(line),flush=True)
    ck=torch.load(a.run/'best.pt',map_location=a.device,weights_only=False);model.load_state_dict(ck['model'])
    y,p,b,gt=predict(model,loaders['test'],a.device)
    report=dict(seed=seed,trainable_parameters=sum(p.numel() for p in model.parameters()),selected_epoch=epoch_best,
        validation_baseline=base_val,validation_best_f1=best,test_baseline=metrics(y,b,gt,base_threshold),test_v3=metrics(y,p,gt,threshold),
        fallback_to_v2=epoch_best==-1,selection='Validation F1 with recall and true-reattack retention floors within 0.5 percentage points of baseline',
        limitation='Same synthetic test set as v2, not fresh real-audio validation. No song-specific edits or timing labels.')
    (a.run/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
