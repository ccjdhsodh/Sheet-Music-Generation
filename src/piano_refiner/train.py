from pathlib import Path
import argparse,json,random
import numpy as np
import torch
from torch.nn import functional as F
from .data import FeatureDataset,collate
from .model import DualViewRefiner


def objective(out,b):
    valid=b['valid'];pos=valid & (b['action']>0)
    loss=F.cross_entropy(out['action'][valid],b['action'][valid])
    if pos.any():
        loss=loss+.3*F.smooth_l1_loss(out['duration'][pos],b['duration'][pos])+.2*F.l1_loss(out['velocity'][pos],b['velocity'][pos])
    return loss


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ys=[];scores=[];duration_errors=[];velocity_errors=[];gt=0;loss=0;events=0
    for b in loader:
        gt+=b['gt_count'];b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
        out=model(b['patch'],b['context'],b['position'],b['valid']);mask=b['valid'];pos=mask & (b['action']>0)
        events+=int(mask.sum());loss+=float(objective(out,b))*int(mask.sum())
        ys.extend(b['action'][mask].cpu().tolist());scores.extend((1-out['action'].softmax(-1)[...,0])[mask].cpu().tolist())
        duration_errors.extend((out['duration'][pos].expm1()-b['duration'][pos].expm1()).abs().cpu().tolist())
        velocity_errors.extend(((out['velocity'][pos]-b['velocity'][pos]).abs()*127).cpu().tolist())
    return dict(y=np.array(ys),scores=np.array(scores),gt=gt,loss=loss/max(events,1),
        duration_mae=float(np.mean(duration_errors)),velocity_mae=float(np.mean(velocity_errors)))


def metrics(result,threshold):
    pred=result['scores']>=threshold;positive=result['y']>0
    tp=int((pred&positive).sum());fp=int((pred&~positive).sum());fn=result['gt']-tp
    repeat=result['y']==2
    return dict(threshold=float(threshold),tp=tp,fp=fp,fn_including_unproposed=fn,
        precision=tp/max(tp+fp,1),recall=tp/max(result['gt'],1),f1=2*tp/max(2*tp+fp+fn,1),
        true_reattack_candidate_retention=float(pred[repeat].mean()) if repeat.any() else None,
        duration_mae_on_matched_candidates=result['duration_mae'],velocity_mae_on_matched_candidates=result['velocity_mae'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20261016);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    a.run.mkdir(parents=True,exist_ok=True)
    loaders={s:torch.utils.data.DataLoader(FeatureDataset(a.root,a.root/'manifest.jsonl',s),batch_size=a.batch_size,shuffle=s=='train',collate_fn=collate) for s in ['train','validation','test']}
    model=DualViewRefiner().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    start=0;best=float('inf')
    if a.resume:
        ck=torch.load(a.run/'last.pt',map_location=a.device,weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1;best=ck['best']
        torch.set_rng_state(ck['rng'].cpu())
        if a.device.startswith('cuda') and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    history=[]
    for epoch in range(start,a.epochs):
        model.train();losses=[]
        for b in loaders['train']:
            b={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in b.items()}
            opt.zero_grad();out=model(b['patch'],b['context'],b['position'],b['valid']);loss=objective(out,b)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss))
        val=evaluate(model,loaders['validation'],a.device)
        if val['loss']<best:
            best=val['loss'];torch.save({'model':model.state_dict(),'config':model.config,'epoch':epoch,'schema':3},a.run/'best.pt')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None},a.run/'last.pt')
        line=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=val['loss']);history.append(line)
        with (a.run/'history.jsonl').open('a') as f:f.write(json.dumps(line)+'\n')
        print(json.dumps(line),flush=True)
    ck=torch.load(a.run/'best.pt',map_location=a.device,weights_only=False);model.load_state_dict(ck['model'])
    val=evaluate(model,loaders['validation'],a.device)
    # Choose on validation only. Test is opened only for the final report.
    thresholds=np.linspace(.02,.98,97);threshold=float(max(thresholds,key=lambda t:metrics(val,t)['f1']))
    test=evaluate(model,loaders['test'],a.device)
    ck['threshold']=threshold;torch.save(ck,a.run/'best.pt')
    report=dict(seed=a.seed,parameters=sum(p.numel() for p in model.parameters()),selected_epoch=ck['epoch'],
                validation=metrics(val,threshold),test=metrics(test,threshold),candidate_pool_test=metrics(test,0),
                limitation='Procedural synthetic compositions; no claim of real-recording accuracy or cross-font generalization.')
    (a.run/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
