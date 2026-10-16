from pathlib import Path
import argparse,json,hashlib
import numpy as np,soundfile as sf,torch
from .acoustic import acoustic_patches,SCHEMA
from .model import DualViewRefiner
from .data import load_manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    digest=hashlib.sha256(a.baseline.read_bytes()).hexdigest()
    ck=torch.load(a.baseline,map_location='cpu',weights_only=False)
    model=DualViewRefiner(**ck['config']);model.load_state_dict(ck['model']);model.eval()
    out=a.root/'acoustic-v3';out.mkdir(exist_ok=True)
    rows=load_manifest(a.root/'manifest.jsonl')
    for i,r in enumerate(rows):
        dest=out/f"{r['id']}.npz"
        if dest.exists():
            with np.load(dest) as z:
                if int(z['schema'])!=SCHEMA or str(z['baseline_sha256'])!=digest:raise ValueError('Stale acoustic cache')
            continue
        with np.load(a.root/'features'/f"{r['id']}.npz") as z:d={k:z[k] for k in z.files}
        notes=json.loads((a.root/'features'/f"{r['id']}.json").read_text())
        audio,sr=sf.read(a.root/'synthetic'/f"{r['id']}.wav")
        raw=acoustic_patches(audio,sr,notes)
        with torch.no_grad():
            result=model(torch.from_numpy(d['patch'])[None],torch.from_numpy(d['context'])[None],torch.from_numpy(d['position'])[None],torch.ones(1,len(notes),dtype=torch.bool))
        temp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(temp,raw=raw.astype(np.float16),base_logits=result['action'][0].numpy(),schema=SCHEMA,baseline_sha256=digest)
        temp.replace(dest)
        print(json.dumps({'stage':'acoustic_features','index':i+1,'total':len(rows)}),flush=True)
    (out/'metadata.json').write_text(json.dumps({'schema':SCHEMA,'baseline_sha256':digest,'clips':len(rows)},indent=2))
if __name__=='__main__':main()
