from pathlib import Path
import json
import numpy as np
import torch


def load_manifest(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    seen={}
    for r in rows:
        key=r['composition_id']
        if key in seen and seen[key]!=r['split']:raise ValueError('Composition leakage across splits')
        seen[key]=r['split']
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate clip IDs')
    return rows


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, root, manifest, split):
        self.root=Path(root)
        self.rows=[r for r in load_manifest(manifest) if r['split']==split]
        if not self.rows:raise ValueError(f'Empty {split} split')
        for r in self.rows:
            if not (self.root/'features'/f"{r['id']}.npz").exists():raise FileNotFoundError(r['id'])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with np.load(self.root/'features'/f"{self.rows[i]['id']}.npz",allow_pickle=False) as z:
            d={k:z[k] for k in z.files}
        if int(d['schema'])!=3:raise ValueError('Incompatible feature schema')
        return d


def collate(items):
    n=max(len(x['action']) for x in items);b=len(items)
    out={'valid':torch.zeros(b,n,dtype=torch.bool),'gt_count':sum(int(x['gt_count']) for x in items)}
    for key in ['patch','context','position','action','velocity','duration']:
        shape=(b,n)+items[0][key].shape[1:]
        out[key]=torch.zeros(shape,dtype=torch.long if key=='action' else torch.float32)
        for i,x in enumerate(items):out[key][i,:len(x[key])]=torch.from_numpy(x[key])
    for i,x in enumerate(items):out['valid'][i,:len(x['action'])]=True
    return out
