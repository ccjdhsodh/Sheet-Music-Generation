import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from piano_refiner.model import DualViewRefiner
from piano_refiner.targets import match_targets,positions
from piano_refiner.data import load_manifest

class CoreTests(unittest.TestCase):
    def test_one_to_one_and_real_reattacks(self):
        gt=[dict(start=1.,end=1.1,pitch=60,velocity=70),dict(start=1.2,end=1.3,pitch=60,velocity=40)]
        notes=[dict(onset_time=t,midi_note=60) for t in [1.,1.02,1.2]]
        labels=match_targets(notes,gt)
        self.assertEqual(labels['action'].tolist(),[1,0,2])
        self.assertAlmostEqual(float(labels['duration'][2]),float(np.log1p(.1)),places=6)
    def test_assignment_maximizes_matches(self):
        gt=[dict(start=t,end=t+.1,pitch=60,velocity=50) for t in [1.,1.10]]
        notes=[dict(onset_time=t,midi_note=60) for t in [1.04,.95]]
        self.assertEqual(int((match_targets(notes,gt)['action']>0).sum()),2)
    def test_padding_does_not_change_real_predictions(self):
        torch.manual_seed(4);torch.set_num_threads(2)
        net=DualViewRefiner(width=32,layers=1,dropout=0).eval()
        p=torch.randn(1,3,12,61);c=torch.randn(1,3,24);pos=torch.randn(1,3,10)
        a=net(p,c,pos,torch.ones(1,3,dtype=torch.bool))
        b=net(torch.cat([p,torch.randn(1,2,12,61)],1),torch.cat([c,torch.randn(1,2,24)],1),torch.cat([pos,torch.randn(1,2,10)],1),torch.tensor([[1,1,1,0,0]],dtype=torch.bool))
        for key in a:torch.testing.assert_close(a[key],b[key][:,:3],atol=2e-6,rtol=2e-6)
        sum(v.sum() for v in a.values()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.jsonl'
            p.write_text('\n'.join(json.dumps(dict(id=str(i),composition_id='same',split=s)) for i,s in enumerate(['train','test'])))
            with self.assertRaises(ValueError):load_manifest(p)
    def test_chord_positions_are_order_independent(self):
        ns=[dict(onset_time=1.,midi_note=p) for p in [60,64]]
        np.testing.assert_array_equal(positions(ns),positions(ns[::-1])[::-1])
if __name__=='__main__':unittest.main()
