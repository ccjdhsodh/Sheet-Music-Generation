import unittest
import numpy as np,torch
from piano_refiner.acoustic import acoustic_patches
from piano_refiner.acoustic_model import AcousticResidualRefiner,AttackEvidenceRefiner,model_for_checkpoint

class AcousticTests(unittest.TestCase):
    def test_attack_branch_receives_gradient_and_ignores_padding(self):
        torch.manual_seed(11);torch.set_num_threads(2);m=AttackEvidenceRefiner().eval()
        args=[torch.randn(1,3,3,49,41),torch.randn(1,3,3),torch.randn(1,3,24),torch.randn(1,3,10)]
        valid=torch.ones(1,3,dtype=torch.bool);patch=torch.randn(1,3,12,61)
        torch.testing.assert_close(m(*args,valid,patch),args[1])
        torch.nn.init.normal_(m.delta.weight,std=.03)
        a=m(*args,valid,patch)
        pad=lambda x:torch.cat([x,torch.randn((1,2)+x.shape[2:])],1)
        b=m(*[pad(x) for x in args],torch.tensor([[1,1,1,0,0]],dtype=torch.bool),pad(patch))
        torch.testing.assert_close(a,b[:,:3],rtol=2e-5,atol=2e-5)
        a.square().mean().backward()
        self.assertGreater(float(m.attack[0].weight.grad.abs().sum()),0)
        self.assertIsInstance(model_for_checkpoint({}),AcousticResidualRefiner)
        self.assertIsInstance(model_for_checkpoint({'architecture':'attack'}),AttackEvidenceRefiner)

    def test_silence_and_gain(self):
        notes=[dict(onset_time=.3,midi_note=69),dict(onset_time=.6,midi_note=72)]
        sr=16000;t=np.arange(sr,dtype=np.float32)/sr;y=np.sin(2*np.pi*440*t)*(t>.2)
        a=acoustic_patches(y,sr,notes);b=acoustic_patches(y*.1,sr,notes)
        self.assertEqual(a.shape,(2,3,49,41));np.testing.assert_allclose(a,b,atol=2e-4)
        self.assertTrue(np.isfinite(acoustic_patches(np.zeros(sr),sr,notes)).all())
    def test_residual_starts_at_baseline_and_padding_is_ignored(self):
        torch.manual_seed(8);torch.set_num_threads(2);m=AcousticResidualRefiner().eval()
        raw=torch.randn(1,3,3,49,41);base=torch.randn(1,3,3);ctx=torch.randn(1,3,24);pos=torch.randn(1,3,10)
        valid=torch.ones(1,3,dtype=torch.bool)
        torch.testing.assert_close(m(raw,base,ctx,pos,valid),base)
        torch.nn.init.normal_(m.delta.weight,std=.03)
        a=m(raw,base,ctx,pos,valid)
        pad=lambda x:torch.cat([x,torch.randn((1,2)+x.shape[2:])],dim=1)
        b=m(pad(raw),pad(base),pad(ctx),pad(pos),torch.tensor([[1,1,1,0,0]],dtype=torch.bool))
        torch.testing.assert_close(a,b[:,:3],rtol=2e-5,atol=2e-5)
        a.square().mean().backward()
        self.assertGreater(float(m.audio[0].weight.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
