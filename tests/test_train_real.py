import copy,unittest
import numpy as np
from piano_refiner.train_real import retention_weights,acceptable


class AdaptationTests(unittest.TestCase):
    def test_difficult_positives_weighted_false_notes_not_protected(self):
        truth=[dict(pitch=40,velocity=30,start=0,end=.1)]
        w=retention_weights([{},{}],truth,np.array([0,-1]),np.array([2,0]))
        np.testing.assert_equal(w,[3,1])

    def test_condition_and_subgroup_regression_rejected(self):
        base={c:dict(precision=.9,groups={g:dict(recall=.8) for g in ['all','bass','weak','short','repeated']}) for c in ['clean','strings10','strings0']}
        self.assertTrue(acceptable(base,base))
        changed=copy.deepcopy(base);changed['strings10']['groups']['bass']['recall']=.79
        self.assertFalse(acceptable(changed,base))
        changed=copy.deepcopy(base);changed['clean']['precision']=.88
        self.assertFalse(acceptable(changed,base))


if __name__=='__main__':unittest.main()
