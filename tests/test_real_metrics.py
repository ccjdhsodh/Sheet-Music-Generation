import unittest
from piano_refiner.stratified_metrics import score_notes,aggregate,paired_interval
from piano_refiner.real_data import choose

class RealMetricsTests(unittest.TestCase):
    def test_missing_bass_counts_even_without_candidate(self):
        gt=[dict(start=1,end=2,pitch=36,velocity=35),dict(start=2,end=2.1,pitch=72,velocity=70,original_duration=.1,is_repeat=True)]
        pred=[dict(onset_time=2,midi_note=72),dict(onset_time=2.01,midi_note=72)]
        r=aggregate([score_notes(pred,gt)])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,1))
        self.assertEqual(r['groups']['bass']['recall'],0)
        self.assertEqual(r['groups']['short']['recall'],1)
        self.assertEqual(r['groups']['repeated']['reference'],1)
    def test_boundary_and_empty_group(self):
        gt=[dict(start=.1,end=.2,pitch=60,velocity=60)]
        r=aggregate([score_notes([],gt)])
        self.assertEqual(r['fn'],0);self.assertIsNone(r['groups']['bass']['recall'])
    def test_pairing_identical_models(self):
        r=score_notes([],[])
        ci=paired_interval({'a':{'v2':[r],'v3':[r]},'b':{'v2':[r],'v3':[r]}})
        self.assertEqual(ci['f1_delta_95_percentile'],[0,0])
    def test_selection_preserves_groups(self):
        rows=[dict(split=s,canonical_composer='composer',canonical_title=str(i),midi_filename=f'{s}-{i}',duration=100)
              for i,s in enumerate(['train','validation','test'])]
        selected=choose(rows,{'train':1,'validation':1,'test':1})
        self.assertEqual([r['split'] for r in selected],['train','validation','test'])
if __name__=='__main__':unittest.main()
