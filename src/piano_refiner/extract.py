from pathlib import Path
import argparse,contextlib,json,os
import numpy as np
import soundfile as sf
import torch,librosa
from .data import load_manifest
from .features import candidates,build_features
from .targets import match_targets,positions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();torch.set_num_threads(2)
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from piano_transcription_inference import PianoTranscription
    from piano_transcription_inference.utilities import RegressionPostProcessor
    separator=get_model('htdemucs_6s')
    transcriber=PianoTranscription(checkpoint_path=str(a.checkpoint),device=a.device)
    out=a.root/'features';out.mkdir(exist_ok=True)
    rows=load_manifest(a.root/'manifest.jsonl')
    if a.limit:rows=rows[:a.limit]
    for index,r in enumerate(rows):
        dst=out/f"{r['id']}.npz"
        if dst.exists():continue
        x,sr=sf.read(a.root/'synthetic'/f"{r['id']}.wav",dtype='float32',always_2d=True)
        gt=json.loads((a.root/'synthetic'/f"{r['id']}.json").read_text())
        wav=torch.from_numpy(librosa.resample(x.T,orig_sr=sr,target_sr=44100))
        mean=wav.mean();std=wav.mean(0).std().clamp_min(1e-6)
        with torch.no_grad():
            src=apply_model(separator,((wav-mean)/std)[None],device=a.device,shifts=0,split=True,progress=False)[0].cpu()
        piano=(src[separator.sources.index('piano')]*std+mean).numpy().mean(0)
        views=[];events=[]
        with open(os.devnull,'w') as null,contextlib.redirect_stdout(null):
            for signal,rate in [(x.mean(1),sr),(piano,44100)]:
                result=transcriber.transcribe(librosa.resample(signal,orig_sr=rate,target_sr=16000),None)
                views.append(result['output_dict'])
                # Wider proposal pool than v1; positives may be recovered, not only deleted.
                processor=RegressionPostProcessor(100,88,.15,.3,.1,.2)
                ns,_=processor.output_dict_to_midi_events({k:v.copy() for k,v in views[-1].items()})
                events.append(ns)
        ns=candidates(events,len(x)/sr)
        if not ns:raise RuntimeError(f"No candidates for {r['id']}")
        ns,patch,context=build_features(ns,views)
        targets=match_targets(ns,gt['notes'])
        np.savez_compressed(dst,schema=3,patch=patch,context=context,position=positions(ns),
             gt_count=len(gt['notes']),**targets)
        (out/f"{r['id']}.json").write_text(json.dumps(ns,default=lambda x:x.item()),encoding='utf-8')
        print(json.dumps({'clip':r['id'],'index':index+1,'total':len(rows),'candidates':len(ns),'positive':int((targets['action']>0).sum())}),flush=True)

if __name__=='__main__':main()
