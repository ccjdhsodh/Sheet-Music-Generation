"""Small, provenance-recorded MAESTRO pilot; real piano plus synthetic strings."""
from pathlib import Path
import argparse,csv,hashlib,io,json,urllib.request,zipfile
import numpy as np,soundfile as sf
from .remote_zip import HTTPRangeReader

BASE='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/'

def composition_key(r):return r['canonical_composer'].strip()+' :: '+r['canonical_title'].strip()

def choose(rows,counts):
    selected=[];seen={}
    for split,count in counts.items():
        options=[r for r in rows if r['split']==split and 80<float(r['duration'])<420]
        options.sort(key=lambda r:hashlib.sha256(('real-pilot-20261016'+r['midi_filename']).encode()).hexdigest())
        chosen=0
        for r in options:
            key=composition_key(r)
            if key in seen:continue
            seen[key]=split;selected.append(r);chosen+=1
            if chosen==count:break
        if chosen!=count:raise ValueError('Insufficient distinct compositions')
    return selected

def crop_truth(allnotes,start):
    result=[]
    for n in allnotes:
        if not start<=n.start<start+12:continue
        previous=[p for p in allnotes if p.pitch==n.pitch and 0<n.start-p.start<=.6]
        result.append(dict(start=float(n.start-start),end=float(min(12,n.end-start)),pitch=int(n.pitch),velocity=int(n.velocity),
            original_duration=float(n.end-n.start),is_repeat=bool(previous)))
    return sorted(result,key=lambda n:(n['start'],n['pitch']))

def main():
    import pretty_midi,librosa
    from .synthesize import render
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--font',type=Path,required=True)
    ap.add_argument('--train-pieces',type=int,default=8);ap.add_argument('--validation-pieces',type=int,default=4);ap.add_argument('--test-pieces',type=int,default=8)
    a=ap.parse_args();dest=a.root/'synthetic';dest.mkdir(parents=True,exist_ok=True)
    meta=a.root/'maestro-v3.0.0.csv'
    if not meta.exists():urllib.request.urlretrieve(BASE+meta.name,meta)
    rows=list(csv.DictReader(meta.open(encoding='utf-8')));selected=choose(rows,{'train':a.train_pieces,'validation':a.validation_pieces,'test':a.test_pieces})
    plan={'dataset':'MAESTRO v3.0.0','source':BASE+'maestro-v3.0.0.zip','license':'CC BY-NC-SA 4.0',
          'license_url':'https://creativecommons.org/licenses/by-nc-sa/4.0/','selection':selected,
          'font_sha256':hashlib.sha256(a.font.read_bytes()).hexdigest(),'note':'Real piano recording; added strings are synthesized, not a real ensemble recording.'}
    planfile=a.root/'provenance.json'
    if planfile.exists() and json.loads(planfile.read_text())!=plan:raise ValueError('Different dataset plan; use fresh output directory')
    planfile.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    reader=HTTPRangeReader(plan['source']);archive=zipfile.ZipFile(reader);manifest=[]
    for index,row in enumerate(selected):
        pid=hashlib.sha256(row['midi_filename'].encode()).hexdigest()[:12]
        expected=[dest/f'{pid}_{j}_{c}.json' for j in range(3) for c in ['clean','strings10','strings0']]
        if all(p.exists() for p in expected):
            for p in expected:
                d=json.loads(p.read_text());manifest.append({k:d[k] for k in ['id','composition_id','split','condition','piece_id']})
            continue
        prefix='maestro-v3.0.0/'
        midi_bytes=archive.read(prefix+row['midi_filename'])
        midi=pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes));allnotes=sorted([n for inst in midi.instruments if not inst.is_drum for n in inst.notes],key=lambda n:n.start)
        # ZipFile validates per-entry CRC; full archive SHA is not claimed.
        audio_bytes=archive.read(prefix+row['audio_filename'])
        audio,sr=sf.read(io.BytesIO(audio_bytes),dtype='float32',always_2d=True)
        duration=len(audio)/sr
        if abs(duration-float(row['duration']))>15:raise ValueError('Unexpected audio/MIDI duration discrepancy')
        for j,fraction in enumerate([.2,.5,.8]):
            start=float(np.clip(duration*fraction,1,duration-13))
            clean=librosa.resample(audio[round(start*sr):round((start+12)*sr)].T,orig_sr=sr,target_sr=16000).T
            if len(clean)!=192000:clean=np.pad(clean,((0,max(0,192000-len(clean))),(0,0)))[:192000]
            gt=crop_truth(allnotes,start)
            rng=np.random.default_rng(int(pid[:8],16)+j)
            strings=[]
            for t in np.arange(.2,11,.8):
                pool=[n for n in gt if abs(n['start']-t)<1.2]
                pitch=(int(rng.choice(pool)['pitch']) if pool else int(rng.integers(60,85)))
                pitch=int(np.clip(pitch+int(rng.choice([0,0,12,-12])),48,96))
                strings.append(dict(start=float(t),end=float(min(11.8,t+rng.uniform(.8,2.5))),pitch=pitch,velocity=int(rng.integers(45,90))))
            interference=render(strings,a.font,40,[],True)
            piano_rms=max(float(np.sqrt(np.mean(clean**2))),1e-8);string_rms=max(float(np.sqrt(np.mean(interference**2))),1e-8)
            for condition,snr in [('clean',None),('strings10',10),('strings0',0)]:
                clip=f'{pid}_{j}_{condition}';x=clean.copy()
                if snr is not None:x+=interference*(piano_rms/string_rms)*10**(-snr/20)
                x*=min(1,.95/max(float(abs(x).max()),1e-8))
                sf.write(dest/f'{clip}.wav',x,16000,subtype='PCM_24')
                record=dict(id=clip,composition_id=composition_key(row),piece_id=pid,split=row['split'],condition=condition)
                manifest.append(record)
                (dest/f'{clip}.json').write_text(json.dumps({**record,'schema':3,'notes':gt,'seconds':12,'crop_start':start,
                    'source_audio':row['audio_filename'],'source_midi':row['midi_filename'],'snr_db':snr,
                    'evaluation_window':[.3,11.7],'string_notes':strings if snr is not None else [],
                    'source_audio_sha256':hashlib.sha256(audio_bytes).hexdigest()},indent=2),encoding='utf-8')
        print(json.dumps({'stage':'real_data','pieces':index+1,'total':len(selected),'downloaded_mb':round(reader.used/1024**2)}),flush=True)
    (a.root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest),encoding='utf-8')
    print(json.dumps({'clips':len(manifest),'hours':len(manifest)*12/3600,'pieces':len(selected)}),flush=True)
if __name__=='__main__':main()
