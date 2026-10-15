"""Run a logged pipeline with external storage and explicit state, CPU by default."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--fonts',type=Path,nargs='+',required=True);ap.add_argument('--compositions',type=int,default=256)
    ap.add_argument('--variants',type=int,default=2);ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--device',default='cpu')
    a=ap.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    # Validate RIFF file length: never start synthesis on a partial SoundFont.
    for font in a.fonts:
        with font.open('rb') as f:header=f.read(12)
        if header[:4]!=b'RIFF' or header[8:]!=b'sfbk':raise ValueError(f'Invalid SoundFont: {font.name}')
        if int.from_bytes(header[4:8],'little')+8!=font.stat().st_size:raise ValueError(f'Incomplete SoundFont: {font.name}')
    commands=[('synthesis',[sys.executable,'-m','piano_refiner.synthesize','--root',str(a.root),'--fonts',*[str(f) for f in a.fonts],'--compositions',str(a.compositions),'--variants',str(a.variants)]),
              ('extraction',[sys.executable,'-m','piano_refiner.extract','--root',str(a.root),'--checkpoint',str(a.checkpoint),'--device',a.device]),
              ('training',[sys.executable,'-m','piano_refiner.train','--root',str(a.root),'--run',str(a.root/'runs/dual-view-v2'),'--device',a.device,'--epochs',str(a.epochs)])]
    state=a.root/'job-status.json'
    def write(**kw):
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),device=a.device,**kw),indent=2));temp.replace(state)
    for stage,command in commands:
        write(stage=stage,status='running')
        with (a.root/f'{stage}.log').open('a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(stage=stage,status='failed',returncode=result.returncode);raise SystemExit(result.returncode)
    write(stage='complete',status='complete')

if __name__=='__main__':main()
