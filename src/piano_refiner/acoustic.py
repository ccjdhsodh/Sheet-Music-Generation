"""Pitch-relative multi-resolution audio evidence, independent of AMT scores."""
import numpy as np
import librosa

SCHEMA=1

def acoustic_patches(audio,sr,notes):
    y=np.asarray(audio,np.float32)
    if y.ndim==2:y=y.mean(axis=1)
    y=librosa.resample(y,orig_sr=sr,target_sr=16000)
    # Global gain invariance; no per-note loudness normalization that could
    # turn weak leakage into a strong attack.
    y=y/max(float(np.sqrt(np.mean(y*y))),1e-6)
    maps=[]
    pitches=np.arange(9,145)
    for fft in [1024,4096]:
        spec=abs(librosa.stft(y,n_fft=fft,hop_length=160))
        freq=librosa.fft_frequencies(sr=16000,n_fft=fft)
        bands=[]
        for pitch in pitches:
            hz=librosa.midi_to_hz(pitch)
            distance=abs(freq-hz);width=max(16000/fft,hz*.03)
            w=np.maximum(0,1-distance/width)
            bands.append((w@spec)/max(w.sum(),1e-8))
        energy=np.asarray(bands,np.float32)
        db=20*np.log10(np.maximum(energy,1e-6))
        db=np.clip((db-20*np.log10(max(float(energy.max()),1e-6)))/80,-1,0)
        maps.append(db)
    flux=np.clip(np.diff(maps[0],axis=1,prepend=maps[0][:,:1]),-.5,.5)
    features=np.stack([maps[0],maps[1],flux])
    result=[]
    for note in notes:
        k=int(note['midi_note']);t=round(float(note['onset_time'])*100)
        pi=np.clip(np.arange(k-12,k+37)-9,0,len(pitches)-1)
        ti=np.clip(np.arange(t-16,t+25),0,features.shape[-1]-1)
        result.append(features[:,pi[:,None],ti[None,:]])
    return np.asarray(result,np.float32).reshape(len(notes),3,49,41)
