import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class AcousticResidualRefiner(nn.Module):
    """Audio-driven residual to frozen v2 logits; does not modify durations."""
    def __init__(self):
        super().__init__()
        self.audio=nn.Sequential(nn.Conv2d(3,12,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(12,24,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d((3,3)),
            nn.Flatten(),nn.Linear(216,64),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(64+3+24+10,96),nn.GELU(),nn.Dropout(.15))
        self.rnn=nn.GRU(96,48,batch_first=True,bidirectional=True)
        self.delta=nn.Linear(96,3)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self,raw,base_logits,context,position,valid):
        b,n,c,f,t=raw.shape
        encoded=self.audio(raw.reshape(b*n,c,f,t)).reshape(b,n,-1)
        x=self.fuse(torch.cat([encoded,base_logits.detach(),context,position],-1))
        packed=pack_padded_sequence(x,valid.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        seq,_=self.rnn(packed);seq,_=pad_packed_sequence(seq,batch_first=True,total_length=n)
        return base_logits.detach()+self.delta(seq+x)


class AttackEvidenceRefiner(AcousticResidualRefiner):
    """Learn attack shape from both AMT views alongside independent audio evidence."""
    def __init__(self):
        super().__init__()
        self.attack=nn.Sequential(nn.Conv1d(12,24,5,padding=2),nn.GELU(),
            nn.Conv1d(24,32,5,stride=2,padding=2),nn.GELU(),
            nn.AdaptiveAvgPool1d(5),nn.Flatten(),nn.Linear(160,32),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(64+32+3+24+10,96),nn.GELU(),nn.Dropout(.15))

    def forward(self,raw,base_logits,context,position,valid,patch):
        b,n,c,f,t=raw.shape
        audio=self.audio(raw.reshape(b*n,c,f,t)).reshape(b,n,-1)
        attack=self.attack(patch.reshape(b*n,12,61)).reshape(b,n,-1)
        x=self.fuse(torch.cat([audio,attack,base_logits.detach(),context,position],-1))
        packed=pack_padded_sequence(x,valid.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        seq,_=self.rnn(packed);seq,_=pad_packed_sequence(seq,batch_first=True,total_length=n)
        return base_logits.detach()+self.delta(seq+x)


def model_for_checkpoint(checkpoint):
    architecture=checkpoint.get('architecture','acoustic')
    if architecture=='attack':return AttackEvidenceRefiner()
    if architecture=='acoustic':return AcousticResidualRefiner()
    raise ValueError('Unsupported architecture: '+architecture)
