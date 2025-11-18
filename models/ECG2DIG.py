import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

""" ECG2DID model from paper: (coming soon). 
    Extends ECG12Net from paper: Lin C.-S, Lin C, Fang W.-H, Hsu C.-J, 
    Chen S.-J, Huang K.-H, et al. A Deep-Learning Algorithm
    (ECG12Net) for Detecting Hypokalemia and Hyperkalemia by Electrocardiography: Algorithm
    Development. JMIR medical informatics 2020;8:e15931–e15931.

    With the following modifications by us:
    0. Downsampling
    1. Addition of FIRFilterLayer
    2. Addition of depthwise convolutions
    3. Updated DenseUnit
    4. Updated Pooling unit
    5. Added a whole new AttentionBlockV2

"""

# FIRFilterLayer: Fixed FIR filter using Conv1d
class FIRFilterLayer(nn.Module):
    def __init__(self, num_channels, fs=500.0, lowcut=0.5, highcut=40.0, numtaps=101):
        super().__init__()
        nyquist = 0.5 * fs
        taps = signal.firwin(numtaps, [lowcut/nyquist, highcut/nyquist], pass_zero=False)
        kernel = torch.tensor(taps, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=numtaps,
                              groups=num_channels, bias=False, padding=numtaps//2)
        with torch.no_grad():
            self.conv.weight.copy_(kernel.repeat(num_channels,1,1))
        for p in self.conv.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.conv(x)

# Depthwise separable convolution
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# DownsampleBlock: Convolution + Adaptive Pooling
class DownsampleBlock(nn.Module):
    def __init__(self, target_length=2048):
        super().__init__()
        self.conv = nn.Conv1d(1,1,kernel_size=7,stride=1,padding=3)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool1d(target_length)
    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class DenseUnit(nn.Module):
    def __init__(self, in_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,128,kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = DepthwiseSeparableConv1d(128,128,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,32,kernel_size=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu = nn.SiLU()
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return torch.cat([x,out],dim=1) if out.shape[-1]==x.shape[-1] else out

class PoolingUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,128,1)
        self.bn1   = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128,128,3,stride=2,padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,32,1)
        self.bn3   = nn.BatchNorm1d(32)
        self.relu  = nn.SiLU()
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out

class PoolingBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.unit = PoolingUnit(in_channels)
        self.avg  = nn.AvgPool1d(2,2)
    def forward(self,x):
        return torch.cat([self.avg(x), self.unit(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self,in_channels,num_units):
        super().__init__()
        layers=[]; ch=in_channels
        for _ in range(num_units): layers.append(DenseUnit(ch)); ch+=32
        self.block=nn.Sequential(*layers)
        self.out_channels=ch
    def forward(self,x): return self.block(x)

class ECGLeadBlock(nn.Module):
    '''
    ECGLeadBlock parametrizes the per-lead classifier by num_classes
    '''
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fir    = FIRFilterLayer(1)
        self.down  = DownsampleBlock(target_length=1024)  # ← explicitly pass 1024
        self.bn0    = nn.BatchNorm1d(1)
        self.conv0  = nn.Conv1d(1,32,7,2,3)
        self.bn1    = nn.BatchNorm1d(32)
        self.relu   = nn.SiLU()
        self.pool1  = PoolingBlock(32)   
        self.dense1 = DenseBlock(64,3)
        self.pool2  = PoolingBlock(160)  
        self.dense2 = DenseBlock(192,3)
        self.pool3  = PoolingBlock(288)  
        self.dense3 = DenseBlock(320,6)
        self.pool4  = PoolingBlock(512)  
        self.dense4 = DenseBlock(544,6)
        self.pool5  = PoolingBlock(736) 
        self.dense5 = DenseBlock(768,3)
        self.bn_f   = nn.BatchNorm1d(864)
        self.relu_f = nn.SiLU()
        self.gpool  = nn.AdaptiveAvgPool1d(1)
        self.dropout= nn.Dropout(0.6)  # stronger regularization was 0.5
        #self.fc     = nn.Linear(864,2)
        self.fc     = nn.Linear(864, num_classes)   # instead of 2 now dynamic
        
    def forward(self,x):
        x = self.fir(x) 
        x = self.down(x)
        x = self.relu(self.bn1(self.conv0(self.bn0(x))))
        x = self.dense1(self.pool1(x)) 
        x = self.dense2(self.pool2(x))
        x = self.dense3(self.pool3(x))
        x = self.dense4(self.pool4(x))
        x = self.dense5(self.pool5(x))
        x = self.relu_f(self.bn_f(x))
        f = x = self.gpool(x).squeeze(-1)      # per-lead feature (864)
        return f, self.fc(self.dropout(f))     # per-lead logits: [B*12, num_classes]

class AttentionBlockV2(nn.Module):
    """ AN ALTERNATIVE BLOCK
        Lead-wise attention (across 12 leads) with:
      - per-lead embeddings to break symmetry
      - LayerNorm (no BatchNorm)
      - 2-layer MLP scorer with GELU
      - learnable temperature for softmax sharpness

    Input:  features [B, 12, F]
    Output: attn_weights [B, 12, 1]  (softmax over the lead axis)
    """
    def __init__(self, 
                 in_features=864, 
                 hidden=128, 
                 num_leads=12, 
                 p_drop=0.2,
                 init_embed_std=1e-3,
                 use_temperature=True,
                 tau_min=0.5,
                 tau_max=3.0,
                 init_tau=1.0,):
        
        super().__init__()
        self.num_leads = num_leads
        self.use_temperature = use_temperature

        # Small random lead embeddings to break permutation symmetry
        self.lead_embed = nn.Parameter(torch.zeros(num_leads, in_features))
        nn.init.trunc_normal_(self.lead_embed, std=init_embed_std)

        # MLP scorer (no BatchNorm; LayerNorm per-lead features)
        self.ln = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.drop = nn.Dropout(p_drop)  # <--- added for extra regularization
        self.fc2 = nn.Linear(hidden, 1)

        # Learnable temperature (tau>0) with bounds. We store log_tau for positivity.
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        if self.use_temperature:
            # invert the squash so that initial tau ≈ init_tau
            init_tau = float(init_tau)
            init_tau_clamped = max(min(init_tau, self.tau_max - 1e-6), self.tau_min + 1e-6)
            # map init_tau to raw via inverse of affine+sigmoid
            t = (init_tau_clamped - self.tau_min) / (self.tau_max - self.tau_min)
            t = max(min(t, 1 - 1e-6), 1e-6)
            raw = torch.log(torch.tensor(t/(1 - t)))  # logit
            self.raw_tau = nn.Parameter(raw)
        else:
            self.register_parameter('raw_tau', None)

    def _tau(self):
        if not self.use_temperature:
            return torch.tensor(1.0, device=self.fc2.weight.device)
        # τ in [tau_min, tau_max] using a sigmoid squash
        s = torch.sigmoid(self.raw_tau)
        return self.tau_min + (self.tau_max - self.tau_min) * s
            
    def forward(self, features):  # [B, 12, F]
        B, L, Fdim = features.shape
        assert L == self.num_leads, f"expected {self.num_leads} leads, got {L}"

        # Add per-lead embeddings
        x = features + self.lead_embed.unsqueeze(0)  # [B, 12, F]

        # Score each lead independently
        x = self.ln(x)
        x = self.fc1(x)           # [B, 12, hidden]
        x = F.gelu(x)
        x = self.drop(x)
        logits = self.fc2(x)      # [B, 12, 1]
        
        # Temperature (scalar per module)
        tau = self._tau()

        with torch.cuda.amp.autocast(False):
            logits32 = logits.float()
            attn = F.softmax(logits32 / tau, dim=1).to(logits.dtype)  # [B, 12, 1]
        return attn
    

class AttentionBlock(nn.Module):
    def __init__(self,in_features=864):
        super().__init__()
        self.fc1=nn.Linear(in_features,8); 
        self.bn1=nn.BatchNorm1d(8)
        self.relu=nn.SiLU(); 
        self.fc2=nn.Linear(8,1); 
    def forward(self,feat):
        b,l,d = feat.shape
        x = feat.view(b*l,d)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)                     
        x = x.view(b, l, 1)
        return torch.softmax(x,dim=1)

class ECG12DrugNet(nn.Module):
    """
    Multi-head ECG network with dynamic CLS head size.
    Expected encoder API: encoder(x_12T) -> (z, attn) or z
      - z: [B, feat_dim] global embedding
      - attn (optional): lead weights with a 12-dim (e.g., [B,12] or [B,12,1])
    Pass num_classes through; everything else unchanged
    """
    def __init__(
        self, 
        name: str, 
        num_classes: int = 2,
        feat_dim: int = 512,  # embedding size from encoder
        cls_dropout: float = 0.1,
        encoder: Optional[nn.Module] = None,  
        **encoder_kwargs,              
    ):
        super().__init__()
        if num_classes not in (2, 3):
            raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")
        self.name = name
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)

        # Dynamic loss‐weight params. Ultimately not all of them
        # will be used. That is up to the Trainer.
        self.log_sigma_class = nn.Parameter(torch.zeros(1))
        self.log_sigma_age = nn.Parameter(torch.zeros(1))
        self.log_sigma_digoxin = nn.Parameter(torch.zeros(1))
        self.log_sigma_sex = nn.Parameter(torch.zeros(1))
        self.log_sigma_hr = nn.Parameter(torch.zeros(1))

        # shared per-lead trunk (added num_classes)
        self.lead_block = ECGLeadBlock(num_classes=self.num_classes)

        #self.attn_class = AttentionBlock(in_features=864) # Replaced by AttentionBlockV2
        self.attn_class = AttentionBlockV2(in_features=864, hidden=128, 
                                          num_leads=12, init_embed_std=1e-3,
                                          init_tau=1.0)

        self.classifier = nn.Identity()

        # ─────────── new projection layer ───────────
        # Collapse 864 -> 256
        self.feature_proj = nn.Linear(864, 256)

        # Now all downstream heads take 256-dim inputs
        self.age_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.digoxin_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.sex_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2))
        self.hr_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,12,5000) or (B,5000,12)
        if x.shape[1] != 12:
            x = x.permute(0,2,1)
        B,L,T = x.size()
        x = x.reshape(B*L, 1, T)

        # Per-lead features + per-lead logits (num_classes dynamic)
        feats, lead_logits = self.lead_block(x)
        feats = feats.reshape(B, L, -1)             # (B,12,864)
        lead_logits = lead_logits.reshape(B, L, -1) # (B,12,2)

        # Attention & classification 
        attn_cls = self.attn_class(feats)              # (B,12,1)
        agg_log = (lead_logits * attn_cls).sum(1)      # (B,2)
        cls_out = self.classifier(agg_log)             # logits

        # Shared pooled features for all other heads
        agg_feat = (feats * attn_cls).sum(1)           # (B,864)

        # ─ shrink to 256 ─
        agg_feat = self.feature_proj(agg_feat)      # (B,256)

        # — age regression —
        age_out = self.age_head(agg_feat)

        # — digoxin regression —
        dig_out = self.digoxin_head(agg_feat)

        # — sex classification —
        sex_out = self.sex_head(agg_feat)

        # — HR regression —
        hr_out  = self.hr_head(agg_feat)

        return cls_out, attn_cls, hr_out, dig_out, age_out, sex_out
    
