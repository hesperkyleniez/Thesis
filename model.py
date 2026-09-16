# model.py — CNN-GRU with adaptive late fusion
#
# NOTE: this file is not imported by train.py, retrain.py, evaluate.py, or
# latency_test.py -- they all define/import their own CNNGRUFusion (in
# train.py), which uses a gated fusion mechanism and is architecturally
# different from the simple alpha/beta/gamma scalar version below. This
# file is currently unused by the actual pipeline; keep it in sync with
# train.py's classes (or remove it) so it doesn't misrepresent the model
# that was actually trained.
import random
import torch
import torch.nn as nn
from config import DROPOUT



class CNNBlock(nn.Module):
    """C.6: Conv2D → BatchNorm → ReLU → MaxPool"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class SpecAugment(nn.Module):
    """
    SpecAugment — randomly masks frequency bands and time frames.
    Only active during training. Reduces overfitting significantly.
    """
    def __init__(self, freq_mask=20, time_mask=20, num_masks=2):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.num_masks = num_masks

    def forward(self, x):
        if not self.training:
            return x

        # x: (batch, 1, 128, 128)
        batch, _, freq, time = x.shape
        x = x.clone()

        for _ in range(self.num_masks):
            # Frequency masking
            f  = random.randint(0, self.freq_mask)
            f0 = random.randint(0, freq - f)
            x[:, :, f0:f0+f, :] = 0

            # Time masking
            t  = random.randint(0, self.time_mask)
            t0 = random.randint(0, time - t)
            x[:, :, :, t0:t0+t] = 0

        return x


class CNNGRU(nn.Module):
    """
    C.6 + C.7: CNN-GRU with adaptive late fusion

    Log-Mel (1, 128, 128)
        → SpecAugment (training only)
        → CNN 3 blocks (32, 64, 128)
        → AdaptiveAvgPool (128, 4, 4)
        → Reshape (4 timesteps, 512 features)
        → GRU (128 hidden)
        → 128-dim embedding

    C.7 Fusion:
        α·CNN-GRU(128) ∥ β·MFCC(80) ∥ γ·F0(2) = 210-dim
        → FC classifier → logit
    """

    def __init__(self, dropout=0.5):
        super().__init__()

        self.spec_augment = SpecAugment(
            freq_mask=20, time_mask=20, num_masks=2
        )

        self.cnn = nn.Sequential(
            CNNBlock(1,   32),
            CNNBlock(32,  64),
            CNNBlock(64, 128),
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.gru = nn.GRU(
            input_size  = 128 * 4,   # 512
            hidden_size = 128,
            num_layers  = 1,
            batch_first = True
        )
        self.gru_dropout = nn.Dropout(dropout * 0.6)

        # C.7: Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

        # Classifier: 128 + 80 + 2 = 210
        self.classifier = nn.Sequential(
            nn.Linear(210, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, 1)
        )

    def forward(self, logmel, mfcc, f0):
        b = logmel.size(0)

        # SpecAugment (training only)
        x = self.spec_augment(logmel)

        # CNN
        x = self.cnn(x)
        # (b, 128, 16, 16)

        # Spatial pooling
        x = self.spatial_pool(x)
        # (b, 128, 4, 4)

        # Reshape for GRU
        x = x.permute(0, 2, 1, 3).reshape(b, 4, 128 * 4)
        # (b, 4, 512)

        # GRU
        out, _ = self.gru(x)
        emb    = self.gru_dropout(out[:, -1, :])
        # (b, 128)

        # Adaptive late fusion (C.7)
        fused = torch.cat([
            self.alpha * emb,
            self.beta  * mfcc,
            self.gamma * f0
        ], dim=1)
        # (b, 210)

        return self.classifier(fused)

    def get_fusion_weights(self):
        return {
            "alpha (CNN-GRU)": self.alpha.item(),
            "beta  (MFCC)   ": self.beta.item(),
            "gamma (F0)     ": self.gamma.item()
        }



if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNGRU(dropout=0.5).to(device)

    logmel = torch.randn(4, 1, 128, 128).to(device)
    mfcc   = torch.randn(4, 80).to(device)
    f0     = torch.randn(4, 2).to(device)
    out    = model(logmel, mfcc, f0)

    total  = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {list(out.shape)}")
    print(f"Parameters   : {total:,}")
    print(f"Device       : {device}")
    print("\n✅ model.py OK")