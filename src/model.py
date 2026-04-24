import torch
import torch.nn as nn

class FusionNet(nn.Module):
    """
    Weighted Multimodal Fusion Network for Text and Audio embeddings.
    """
    def __init__(self, n_classes, dim=768, hidden=256):
        super().__init__()
        # Encoders for each modality
        self.enc_t = nn.Sequential(
            nn.Linear(dim, hidden), 
            nn.ReLU(), 
            nn.Dropout(0.3)
        )
        self.enc_a = nn.Sequential(
            nn.Linear(dim, hidden), 
            nn.ReLU(), 
            nn.Dropout(0.3)
        )
        
        # Attention/Weighting layers
        self.w_t = nn.Linear(hidden, 1)
        self.w_a = nn.Linear(hidden, 1)
        
        # Final classification head
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, t, a):
        ht = self.enc_t(t)
        ha = self.enc_a(a)
        
        # Calculate dynamic weights for text vs audio
        weights = torch.softmax(
            torch.stack([self.w_t(ht), self.w_a(ha)], dim=1), dim=1
        )
        
        # Fuse the representations
        fused = weights[:,0]*ht + weights[:,1]*ha
        
        # Return both the logits (for classification) and the fused embedding (for OOD detection)
        return self.classifier(fused), fused
