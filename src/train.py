import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Import from your other files
from src.model import FusionNet
from src.ood_utils import compute_mahalanobis_parameters, score_ood, evaluate_metrics
# Note: Ensure IntentDataset is imported or defined here

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Load Features
    # ... (Your logic for np.load, masking OOD vs ID, and dropping rare classes goes here) ...
    # Let's assume you've generated X_t_tr, X_a_tr, y_tr, etc., and text_ood, audio_ood
    
    n_classes = len(np.unique(y_tr))
    
    # 2. Setup DataLoaders
    # train_dl = DataLoader(IntentDataset(X_t_tr, X_a_tr, y_tr), batch_size=16, shuffle=True)
    # test_dl  = DataLoader(IntentDataset(X_t_te, X_a_te, y_te), batch_size=16)

    # 3. Initialize Model
    model = FusionNet(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training Loop
    print("\nTraining...")
    for epoch in range(150):
        model.train()
        total_loss = 0
        for t, a, y in train_dl:
            t, a, y = t.to(device), a.to(device), y.to(device)
            logits, _ = model(t, a)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if (epoch+1) % 25 == 0:
            print(f"Epoch {epoch+1}/150  loss={total_loss/len(train_dl):.4f}")

    # 5. Extraction for Evaluation
    model.eval()
    # ... (Your logic to extract ID embeddings, OOD embeddings, and predictions goes here) ...
    
    # 6. Mahalanobis and Metrics
    # means, cov_inv = compute_mahalanobis_parameters(train_embeddings, y_tr, n_classes)
    # id_scores = score_ood(id_embs, means, cov_inv, n_classes)
    # ood_scores = score_ood(ood_embs, means, cov_inv, n_classes)
    # evaluate_metrics(trues, preds, id_scores, ood_scores)

    # 7. Save
    # torch.save(model.state_dict(), "results/fusion_model.pt")

if __name__ == "__main__":
    main()
