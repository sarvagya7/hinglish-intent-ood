import numpy as np
import numpy.linalg as LA
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_mahalanobis_parameters(train_embeddings, train_labels, n_classes):
    """Calculates the class means and inverse covariance matrix."""
    class_embeds = {c: [] for c in range(n_classes)}
    
    for emb, label in zip(train_embeddings, train_labels):
        class_embeds[label].append(emb)

    # Calculate mean vector for each class
    means = {c: np.mean(class_embeds[c], axis=0) for c in range(n_classes)}
    
    # Calculate shared covariance matrix
    all_emb = np.vstack(list(class_embeds.values()))
    # Add small constant to diagonal for numerical stability
    cov = np.cov(all_emb.T) + np.eye(all_emb.shape[1]) * 1e-5
    cov_inv = LA.inv(cov)
    
    return means, cov_inv

def score_ood(embeddings, means, cov_inv, n_classes):
    """Calculates the minimum Mahalanobis distance across all classes."""
    scores = []
    for emb in embeddings:
        # Distance to the closest class mean
        score = min(mahalanobis(emb, means[c], cov_inv) for c in range(n_classes))
        scores.append(score)
    return scores

def evaluate_metrics(trues, preds, id_scores, ood_scores):
    """Prints Intent Accuracy and OOD AUROC."""
    acc = accuracy_score(trues, preds)
    print(f"Intent Accuracy: {acc*100:.1f}%")

    # Combine scores for AUROC (0 for ID, 1 for OOD)
    y_true_ood = [0]*len(id_scores) + [1]*len(ood_scores)
    y_scores_ood = id_scores + ood_scores
    auroc = roc_auc_score(y_true_ood, y_scores_ood)
    print(f"OOD AUROC: {auroc*100:.1f}%")
