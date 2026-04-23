# Multimodal Intent Recognition and OOD Detection in Hinglish Talk-Show Dialogue

Adapting the [MIntOOD](https://github.com/thuiar/MIntOOD) framework for Hinglish 
dialogue from The Kapil Sharma Show, using text and audio modalities with the 
MIntRec2.0 intent label taxonomy.

## Key Contribution

Unlike prior benchmarks that synthesize OOD data artificially using Dirichlet 
mixup, this dataset contains **128 genuine human-labeled OOD samples** held out 
entirely from training — providing a realistic evaluation setting.

To our knowledge, this is the first multimodal intent recognition and OOD 
detection system applied to Hinglish entertainment media.

## Dataset

| Property | Value |
|---|---|
| Source | The Kapil Sharma Show, Episode 319 |
| Language | Hinglish (code-switched Hindi-English) |
| Total utterances | 324 |
| Intent classes | 18 (MIntRec2.0 taxonomy) |
| OOD samples | 128 genuine "Other" utterances |
| Modalities | Text + Audio (.mp4 clips) |

> Raw video clips and CSV are not included due to copyright restrictions 
> on the source material. See `data/README.md` for setup instructions.

## Method

- **Text backbone**: [MuRIL](https://huggingface.co/google/muril-base-cased) 
  (Multilingual Representations for Indian Languages)
- **Audio backbone**: [WavLM](https://huggingface.co/microsoft/wavlm-base)
- **Fusion**: Weighted multimodal fusion with dynamic per-modality importance scores
- **OOD detection**: Mahalanobis distance scoring on fused embeddings

## Results

| Modality | Intent Accuracy | OOD AUROC |
|---|---|---|
| Text only (MuRIL) | 23.7% | 51.9% |
| Text + Audio (MuRIL + WavLM) | 28.9% | 56.6% |

Adding audio modality improves OOD AUROC by **+4.7%** over text-alone, 
which performs near-randomly (51.9% ≈ 50% baseline). Both accuracy and 
AUROC improve with multimodal fusion, confirming that non-verbal cues 
carry meaningful intent signal in Hinglish entertainment dialogue.

## Setup

```bash
pip install -r requirements.txt
```

## Notebook

Full pipeline available on Kaggle:  
[kaggle.com/code/kromniscient/capstone](https://www.kaggle.com/code/kromniscient/capstone)

Includes:
- MuRIL text feature extraction
- WavLM audio feature extraction  
- Weighted fusion model training
- Mahalanobis OOD scoring
- Ablation visualisations

## Reference

Based on [thuiar/MIntOOD](https://github.com/thuiar/MIntOOD)

```bibtex
@inproceedings{mintood2024,
  title={MIntOOD: Multimodal Intent Detection with Out-of-Scope Detection},
  author={Zhang et al.},
  booktitle={IEEE TMM},
  year={2025}
}
```
