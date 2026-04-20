# Multimodal Intent Recognition and OOD Detection in Hinglish Talk-Show Dialogue

Adapting the MIntOOD framework for Hinglish dialogue from The Kapil Sharma Show,
using text and audio modalities with the full MIntRec2.0 label taxonomy.

## Key contribution
Unlike prior benchmarks that synthesize OOD data artificially, this dataset
contains 129 genuine human-labeled OOD samples held out entirely from training.

## Dataset
- 320 utterances from Episode 319 of The Kapil Sharma Show
- Language: Hinglish (code-switched Hindi-English)
- Labels: MIntRec2.0 taxonomy (30 intent classes)
- OOD split: 131 "Other" utterances → 129 with matching clips

> Note: Raw video clips and CSV are not included due to copyright.
> See `data/README.md` for instructions on reproducing the dataset.

## Setup
```bash
conda create -n hinglish-ood python=3.9
conda activate hinglish-ood
pip install -r requirements.txt
```

## Run
```bash
# Extract features (run once)
python extract_features.py

# Train and evaluate
sh examples/run_train.sh
```

## Results
| Modality | Intent Acc | OOD AUROC |
|---|---|---|
| Text only | - | - |
| Text + Audio | - | - |

*(fill after running)*

## Reference
Based on [MIntOOD](https://github.com/thuiar/MIntOOD) — Zhang et al., IEEE TMM 2025.
