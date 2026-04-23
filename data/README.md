# Data Setup

The raw data is not included in this repository due to copyright 
restrictions on The Kapil Sharma Show (Sony Entertainment).

## Expected folder structure

Place your files as follows before running the notebook:

```
data/
├── Dataset - Transcript.csv
└── clips/
    ├── Ep319_s1_dia1.mp4
    ├── Ep319_s1_dia2.mp4
    └── ...
```
## CSV format

| Column | Description |
|---|---|
| Video ID | Matches clip filename (without .mp4) |
| Hinglish Text | Code-switched Hindi-English utterance |
| Hindi Text | Hindi translation |
| Label | MIntRec2.0 intent class or "Other" |
| Start Time | Clip start in source episode |
| End Time | Clip end in source episode |

## Label distribution

- 18 named intent classes after cleaning  
- 128 "Other" utterances used as OOD test set  
- 5 clips missing from source, dropped from dataset  
- Final: 324 usable utterances
