class Config:
    # data
    csv_path = "data/Dataset - Transcript.csv"
    clip_dir = "data/clips/"
    text_col = " Hinglish Text"    # note the leading space from your CSV
    label_col = "Label"
    ood_label = "other"

    # model
    text_backbone = "google/muril-base-cased"
    audio_backbone = "microsoft/wavlm-base"
    hidden_dim = 256
    n_epochs = 30
    batch_size = 16
    lr = 2e-4

    # OOD
    ood_threshold_percentile = 95
