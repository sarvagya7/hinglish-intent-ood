import os
import pandas as pd
from torch.utils.data import Dataset

def load_and_clean_data(base_dir="data/"):
    """Loads the transcript and filters out missing video clips."""
    clip_dir = os.path.join(base_dir, "Scenes/Scenes/")
    csv_path = os.path.join(base_dir, "Dataset - Transcript.csv")
    
    # Load and clean
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.strip().str.lower()

    # Drop missing clips
    df["clip_exists"] = df["Video ID"].apply(
        lambda x: os.path.exists(os.path.join(clip_dir, f"{x}.mp4"))
    )
    df = df[df["clip_exists"]].reset_index(drop=True)
    
    print(f"Dataset ready: {len(df)} rows loaded.")
    return df

class HinglishDataset(Dataset):
    """PyTorch Dataset for Hinglish Intent Recognition."""
    def __init__(self, base_dir="data/"):
        self.data_frame = load_and_clean_data(base_dir)
        self.clip_dir = os.path.join(base_dir, "Scenes/Scenes/")
        # ... Add your tokenizer or audio feature extractor initialization here ...

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # ... Add your logic to return the text and audio features for a specific row ...
        pass
