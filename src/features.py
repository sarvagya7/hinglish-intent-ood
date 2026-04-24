import os
import subprocess
import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, WavLMModel
from tqdm import tqdm  # This gives us nice progress bars!

def get_device():
    """Returns the available hardware device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_features(texts, output_path="results/text_feats.npy"):
    """
    Extracts 768-dimensional MuRIL embeddings for a list of texts.
    """
    device = get_device()
    print(f"Using device: {device} for Text Extraction")
    print("Loading MuRIL backbone...")
    
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    text_model = AutoModel.from_pretrained("google/muril-base-cased").to(device)
    text_model.eval()

    text_feats = []
    
    # tqdm wraps the list to show a loading bar in the terminal
    for text in tqdm(texts, desc="Processing MuRIL text"):
        inputs = tokenizer(
            str(text), 
            return_tensors="pt",
            truncation=True, 
            max_length=128,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            out = text_model(**inputs)
            
        # Extract the [CLS] token representation
        feat = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        text_feats.append(feat)

    text_feats = np.array(text_feats)
    
    # Create the folder if it doesn't exist, then save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, text_feats)
    
    print(f"Text features saved to {output_path} | Shape: {text_feats.shape}\n")
    return text_feats

def extract_audio_features(video_ids, clip_dir="data/Scenes/Scenes/", output_path="results/audio_feats.npy"):
    """
    Extracts 768-dimensional WavLM embeddings from .mp4 files using FFmpeg.
    """
    device = get_device()
    print(f"Using device: {device} for Audio Extraction")
    print("Loading WavLM backbone...")
    
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)
    wavlm.eval()
    
    audio_feats = []
    failed = []

    for i, vid_id in enumerate(tqdm(video_ids, desc="Processing WavLM audio")):
        mp4_path = os.path.join(clip_dir, f"{vid_id}.mp4")
        wav_path = f"temp_audio_{i}.wav"
        
        # 1. Extract audio from mp4 using FFmpeg
        subprocess.run([
            "ffmpeg", "-i", mp4_path,
            "-ac", "1", "-ar", "16000",
            "-vn", wav_path,
            "-y", "-loglevel", "error"
        ])
        
        # 2. Process with WavLM
        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            
            inputs = processor(
                wav.squeeze().numpy(), 
                sampling_rate=16000,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                out = wavlm(**inputs)
                
            # Mean pooling over the sequence length
            feat = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            audio_feats.append(feat)
            
            # Clean up temp file immediately
            if os.path.exists(wav_path):
                os.remove(wav_path)
                
        except Exception as e:
            failed.append(vid_id)
            # Append zero array so dimensions match the dataframe later
            audio_feats.append(np.zeros(768))

    audio_feats = np.array(audio_feats)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, audio_feats)
    
    print(f"Audio features saved to {output_path} | Shape: {audio_feats.shape}")
    if failed:
        print(f"WARNING: Failed to process {len(failed)} clips. Check paths or FFmpeg installation.")
        
    return audio_feats
