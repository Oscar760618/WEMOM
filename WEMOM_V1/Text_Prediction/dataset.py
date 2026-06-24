import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer

data_path = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Text_dataset/text_VA.csv'

def get_data():
    '''
    Reads the CSV data and returns lists of texts, valence, and arousal.
    '''
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return [], [], []
    
    # Simple check for relevant columns
    if 'sentence' in df.columns:
        text_col = 'sentence'
    elif 'text' in df.columns:
        text_col = 'text'
    else:
        raise ValueError("Column 'sentence' or 'text' not found")
        
    # Drop NaNs
    df = df.dropna(subset=[text_col, 'Valence', 'Arousal'])
    
    # Convert to list
    texts = df[text_col].astype(str).tolist()
    valence = df['Valence'].tolist()
    arousal = df['Arousal'].tolist()
    
    return texts, valence, arousal


class TextPredictionDataset(Dataset):
    def __init__(self, texts, labels_v, labels_a, max_len=128, mode="train"):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_len = max_len
        self.mode = mode
        
        # Consistent splitting logic
        if mode in ["train", "val"]:
            X_train, X_val, y_rv_train, y_rv_val, y_ra_train, y_ra_val = train_test_split(
                texts, labels_v, labels_a, test_size=0.1, random_state=42
            )
            
            if mode == "train":
                self.texts = X_train
                self.labels_v = y_rv_train
                self.labels_a = y_ra_train
            else:
                self.texts = X_val
                self.labels_v = y_rv_val
                self.labels_a = y_ra_val
        else:
            # inference mode or full data
            self.texts = texts
            self.labels_v = labels_v
            self.labels_a = labels_a

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Original logic was incorrect for data range outside 1-9
        # Based on analysis: Valence min -0.31, max 10.06. Arousal min -0.7, max 10.23.
        # We will use simple scaling that covers the potential range [-2, 12] to be safe and map to roughly 0-1, 
        # but honestly for regression with MSE, pure Scaling is better.
        # Let's use a safe range normalization: x_norm = (x - min_safe) / (max_safe - min_safe)
        # Safe range: [-2, 12] (span 14). 
        #   -0.31 -> (-0.31 - (-2))/14 = 0.12
        #    5.0  -> (5.0 - (-2))/14 = 0.5
        #   10.23 -> (10.23 - (-2))/14 = 0.87
        
        min_safe = -2.0
        max_range = 14.0 # 12 - (-2) = 14
        
        label_v = (float(self.labels_v[idx]) - min_safe) / max_range
        label_a = (float(self.labels_a[idx]) - min_safe) / max_range

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor([label_v, label_a], dtype=torch.float)
        }

if __name__ == "__main__":
    texts, valence, arousal = get_data()
    if len(texts) > 0:
        print("Sample text:", texts[0])
        print(f"Sample V: {valence[0]}, A: {arousal[0]}")

        train_ds = TextPredictionDataset(texts, valence, arousal, mode="train")
        val_ds = TextPredictionDataset(texts, valence, arousal, mode="val")

        print("Train dataset size:", len(train_ds))
        print("Test dataset size:", len(val_ds))
        
        # Verify shuffling and shapes
        sample = train_ds[0]
        print("Input IDs shape:", sample['input_ids'].shape)
        print("Labels:", sample['labels'])
    else:
        print("No data found.")
    