
import numpy as np
import pandas as pd
import json

# Paths
save_path = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Saves/'
data_path = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Text_Dataset/text_data.csv'

# Load original CSV to compare
df = pd.read_csv(data_path)
print(f"CSV Shape: {df.shape}")
print("First 5 CSV rows:")
print(df.head())

# Load processed data
try:
    # Try loading the VAE data used in training
    sentence_lst = np.load(save_path + 'Text_VAE_sentence_lst.npy', allow_pickle=True)
    label_lst = np.load(save_path + 'Text_VAE_label_lst.npy', allow_pickle=True)
    
    print(f"\nLoaded Numpy Arrays - Sentences: {sentence_lst.shape}, Labels: {label_lst.shape}")

    # Load Vocab to decode
    id2word = np.load(save_path + 'id2word_text.npy', allow_pickle=True).item()
    
    # Helper to decode
    def decode(ids):
        return " ".join([id2word.get(i, '<UNK>') for i in ids if i not in [0, 1, 2, 3]]) # Exclude special tokens for readability

    # Check alignment for first few items
    print("\nChecking Alignment (First 5 items):")
    for i in range(5):
        print(f"Index {i}:")
        print(f"  Processed Sentence (Decoded): {decode(sentence_lst[i])}")
        print(f"  Processed Label: {label_lst[i]}")
        
        # Calculate expected label from CSV 
        # Note: The VAE list might be a subset (first 6000), so indices should match CSV indices 0-4
        csv_row = df.iloc[i]
        val = csv_row['Valence']
        ar = csv_row['Arousal']
        
        # Re-implement label logic from data_processing
        # But wait, data_processing applies transformation first! 
        # We need to approximate or just look at raw values to see if it makes sense.
        print(f"  CSV Original: '{csv_row['sentence']}' (V={val}, A={ar})")

except Exception as e:
    print(f"Error loading files: {e}")

