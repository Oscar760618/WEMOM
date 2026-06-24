import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define Paths
EMOBANK_PATH = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Text_dataset/emobank.csv'
FACEBOOK_PATH = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Text_dataset/dataset-fb-valence-arousal-anon.csv'
SAVE_PATH = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Text_dataset/text_data.csv'

def process_emobank():
    print(f"Processing Emobank dataset from {EMOBANK_PATH}...")
    emobank = pd.read_csv(EMOBANK_PATH, index_col=0)
    
    emobank = emobank.drop(columns=['D', 'split'], errors='ignore')
    emobank.reset_index(drop=True, inplace=True)
    
    cols = emobank.columns.tolist()
    if len(cols) >= 3:
        cols[0], cols[1], cols[2] = cols[2], cols[0], cols[1]
        emobank = emobank[cols]

    V_min, V_max = emobank['V'].min(), emobank['V'].max()
    A_min, A_max = emobank['A'].min(), emobank['A'].max()  
    emobank['V'] = ((emobank['V'] - V_min) / (V_max - V_min)) * 10
    emobank['A'] = ((emobank['A'] - A_min) / (A_max - A_min)) * 10
    
    return emobank

def process_facebook():
    print(f"Processing Facebook dataset from {FACEBOOK_PATH}...")
    facebook = pd.read_csv(FACEBOOK_PATH, index_col=0)
    facebook.reset_index(drop=False, inplace=True)
    facebook.columns = ['text', 'Valence1', 'Valence2', 'Arousal1', 'Arousal2']

    facebook['V'] = (facebook['Valence1'] + facebook['Valence2']) / 2
    facebook['A'] = (facebook['Arousal1'] + facebook['Arousal2']) / 2
    
    facebook = facebook.drop(columns=['Valence1', 'Valence2', 'Arousal1', 'Arousal2'])

    V_min, V_max = facebook['V'].min(), facebook['V'].max()
    A_min, A_max = facebook['A'].min(), facebook['A'].max()  
    facebook['V'] = ((facebook['V'] - V_min) / (V_max - V_min)) * 10
    facebook['A'] = ((facebook['A'] - A_min) / (A_max - A_min)) * 10
    
    return facebook

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Regex cleaning
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
    text = re.sub(r'\s+', ' ', text)              # Remove extra whitespace
    text = re.sub(r'\d+', '', text)               # Remove numbers
    text = text.lower()                           # Lowercase
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    text = ' '.join(words)
    
    # Remove non-alpha
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

def preprocess_and_merge():
    # 1. Load and specific process
    df_emobank = process_emobank()
    df_facebook = process_facebook()

    # 2. Combine
    print("Merging datasets...")
    combined_df = pd.concat([df_emobank, df_facebook], ignore_index=True)
    combined_df = combined_df.round(2)
    
    # 3. Clean Text
    print("Cleaning text...")
    # Filter non-strings first
    combined_df = combined_df[combined_df['text'].apply(lambda x: isinstance(x, str))]
    
    combined_df['text'] = combined_df['text'].apply(clean_text)
    
    # 4. Final Cleanup
    # Remove empty strings resulting from cleaning
    combined_df = combined_df[combined_df['text'].str.strip() != ""]
    combined_df = combined_df.drop_duplicates()
    
    print(f"Null values:\n{combined_df.isnull().sum()}")
    print(f"Duplicates: {combined_df.duplicated().sum()}")
    print(f"Final shape: {combined_df.shape}")
    
    # Rename columns and reset index before saving
    combined_df = combined_df.rename(columns={'text': 'sentence', 'V': 'Valence', 'A': 'Arousal'})
    
    # 5. Save
    print(f"Saving to {SAVE_PATH}...")
    # Use index=False to avoid saving the index column
    combined_df.to_csv(SAVE_PATH, index=False)
    return combined_df

if __name__ == "__main__":
    preprocess_and_merge()
