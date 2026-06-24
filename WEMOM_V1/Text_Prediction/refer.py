import torch
from transformers import RobertaTokenizer, RobertaConfig
from Roberta import RobertaForSequenceClassificationSig
import os
import sys

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to the 'hf_best' folder created by trainer.py
MODEL_PATH = os.path.join(CURRENT_DIR, 'params', 'roberta_regression_hf_best')

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        # Load Config
        config = RobertaConfig.from_pretrained(model_path)
        # Load Tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        # Load Model
        model = RobertaForSequenceClassificationSig.from_pretrained(model_path, config=config)
        
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure you have trained the model and the path '{model_path}' exists.")
        return None, None

def predict_emotion(text, tokenizer, model):
    if not text:
        return None
    
    # Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()[0]
    
    # Reverse scaling (Consistent with trainer.py)
    min_safe = -2.0
    max_range = 14.0
    
    valence = float(logits[0]) * max_range + min_safe
    arousal = float(logits[1]) * max_range + min_safe
    
    # Clip to valid range just in case
    valence = max(0.0, min(10.0, valence))
    arousal = max(0.0, min(10.0, arousal))
    
    return valence, arousal

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model path {MODEL_PATH} does not exist.")
        print("Please run trainer.py first to generate the model.")
        sys.exit(1)
        
    tokenizer, model = load_model(MODEL_PATH)
    
    if model:
        print("\n--- Text Emotion Prediction (Valence/Arousal 0-10) ---")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            user_input = input("Enter text: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            val, aro = predict_emotion(user_input, tokenizer, model)
            
            print(f"Text: \"{user_input}\"")
            print(f"Predicted Valence (愉悦度): {val:.2f} / 10")
            print(f"Predicted Arousal (唤醒度): {aro:.2f} / 10")
            
            # Simple interpretation
            v_desc = "Positive" if val > 6 else "Negative" if val < 4 else "Neutral"
            a_desc = "Excited/Active" if aro > 6 else "Calm/Passive" if aro < 4 else "Moderate"
            print(f"Interpretation: {v_desc}, {a_desc}")
            print("-" * 30)
