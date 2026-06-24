import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaConfig
from Roberta import RobertaForSequenceClassificationSig
from dataset import TextPredictionDataset, get_data
import os
import numpy as np
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys

# Logger class to redirect output to both terminal and file
class Logger(object):
    def __init__(self, filename='training_log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False # mimicking a file output, disables colors to avoid error

# Redirect print to logger
sys.stdout = Logger('D:/Projects/URIS/URIS/WEMOM_V1/Text_Prediction/training_log.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

with open('D:/Projects/URIS/URIS/WEMOM_V1/Text_Prediction/Roberta_config.json') as f:
    args = json.load(f)

# Metrics calculation
def compute_metrics(predictions, labels):
    # Reverse scaling based on new dataset logic
    min_safe = -2.0
    max_range = 14.0
    
    np_preds = np.array(predictions) * max_range + min_safe
    np_labels = np.array(labels) * max_range + min_safe

    # Valence (Index 0)
    mse_valence = mean_squared_error(np_labels[:,0], np_preds[:,0])
    mae_valence = mean_absolute_error(np_labels[:,0], np_preds[:,0])   
    pearson_corr_valence = stats.pearsonr(np_labels[:,0], np_preds[:,0])[0] 
    
    # Arousal (Index 1)
    mse_arousal = mean_squared_error(np_labels[:,1], np_preds[:,1])
    mae_arousal = mean_absolute_error(np_labels[:,1], np_preds[:,1])   
    pearson_corr_arousal = stats.pearsonr(np_labels[:,1], np_preds[:,1])[0] 

    # Check spread of predictions to detect "Mean Collapse"
    pred_std_val = np.std(np_preds[:, 0])
    pred_std_aro = np.std(np_preds[:, 1])

    print('\n')
    print("mse_valence : " + str(mse_valence) + '\n' + 
            "mae_valence : " + str(mae_valence) + '\n' + 
            "pearson_corr_valence : " + str(pearson_corr_valence) + '\n' + 
            "mse_arousal : " + str(mse_arousal) + '\n' + 
            "mae_arousal : " + str(mae_arousal) + '\n' + 
            "pearson_corr_arousal : " + str(pearson_corr_arousal))
    
    print(f"DEBUG - Prediction Std Dev (Valence): {pred_std_val:.4f}")
    print(f"DEBUG - Prediction Std Dev (Arousal): {pred_std_aro:.4f}")
    if pred_std_val < 0.1 or pred_std_aro < 0.1:
        print("WARNING: Model is predicting nearly constant values (Mean Collapse).")
    
    print('\n')

    return {
        "mse_valence": mse_valence, 
        "mae_valence": mae_valence,
        "pearson_corr_valence": pearson_corr_valence,
        "mse_arousal": mse_arousal, 
        "mae_arousal": mae_arousal,
        "pearson_corr_arousal": pearson_corr_arousal
    }

# Load Data
print("Loading data...")

texts, valence, arousal = get_data()
print(f"Loaded {len(texts)} samples.")

train_dataset = TextPredictionDataset(texts, valence, arousal, mode="train")
val_dataset = TextPredictionDataset(texts, valence, arousal, mode="val")

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

config = RobertaConfig.from_pretrained('roberta-base', num_labels=2)
# Increase dropout to prevent overfitting
config.hidden_dropout_prob = args.get('hidden_dropout_prob', 0.1)
config.attention_probs_dropout_prob = args.get('attention_probs_dropout_prob', 0.1)

def save_checkpoint(epoch, step, model, optimizer, loss):
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    save_path_epoch = os.path.join(args['save_path'], f"{args['name']}_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path_epoch)
    print(f"Checkpoint saved: {save_path_epoch}")

    # Save latest checkpoint for resuming training
    resume_path = os.path.join(args['save_path'], f"{args['name']}_latest.pt")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, resume_path)
    print(f"Latest checkpoint saved: {resume_path}")

    # Save as HuggingFace pretrained format for easy inference in app.py
    hf_save_path = os.path.join(args['save_path'], f"{args['name']}_hf_best")
    if not os.path.exists(hf_save_path):
        os.makedirs(hf_save_path)
    
    # Save model and tokenizer (we need to access tokenizer from dataset or global)
    model.save_pretrained(hf_save_path)
    # Note: Tokenizer is defined inside Dataset class, ideally should be saved too.
    # We will instantiate a new one to save, to ensure vocab matches base.
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained(hf_save_path)
    
    print(f"HuggingFace format model saved to: {hf_save_path}")

def evaluation_phase(model_path=None, model=None):
    if model_path:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path} for evaluation...")
            # If loading from path, we create a new local model instance
            model = RobertaForSequenceClassificationSig.from_pretrained(model_path, num_labels=2)
            model.to(device)
        else:
            print(f"Model path {model_path} not found.")
            return None, None
    elif model is None:
        print("Error: No model and no model path provided for evaluation.")
        return None, None

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    print("Starting evaluation...")
    with torch.no_grad():
        criterion = nn.MSELoss()
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # DEBUG: Print first batch of validation predictions
            if len(all_preds) <= args['batch_size']:
                print(f"[DEBUG-VAL] Sample Val Label: {labels[0].cpu().numpy()} | Prediction: {logits[0].cpu().numpy()}")

    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(all_preds, all_labels)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss, metrics

def training_phase(step=0):
    # Model Setup

    model = RobertaForSequenceClassificationSig.from_pretrained('roberta-base', config=config)
    
    # --- PARTIAL FREEZING (Conservative) ---
    # Freeze the bottom 9 layers, Train the top 3 layers + Pooler + Classifier
    # This prevents the model from forgetting generalized features while learning tasks specific ones
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 3 encoder layers
    for layer in model.roberta.encoder.layer[-3:]:
        for param in layer.parameters():
            param.requires_grad = True
            
    # Unfreeze Pooler and Classifier
    if hasattr(model.roberta, 'pooler') and model.roberta.pooler is not None:
        for param in model.roberta.pooler.parameters():
            param.requires_grad = True
            
    for param in model.classifier.parameters():
        param.requires_grad = True

    # --- REMOVED MANUAL BIAS INIT (Now using Sigmoid + Scaled Data) ---
    
    model.to(device)

    # --- DIFFERENTIAL LEARNING RATES ---
    # Lower LR for base layers now that we have stable freezing
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad], 'lr': 2e-5},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad], 'lr': 1e-4}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=args.get('weight_decay', 0.02))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Revert to Huber with mid-range delta to balance robustness and detail
    criterion = nn.HuberLoss(delta=0.5) 

    print("Starting training...")
    resume_path = os.path.join(args['save_path'], f"{args['name']}_latest.pt")
    start_epoch = 1

    # Resume logic (optional)
    # if os.path.exists(resume_path):
    #     print(f"Found checkpoint at {resume_path}. Resuming...")
    #     checkpoint = torch.load(resume_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     step = checkpoint.get('step', step)
    #     print(f"Resuming from epoch {start_epoch}")

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 8  # Increased patience to allow Arousal to converge

    for epoch in range(start_epoch, args['train_epochs'] + 1):
        model.train()
        epoch_loss = 0
        
        print(f"Epoch {epoch} started. Current LR: {optimizer.param_groups[0]['lr']}")
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # --- CUSTOM WEIGHTED LOSS ---
            # Valence is learned quickly, Arousal lags behind.
            # We split the loss and weight Arousal higher to force learning.
            loss_v = criterion(logits[:, 0], labels[:, 0])
            loss_a = criterion(logits[:, 1], labels[:, 1])
            
            # Weight: Valence 1.0, Arousal 2.0
            loss = loss_v + 2.0 * loss_a
            
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad_norm'])
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            
            # if i % 10 == 0:
            #     print(f"[Epoch {epoch}, Step {i}] Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_train_loss:.4f}")

        # Evaluate
        eval_loss, eval_metrics = evaluation_phase(model=model)
        
        # Scheduler Step CANCELLED HERE - MOVED TO BOTTOM
        # scheduler.step(eval_loss)


        # Early Stopping & Best Model Saving
        # Change: Prioritize Arousal improvement (since Valence is stable)
        current_combined_metric = eval_metrics['mse_arousal'] + 0.5 * eval_metrics['mse_valence']
        
        if current_combined_metric < best_val_loss:
            best_val_loss = current_combined_metric
            patience_counter = 0
            save_checkpoint(epoch, step, model, optimizer, avg_train_loss)
            print(f"New best model (Combined MSE) found at epoch {epoch}. Metric: {current_combined_metric:.4f}")
        else:
            patience_counter += 1
            print(f"Combined Metric did not improve. Patience: {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print("Early stopping triggered. Training stopped.")
                break
        
        # Scheduler Step (Monitor validation loss or combined?)
        scheduler.step(eval_metrics['mse_arousal']) # Focus scheduler on Arousal loss
        
        print(f"-------------------------------------------------------------------------")

# training_phase()

if __name__ == "__main__":
    # Ensure save directory exists
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
        print(f"Created checkpoint directory: {args['save_path']}")

    # Path to the checkpoint you want to load
    # checkpoint_path = r"D:\Projects\URIS\URIS\WEMOM_V1\Text_Prediction\params\checkpoint-9630"
    
    # if os.path.exists(checkpoint_path):
    #     # Run Evaluation with the specified checkpoint
    #     loss, metrics = evaluation_phase(checkpoint_path)
    #     print("Final Evaluation Results:", metrics)
    # else:
        # print(f"Checkpoint not found at {checkpoint_path}. Starting training from scratch...")
    training_phase()
    



