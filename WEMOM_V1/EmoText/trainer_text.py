
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from EmoText.TextVAEtestVersion import TextVAEtestVersion
from TextVAE import TextVAE
from text_dataset import TextDataset, get_text_data
import json
import os
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
import wandb
import numpy as np

# wandb.init(project="TextVAE", config={
#     "batch_size": 32,
#     "embedding_size": 128,
#     "vocab_size": 10000,
#     "hidden_size": 128,
#     "num_layers": 2,
#     "dropout": 0.2,
#     "lr": 0.0001,
#     "num_epochs": 50
# })

with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

id2word = np.load('D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/All_Data/id2word_text.npy', allow_pickle=True).item()

batch_size = args['batch_size']
embedding_size = args['embedding_size']
vocab_size = args['vocab_size']
hidden_size = args['hidden_size']
num_layers = args['num_layers']
dropout = args['dropout']
clip_grad_norm = args['clip_grad_norm']

save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/params/'
resume_path = save_path + args['name'] + '_latest.pt'

if not os.path.exists(save_path):
    os.makedirs(save_path)

sentence_lst, label_lst = get_text_data(mode='VAE')
print(len(sentence_lst), len(label_lst))
sentence_lst, label_lst = sentence_lst, label_lst
train_dataset = TextDataset(sentence_lst, label_lst, mode="train")
test_dataset = TextDataset(sentence_lst, label_lst, mode="val")

label_counts = Counter(label_lst)
total_count = sum(label_counts.values())
class_weights = [total_count / label_counts[i] for i in range(len(label_counts))]

train_label_counts = Counter(train_dataset.label.cpu().numpy())
val_label_counts = Counter(test_dataset.label.cpu().numpy())

print("Training set label distribution:", train_label_counts)
print("Validation set label distribution:", val_label_counts)

# wandb.log({"train_label_distribution": train_label_counts, "val_label_distribution": val_label_counts})


train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

model = TextVAE(
    vocab_size=vocab_size,
    embed_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
)

# model = TextVAEtestVersion(
#     vocab_size=vocab_size,
#     embed_size=embedding_size,
#     hidden_size=hidden_size,
#     num_classes=4,  # 或你的类别数
#     pad_idx=args["PAD_INDEX"]
# )

if torch.cuda.is_available():
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


def loss_function(logit, text_target, mean, std, logits, labels, KL_weight, class_weights=class_weights):
    criterion = nn.CrossEntropyLoss(ignore_index=args["PAD_INDEX"])
    reconstruction_loss = criterion(logit, text_target)
    kl_loss = 0.5 * (-torch.log(std ** 2) + mean ** 2 + std ** 2 - 1).mean()
    classification_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).cuda())
    classification_loss = classification_loss_fn(logits, labels)
    loss = reconstruction_loss + kl_loss * KL_weight + classification_loss * args['alpha']
    return loss, reconstruction_loss, kl_loss, classification_loss

def training_phase(step):
    print("Starting training...")
    for epoch in range(step, args['num_epochs'] + 1):
        torch.cuda.empty_cache()
        print(f"Epoch {epoch} started. Current learning rate:")
        for param_group in optimizer.param_groups:
            print(f"  {param_group['lr']}")

        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for i, (sentence, labels) in enumerate(train_data):
            
            sentence, labels = sentence.cuda(), labels.cuda()
            labels = labels.long()
            optimizer.zero_grad()

            valid_tokens = (sentence[:, 1:] != args["PAD_INDEX"]).sum().item()
            total_tokens = sentence[:, 1:].numel()
            # print(f"Batch {i}: valid tokens {valid_tokens}/{total_tokens} ({valid_tokens/total_tokens:.2%})")


            logit, mean, std, logits = model(sentence)
            target_output = sentence[:, 1:]  # Remove <sos>
            # print(logit, mean, std, logits)

            logit = logit.reshape(-1, logit.size(-1))  
            target_output = target_output.reshape(-1)

            # KL Annealing
            kl_weight = min(1.0, epoch / 10.0) * args["lambda"]  # 在前 10 个 epoch 内逐步增加 KL 权重
            loss, reconstruction_loss, kl_loss, classification_loss = loss_function(
            logit, target_output, mean, std, logits, labels, kl_weight, class_weights
            )

            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm().item()}")
            # nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            step += 1   
            

            if i % 10 == 0:
                probs = logit.softmax(dim=-1)
                # print(f"Decoder softmax mean: {probs.mean().item():.4f}, max: {probs.max().item():.4f}")
                # print(f"Logits distribution (softmax): {logits.softmax(dim=-1).mean(dim=0).tolist()}")
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}, "
                      f"Reconstruction: {reconstruction_loss.item():.4f}, "
                      f"KL: {kl_loss.item():.4f}, Classification: {classification_loss.item():.4f}")
                # print(f"Predicted labels: {logits.argmax(dim=1)[:10].tolist()}")
                # print(f"True labels: {labels[:10].tolist()}")

        train_accuracy = correct / total
        # wandb.log({"epoch": epoch, "train_loss": epoch_loss / len(train_data), "train_accuracy": train_accuracy})

        print(f"-------------------------------------------------------------------------")
        print(f"Epoch {epoch} completed. Average Loss: {epoch_loss / len(train_data):.4f}")

        # Save checkpoint
        if epoch % 5 == 0:
            print(f"Saving checkpoint at epoch {epoch}...")
            save_checkpoint(epoch, step, model, optimizer, epoch_loss)

        

        # Evaluate on validation set
        eval_loss, eval_accuracy = evaluation_phase()
        # wandb.log({"val_loss": eval_loss, "val_accuracy": eval_accuracy})
        scheduler.step(eval_loss)

        print(f"Validation Loss: {eval_loss:.4f}")

# Evaluation phase
def evaluation_phase():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence, labels in test_data:
            sentence, labels = sentence.cuda(), labels.cuda()
    
            logit, mean, std, logits = model(sentence)

            logit = logit.reshape(-1, logit.size(-1)) 
            target_output = sentence[:, 1:]  
            target_output = target_output.reshape(-1)

            loss, _, _, _ = loss_function(
                logit, target_output.reshape(-1), mean, std, logits, labels, KL_weight=1.0,class_weights=class_weights)
        
            total_loss += loss.item()
            decoder_output = logit.view(sentence.size(0), -1, logit.size(-1))
            predicted_ids = decoder_output.argmax(dim=-1)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Loss: {total_loss / len(test_data):.4f}, Accuracy: {accuracy:.4f}")
    return total_loss / len(test_data), accuracy    

# Save checkpoint
def save_checkpoint(epoch, step, model, optimizer, loss):
    save_path_epoch = save_path + f"{args['name']}_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path_epoch)
    print(f"Checkpoint saved: {save_path_epoch}")

    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, resume_path)
    print(f"Latest checkpoint saved: {resume_path}")

# Resume training if checkpoint exists
# if os.path.exists(resume_path):
#     print(f"Resuming training from {resume_path}...")
#     checkpoint = torch.load(resume_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     step = checkpoint.get('step', 0)
#     print(f"Resumed from epoch {start_epoch}")
# else:
#     print("No checkpoint found. Starting training from scratch.")
#     start_epoch = 1
#     step = 0

# Start training
step = 0
training_phase(step)
