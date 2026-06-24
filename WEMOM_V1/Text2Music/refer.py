import torch
import numpy as np
import os
import sys
from model import CLloss
import config

# Use paths from config or define them relative to project
checkpoint_path = "D:/Projects/URIS/URIS/WEMOM_V1/Text2Music/params/params_1000_6.167.pt"
# Assuming text_z.npy is in Data folder as per config
text_latent_path = "D:/Projects/URIS/URIS/WEMOM_V1/Data/All_Data/test_single_text.npy" 
save_path = "D:/Projects/URIS/URIS/WEMOM_V1/Data/All_Data/test_single_music.npy" 
def text_to_music_latent(text_latent, checkpoint_path, save_path):
    print("text_latent numpy shape:", text_latent.shape)
    # Ensure text_latent is 2D: [batch_size, features]
    if text_latent.ndim == 1:
        text_latent = text_latent.reshape(1, -1)
    
    txt_dim = text_latent.shape[1]
    # Use config.MUS_DIM (640) instead of hardcoded 256
    clnet = CLloss(txt_dim=txt_dim, mus_dim=config.MUS_DIM)
    
    if torch.cuda.is_available():
        state = torch.load(checkpoint_path)
        clnet.cuda()
    else:
        state = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in state:
        clnet.load_state_dict(state['model_state_dict'])
    else:
        clnet.load_state_dict(state)
        
    clnet.eval()

    if torch.cuda.is_available():
        pos_txt = torch.tensor(text_latent, dtype=torch.float32).cuda()
    else:
        pos_txt = torch.tensor(text_latent, dtype=torch.float32)

    if pos_txt.dim() == 1:
        pos_txt = pos_txt.unsqueeze(0)
    print("pos_txt torch shape:", pos_txt.shape)
    
    # Dummy inputs for forward pass
    if torch.cuda.is_available():
        neg_txt = torch.zeros_like(pos_txt).cuda()
        pos_muse = torch.zeros((pos_txt.shape[0], config.MUS_DIM), dtype=torch.float32).cuda()
        neg_muse = torch.zeros_like(pos_muse).cuda()
    else:
        neg_txt = torch.zeros_like(pos_txt)
        pos_muse = torch.zeros((pos_txt.shape[0], config.MUS_DIM), dtype=torch.float32)
        neg_muse = torch.zeros_like(pos_muse)

    with torch.no_grad():
        # training=False returns pro_muse
        # Debug: Check intermediate outputs
        pos_txt_emb1 = clnet.txtpro1(pos_txt)
        print(f"DEBUG: emb1 mean={pos_txt_emb1.mean():.4f} std={pos_txt_emb1.std():.4f}")
        pos_txt_emb2 = clnet.txtpro2(pos_txt_emb1)
        print(f"DEBUG: emb2 mean={pos_txt_emb2.mean():.4f} std={pos_txt_emb2.std():.4f}")
        pos_txt_emb3 = clnet.txtpro3(pos_txt_emb2)
        print(f"DEBUG: emb3 mean={pos_txt_emb3.mean():.4f} std={pos_txt_emb3.std():.4f}")
        
        music_latent = clnet.demuspro(pos_txt_emb3)
        print(f"DEBUG: music_latent mean={music_latent.mean():.4f} std={music_latent.std():.4f}")

    # Handle potential dimension squeeze from model.py's [0] indexing

    # Handle potential dimension squeeze from model.py's [0] indexing
    if music_latent.dim() == 1:
        music_latent = music_latent.unsqueeze(0)

    np.save(save_path, music_latent.cpu().numpy())
    print("Saved music latent to", save_path)
    print("Music latent shape:", music_latent.shape)

if __name__ == "__main__":
    # Example usage with a single vector or loading from file
    # For testing, we can create a dummy vector if file doesn't exist
    if os.path.exists(text_latent_path):
        text_latent = np.load(text_latent_path)
        # Take just one for demo if it's a large dataset
        if text_latent.shape[0] > 1:
             text_latent = text_latent[0]
    else:
        print(f"Warning: {text_latent_path} not found. Using dummy data.")
        text_latent = np.random.randn(128).astype(np.float32)

    text_to_music_latent(text_latent, checkpoint_path, save_path)