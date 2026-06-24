import torch
import numpy as np
import json
import os
import sys
from torch.utils.data import DataLoader
from WEMOM_V1.EmoMusic.MusicVAE import MusicAttrRegGMVAE
from music_dataset import VGMIDIDataset, get_vgmidi

# Configuration
checkpoint_path = 'D:/Projects/URIS/URIS/WEMOM_V1/EmoMusic/params/MusicVAE_50.pt'
config_path = 'D:/Projects/URIS/URIS/WEMOM_V1/EmoMusic/MusicVAE.json'

# Ensure config exists
if not os.path.exists(config_path):
    print(f"Config file not found at {config_path}")
    sys.exit(1)

with open(config_path) as f:
    args = json.load(f)

# Dimensions
EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
CHROMA_DIMS = 24
DYNAMIC_DIMS = 5
CHORD_DIMS = 24

# Initialize Model
print("Initializing Model...")
model = MusicAttrRegGMVAE(
    roll_dims=EVENT_DIMS, 
    rhythm_dims=RHYTHM_DIMS, 
    note_dims=NOTE_DIMS, 
    chroma_dims=CHROMA_DIMS,
    dynamic_dims=DYNAMIC_DIMS,
    chord_dims=CHORD_DIMS,
    hidden_dims=args['hidden_dim'], 
    z_dims=args['z_dim'], 
    n_step=args['time_step'],
    n_component=args['num_clusters']
)

# Load Checkpoint
if torch.cuda.is_available():
    model.cuda()
    print('Using GPU')
else:
    print('Using CPU')

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    print(f"Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

model.eval()

# Helper function
def convert_to_one_hot(input, dims):
    if input.dtype != torch.int64:
        input = input.long()
    
    if len(input.shape) > 1:
        input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).to(input.device)
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    else:
        input_oh = torch.zeros((input.shape[0], dims)).to(input.device)
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    return input_oh

# Load Data
print("Loading Data...")
# Using "VAE" mode to match training data distribution
data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst, dynamic_lst, chord_lst, label_lst = get_vgmidi(mode="CL")

# Create Dataset
# mode="ref" in VGMIDIDataset uses the passed arrays directly without splitting
music_dataset = VGMIDIDataset(
    data_lst, rhythm_lst, note_density_lst, chroma_lst, dynamic_lst, chord_lst, 
    arousal_lst, valence_lst, label_lst, mode="ref"
)

music_loader = DataLoader(music_dataset, batch_size=1, shuffle=False)

# Extraction Loop
all_z_r = []
all_z_n = []
all_z_d = []
all_z_ch = []
all_z_c = []

print("Extracting features...")
with torch.no_grad():
    for i, x in enumerate(music_loader):
        # Unpack 11 items
        d, r, n, c, dyn, ch, a, v, l, r_density, n_density = x
        
        # Move to device and cast
        d = d.cuda().long()
        r = r.cuda().long()
        n = n.cuda().long()
        c = c.cuda().float()
        dyn = dyn.cuda().float()
        ch = ch.cuda().float()
        a = a.cuda()
        v = v.cuda()

        # One-hot conversion
        d_oh = convert_to_one_hot(d, EVENT_DIMS)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
        n_oh = convert_to_one_hot(n, NOTE_DIMS)
        
        # Construct VA vector
        va_vec = torch.stack([a.float(), v.float()], dim=1)

        # Forward pass
        res = model(d_oh, r_oh, n_oh, c, dyn, ch, va=va_vec)
        
        # Unpack results
        # res = (output, dis, z_out, logLogit_out, qy_x_out, y_out)
        _, _, z_out, _, _, _ = res
        z_r, z_n, z_d, z_ch, z_c = z_out
        
        all_z_r.append(z_r.cpu().numpy())
        all_z_n.append(z_n.cpu().numpy())
        all_z_d.append(z_d.cpu().numpy())
        all_z_ch.append(z_ch.cpu().numpy())
        all_z_c.append(z_c.cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples")

# Concatenate
all_z_r = np.concatenate(all_z_r, axis=0)
all_z_n = np.concatenate(all_z_n, axis=0)
all_z_d = np.concatenate(all_z_d, axis=0)
all_z_ch = np.concatenate(all_z_ch, axis=0)
all_z_c = np.concatenate(all_z_c, axis=0)

# Save
music_features = np.concatenate([all_z_r, all_z_n, all_z_d, all_z_ch, all_z_c], axis=1)
save_file = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/All_Data/music_features.npy'
np.save(save_file, music_features)

print("Latent space shape:", music_features.shape)
print(f"Saved features to {save_file}")
