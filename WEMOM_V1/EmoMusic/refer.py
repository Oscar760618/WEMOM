import torch
import numpy as np
from MusicVAE import MusicAttrRegGMVAE
from music_dataset import VGMIDIDataset, get_vgmidi
import json
from torch.utils.data import DataLoader

checkpoint_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/params/music_attr_vae_reg_gmm_long_v_100.pt'
with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/MusicVAE.json') as f:
    args = json.load(f)

EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
CHROMA_DIMS = 24

model = MusicAttrRegGMVAE(
    roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
    chroma_dims=CHROMA_DIMS,
    hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
    n_step=args['time_step'],
    n_component=args['num_clusters']
)
model.eval()
if torch.cuda.is_available():
    model.cuda()

checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])

data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst, label_lst = get_vgmidi(mode="CL")
music_dataset = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, chroma_lst, arousal_lst, valence_lst, label_lst, mode="ref")
music_loader = DataLoader(music_dataset, batch_size=1, shuffle=False)

all_z_r = []
all_z_n = []
with torch.no_grad():
    for i, x in enumerate(music_loader):
        d, r, n, c, a, v, l, r_density, n_density = x
        d, r, n, c = d.cuda().long(), r.cuda().long(), n.cuda().long(), c.cuda().float()
       
        def convert_to_one_hot(input, dims):
            if len(input.shape) > 1:
                input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).cuda()
                input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
            else:
                input_oh = torch.zeros((input.shape[0], dims)).cuda()
                input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
            return input_oh
        d_oh = convert_to_one_hot(d, EVENT_DIMS)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
        n_oh = convert_to_one_hot(n, NOTE_DIMS)
       
        res = model(d_oh, r_oh, n_oh, c)
        _, _, z_out, _, _, _ = res
        z_r, z_n = z_out 
        all_z_r.append(z_r.cpu().numpy())
        all_z_n.append(z_n.cpu().numpy())

all_z_r = np.concatenate(all_z_r, axis=0)
all_z_n = np.concatenate(all_z_n, axis=0)

music_features = np.concatenate([all_z_r, all_z_n], axis=1)
np.save('music_features.npy', music_features)
print("Latent space shape:", music_features.shape)
print("Saved as music_features.npy")