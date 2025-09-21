import torch
import numpy as np
from model import CLloss

checkpoint_path = "D:/PolyU/URIS/Part2_projects/WEMOM_V1/Text2Music/params/params_10000.pt"
text_latent_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/text_z_single2.npy'


def text_to_music_latent(text_latent, checkpoint_path, save_path):
    print("text_latent numpy shape:", text_latent.shape)
    # Ensure text_latent is 2D: [batch_size, features]
    if text_latent.ndim == 1:
        text_latent = text_latent.reshape(1, -1)
    txt_dim = text_latent.shape[1]
    clnet = CLloss(txt_dim=txt_dim, mus_dim=256)
    state = torch.load(checkpoint_path)
    clnet.load_state_dict(state['model_state_dict'])
    clnet.cuda()
    clnet.eval()

    pos_txt = torch.tensor(text_latent, dtype=torch.float32).cuda()
    if pos_txt.dim() == 1:
        pos_txt = pos_txt.unsqueeze(0)
    print("pos_txt torch shape:", pos_txt.shape)
    neg_txt = torch.zeros_like(pos_txt).cuda()
    pos_muse = torch.zeros((pos_txt.shape[0], 256), dtype=torch.float32).cuda()
    neg_muse = torch.zeros_like(pos_muse).cuda()

    with torch.no_grad():
        music_latent = clnet.forward(pos_txt, neg_txt, pos_muse, neg_muse, training=False)

    np.save(save_path, music_latent.cpu().numpy())
    print("Saved music latent to", save_path)

if __name__ == "__main__":
    text_latent = np.load(text_latent_path)
    text_to_music_latent(text_latent, checkpoint_path, 'music_source2.npy')