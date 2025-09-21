import torch
from torch.utils.data import DataLoader

import numpy as np
import json

from TextVAEtestVersion import TextVAEtestVersion
from TextVAE import TextVAE
from text_dataset import TextDataset, get_text_data

checkpoint_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/params/TextVAE_epoch_50.pt'
with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

def make_latent_feature():
    model = TextVAE(
        vocab_size=args['vocab_size'],
        embed_size=args['embedding_size'],
        hidden_size=args['hidden_size'],
        num_layers=args['num_layers'],
        dropout=args['dropout']
    )

    sentence_lst, label_lst = get_text_data(mode="CL")
    text_dataset = TextDataset(sentence_lst, label_lst, mode="ref")
    text_loader = DataLoader(text_dataset, batch_size=1, shuffle=False)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict']) 

    all_z = []
    with torch.no_grad():
        for i, (sentence, labels) in enumerate(text_loader):

            sentence = sentence.cuda()
            encoding, mean, std = model.encode(sentence[:, :-1].cuda())
            all_z.append(encoding.cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)

    np.save('text_z.npy', all_z)
    print("Latent space z shape:", all_z.shape)
    print("Saved as text_z.npy")

if __name__ == "__main__":
    make_latent_feature()